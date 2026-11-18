import os
import re
import uuid
import socket
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# força IPv4 para evitar timeout na rede institucional
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "eja_conhecimento"
FAQ_PATH = ROOT_DIR / "data" / "faq" / "FAQ_CONSOLIDADO.md"

# extrai materia, etapa (EM/EFII) e termo(1T-4T) a partr da fonte/tema
def inferir_metadados_pedagogicos(fonte: str, tema: str) -> dict:

    texto_busca = f"{fonte} {tema}".upper()

    # etapa EJA
    if "EFII" in texto_busca or "9EF" in texto_busca:
        etapa = "EFII"
    elif "EM" in texto_busca:
        etapa = "EM"
    else:
        etapa = "GERAL"

    # termo
    match_termo = re.search(r"\b([1-4]T)\b", texto_busca)
    termo = match_termo.group(1) if match_termo else "GERAL"

    # Matéria
    if any(k in texto_busca for k in ["_MAT_", "GEOMETRIA", "EQUAÇÕES", "EQUACOES", "SEQUÊNCIAS", "SEQUENCIAS", "FINANCEIRA", "NÚMEROS", "NUMEROS", "PERIODICIDADE"]):
        materia = "Matemática"
    elif any(k in texto_busca for k in ["_CIE_", "_CI_", "_CN_", "RADIAÇÕES", "RADIACOES", "ELETRICIDADE", "ENERGIA", "SOLO", "ÓPTICA", "OPTICA", "DNA", "UNIVERSO", "SERES VIVOS", "CIENTÍFICO"]):
        materia = "Ciências"
    elif any(k in texto_busca for k in ["_LING_", "_LP_", "_LI_", "LINGUAGENS", "DIVULGA CIÊNCIA", "DESENHO", "ARTE", "RÁDIO", "RADIO", "MÚSICA", "MUSICA", "POESIA"]):
        materia = "Linguagens"
    elif any(k in texto_busca for k in ["_CH_", "_GEO_", "HISTÓRIA", "HISTORIA", "REALIDADES", "TERRITORIAL", "COMUNIDADES", "ÁSIA", "ASIA", "POPULAÇÃO", "POPULACAO", "DESNATURALIZAÇÃO", "DESNATURALIZACAO", "MIGRAÇÃO", "MIGRACAO"]):
        materia = "Ciências Humanas"
    else:
        materia = "Multidisciplinar"

    return {"materia": materia, "etapa_eja": etapa, "termo": termo}

# le o FAQ_CONSOLIDADO e transforma cada bloco em um dicionario
def parse_markdown_faqs(filepath: Path) -> list[dict]:
    
    conteudo = filepath.read_text(encoding="utf-8")
    blocos = re.split(r"\n##\s+", conteudo)
    faqs = []

    for bloco in blocos[1:]:  
        linhas = bloco.strip().splitlines()
        pergunta = linhas[0].strip()

        resposta_linhas = []
        meta = {}
        for linha in linhas[1:]:
            linha_limpa = linha.strip()
            if not linha_limpa or linha_limpa.startswith("# Documento"):
                continue
            match_meta = re.match(r"^\*\*(.+?):\*\*\s*(.+)$", linha_limpa)
            if match_meta:
                chave = match_meta.group(1).strip().lower()
                valor = match_meta.group(2).strip()
                meta[chave] = valor
            else:
                resposta_linhas.append(linha_limpa)

        resposta = " ".join(resposta_linhas).strip()
        id_faq = meta.get("id", "")
        tema = meta.get("tema", "")
        fonte = meta.get("fonte", "")
        pdf = meta.get("pdf", "")
        paginas = meta.get("páginas do pdf / slides", "")
        origem = meta.get("origem", "")

        if not id_faq or not pergunta or not resposta:
            continue

        pedagogico = inferir_metadados_pedagogicos(fonte, tema)

        faqs.append({
            "id_faq": id_faq,
            "pergunta": pergunta,
            "resposta": resposta,
            "tema": tema,
            "materia": pedagogico["materia"],
            "etapa_eja": pedagogico["etapa_eja"],
            "termo": pedagogico["termo"],
            "fonte": fonte,
            "pdf": pdf,
            "paginas": paginas,
            "origem": origem,
        })

    return faqs

def main():
    print(f"Lendo FAQs de: {FAQ_PATH}")
    faqs = parse_markdown_faqs(FAQ_PATH)
    print(f"Total de FAQs extraídas: {len(faqs)}")

    print("Carregando modelo MiniLM (paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # prep textos combinados (tema + pergunta + resposta) p vetorizaçao
    textos_para_vetorizar = [
        f"Tema: {f['tema']} | Pergunta: {f['pergunta']} | Resposta: {f['resposta']}"
        for f in faqs
    ]

    print("Gerando embeddings de 384 dimensões...")
    vetores = model.encode(textos_para_vetorizar, show_progress_bar=True)

    # monta os pontos com UUID det baseado no id_faq (p evitar duplicatas no upsert)
    pontos = []
    for faq, vetor in zip(faqs, vetores):
        ponto_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, faq["id_faq"]))
        pontos.append(
            PointStruct(
                id=ponto_id,
                vector=vetor.tolist(),
                payload=faq
            )
        )

    print("Conectando ao Qdrant Cloud e enviando pontos...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        port=443,
        timeout=60,
        prefer_grpc=False,
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=pontos,
        wait=True
    )

    print(f"\nSucesso! {len(pontos)} FAQs inseridas/atualizadas na collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()