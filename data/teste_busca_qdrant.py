import os
import socket
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# força ipv4
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    port=443,
    timeout=60,
    prefer_grpc=False,
)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def buscar_conhecimento(pergunta: str, materia: str = None, top_k: int = 3):
    vetor_pergunta = model.encode(pergunta).tolist()
    
    filtro = None
    if materia:
        filtro = Filter(
            must=[
                FieldCondition(key="materia", match=MatchValue(value=materia))
            ]
        )

    resultados = client.query_points(
        collection_name="eja_conhecimento",
        query=vetor_pergunta,
        query_filter=filtro,
        limit=top_k,
        with_payload=True,
    ).points

    print(f"\n=== Pergunta: '{pergunta}' (Filtro matéria: {materia}) ===")
    for i, hit in enumerate(resultados, 1):
        p = hit.payload
        print(f"{i}. [Score: {hit.score:.3f}] [{p.get('materia')} | {p.get('etapa_eja')} | {p.get('termo')}]")
        print(f"   Pergunta FAQ: {p.get('pergunta')}")
        print(f"   Resposta: {p.get('resposta')[:160]}...")
        print(f"   Fonte: {p.get('fonte')}")

if __name__ == "__main__":
    buscar_conhecimento("Como calcular juros simples ou compostos no dia a dia?")
    buscar_conhecimento("O que são ondas eletromagnéticas e radiação?", materia="Ciências")