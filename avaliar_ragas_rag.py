"""
Avaliação do Calma AI com RAGAS — gabarito alinhado + filtro de relevância.

Modos:
  --modo rag   (padrão)  → avalia só a camada de RAG (rápido, ~25 min)
  --modo crew            → avalia o tutor completo (lento, ~40 min)

Uso:
  uv run python avaliar_ragas_rag.py --sanity
  uv run python avaliar_ragas_rag.py --so-dataset --max-perguntas 3
  uv run python avaliar_ragas_rag.py
  uv run python avaliar_ragas_rag.py --modo crew --max-perguntas 3
  uv run python avaliar_ragas_rag.py --limpar

Saídas:
  dataset_ragas_real.json
  resultados_ragas_real.csv
  log_avaliacao_real.txt
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIMILARIDADE_MINIMA = 0.30


# =============================================================================
# GABARITO — 30 perguntas alinhadas a conteudos.json (18) + FAQ_CONSOLIDADO.md (12)
# =============================================================================
GABARITO = {
    "question": [
        # --- conteudos.json: porcentagem (3) ---
        "A conta de luz era R$ 110 e teve aumento de 20%. Qual foi o valor do aumento?",
        "Uma loja dá 15% de desconto em uma geladeira de R$ 2.400. Quanto você economiza?",
        "O salário de R$ 1.800 teve reajuste de 5%. Qual é o novo salário?",

        # --- conteudos.json: regra_de_tres (3) — com keyword explícita ---
        "Em uma receita proporcional, com 2 kg de farinha você faz 40 pães. Quantos pães faz com 5 kg?",
        "Em uma proporção inversa, 4 pedreiros fazem um muro em 6 dias. Em quantos dias 3 pedreiros fariam o mesmo muro?",
        "O carro roda 12 km com 1 litro. Quantos litros gasta em uma viagem de 300 km, usando regra de três?",

        # --- conteudos.json: fracoes (3) — com keyword explícita ---
        "Como calcular a fração 1/4 do salário de R$ 2.000 que vai pro aluguel?",
        "Uma pizza foi dividida em 8 pedaços e você comeu 3. Que fração da pizza sobrou?",
        "Como somar duas frações com denominadores diferentes, como 1/2 com 1/3?",

        # --- conteudos.json: juros_simples (2) ---
        "Você pegou R$ 500 emprestado a 3% ao mês de juros simples. Quanto pagará de juros em 4 meses?",
        "Uma TV de R$ 1.200 foi parcelada com juros simples de 2% ao mês por 6 meses. Qual é o valor total pago?",

        # --- conteudos.json: area_perimetro (2) ---
        "Uma sala tem 4 m por 5 m. Quantos m² de piso você precisa comprar?",
        "Um terreno retangular mede 10 m por 25 m. Quantos metros de cerca são necessários para contorná-lo?",

        # --- conteudos.json: interpretacao_texto (2) ---
        "Como encontro a ideia principal de um texto?",
        "Não sei por onde começar a interpretar este parágrafo.",

        # --- conteudos.json: mais porcentagem/frações (3) ---
        "O botijão de gás custava R$ 110 e teve aumento de 20%. Qual foi o valor do aumento?",
        "O que significa 25%?",
        "Por que 50% é a metade?",

        # --- FAQ_CONSOLIDADO.md (12) ---
        "Como calcular a distância entre dois pontos no plano cartesiano?",
        "Quais radiações aparecem no esquema do espectro eletromagnético?",
        "Qual é o objetivo da aula sobre radiações e saúde?",
        "Quais problemas socioambientais urbanos são abordados?",
        "Como resolver o problema do investimento de Antonio e Bruna?",
        "Em que dia Fernanda pode retirar o exame após 20 dias corridos?",
        "Quais fontes não renováveis são citadas na síntese?",
        "O que é um texto de divulgação científica?",
        "Como se calcula a amplitude em um gráfico periódico?",
        "Qual é a diferença entre fonte primária e secundária de luz?",
        "Qual relação central é proposta na aula Solo, saúde e ambiente?",
        "Qual é a diferença entre juros simples e compostos?",
    ],
    "ground_truth": [
        # --- porcentagem ---
        "O aumento foi de R$ 22,00 (110 × 20 / 100).",
        "Você economiza R$ 360,00 (2.400 × 15 / 100).",
        "O novo salário é R$ 1.890,00 (1.800 + 5% de 1.800).",

        # --- regra_de_tres ---
        "Faz 100 pães (40 ÷ 2 × 5).",
        "3 pedreiros levam 8 dias (relação inversamente proporcional).",
        "Gasta 25 litros (300 ÷ 12).",

        # --- fracoes ---
        "Paga R$ 500,00 de aluguel (2.000 ÷ 4).",
        "Sobrou 5/8 da pizza (8/8 − 3/8).",
        "É preciso igualar os denominadores: 1/2 + 1/3 = 3/6 + 2/6 = 5/6.",

        # --- juros_simples ---
        "Juros = 500 × 0,03 × 4 = R$ 60,00.",
        "Total = 1.200 + (1.200 × 0,02 × 6) = R$ 1.344,00.",

        # --- area_perimetro ---
        "Precisa de 20 m² de piso (4 × 5).",
        "Precisa de 70 m de cerca (2 × (10 + 25)).",

        # --- interpretacao_texto ---
        "A ideia principal resume o assunto mais importante do texto; ajuda olhar o título e o início de cada parágrafo.",
        "Comece pelo título e pelo primeiro parágrafo; identifique sobre o que o texto fala antes dos detalhes.",

        # --- mais porcentagem/frações ---
        "O aumento foi de R$ 22,00 (110 × 20 / 100).",
        "25% é 25 partes de 100, ou 1/4.",
        "50% é 50/100, que simplificado é 1/2.",

        # --- FAQ_CONSOLIDADO.md ---
        "Para A = (x_A, y_A) e B = (x_B, y_B), d_AB = √[(x_B − x_A)² + (y_B − y_A)²].",
        "Ondas de rádio, micro-ondas, infravermelho, luz visível, ultravioleta, raios X e raios gama, em frequências crescentes.",
        "Classificar radiações eletromagnéticas por frequência, fontes e aplicações; discutir usos na saúde, com destaque para diagnóstico e tratamento do câncer de mama.",
        "Lixo, poluição, enchentes e falta de saneamento básico nas cidades brasileiras.",
        "Sistema a + b = 30.000 e b = a + 2.000. Substituindo, a = 14.000 e b = 16.000. Antonio investiu R$ 14.000 e Bruna R$ 16.000.",
        "Na terça-feira, usando a repetição semanal e o resto da divisão de 20 por 7.",
        "Combustão e fissão de material radioativo (como urânio).",
        "É um texto voltado a divulgar pesquisas e investigações acadêmicas ao público em geral.",
        "A = (máximo − mínimo) / 2.",
        "A fonte primária produz luz própria (Sol, lâmpada acesa); a secundária reflete luz recebida (Lua, parede iluminada).",
        "Compreender a inter-relação entre qualidade do solo, saneamento básico e saúde.",
        "Simples: J = C × i × t, taxa incide só sobre o capital inicial. Compostos: J = C × [(1 + i)^t − 1], considera o capital atualizado.",
    ],
}


# =============================================================================
# Wrappers do tool — tolerantes a retorno-string
# =============================================================================
def _consultar_base(pergunta: str) -> dict:
    """
    Chama ConsultarConteudoTool e devolve sempre um dict com chaves:
      topico, referencias_faq, conceitos, status_faq, _aviso (opcional)
    Quando o tool retorna string de aviso (sem tópico e sem FAQ), devolve dict vazio-marcado.
    """
    from calma_ai.tools.conteudo import ConsultarConteudoTool

    try:
        bruto = ConsultarConteudoTool()._run(pergunta)
    except Exception as e:
        return {"topico": None, "referencias_faq": [], "conceitos": [],
                "status_faq": "excecao", "_aviso": str(e)[:200]}

    try:
        dados = json.loads(bruto)
        if isinstance(dados, dict):
            dados.setdefault("referencias_faq", [])
            dados.setdefault("conceitos", [])
            return dados
    except (json.JSONDecodeError, TypeError):
        pass

    # Tool retornou string de aviso (sem correspondência)
    return {
        "topico": None,
        "referencias_faq": [],
        "conceitos": [],
        "status_faq": "sem_correspondencia",
        "_aviso": str(bruto)[:200],
    }


def _extrair_contextos_e_metadados(pergunta: str) -> tuple[list[str], dict]:
    dados = _consultar_base(pergunta)
    topico = dados.get("topico")
    topico_detectado = topico is not None and topico != "outro"

    contexts: list[str] = []
    faqs_aceitas = 0
    faqs_descartadas = 0

    if topico_detectado:
        for c in dados.get("conceitos") or []:
            if isinstance(c, str) and c.strip():
                contexts.append(c.strip())

    for ref in dados.get("referencias_faq") or []:
        sim = ref.get("similaridade") or 0.0
        if sim < SIMILARIDADE_MINIMA:
            faqs_descartadas += 1
            continue
        resposta = (ref.get("resposta") or "").strip()
        if resposta:
            contexts.append(f"[FAQ {ref.get('id','?')} sim={sim:.2f}] {resposta}")
            faqs_aceitas += 1

    if topico_detectado and faqs_aceitas:
        fonte = "topico+faq"
    elif topico_detectado:
        fonte = "topico"
    elif faqs_aceitas:
        fonte = "faq"
    else:
        fonte = "vazio"

    meta = {
        "fonte": fonte,
        "topico": topico,
        "faqs_aceitas": faqs_aceitas,
        "faqs_descartadas": faqs_descartadas,
        "aviso": dados.get("_aviso"),
    }
    return contexts, meta


def _resposta_extrativa_rag(pergunta: str, contexts: list[str], meta: dict) -> str:
    dados = _consultar_base(pergunta)

    if meta.get("topico") and meta["topico"] != "outro":
        conceitos = [c for c in (dados.get("conceitos") or []) if isinstance(c, str)]
        if conceitos:
            return " ".join(conceitos[:2])

    faqs = dados.get("referencias_faq") or []
    faqs = [f for f in faqs if (f.get("similaridade") or 0) >= SIMILARIDADE_MINIMA]
    if faqs:
        return faqs[0]["resposta"].strip()

    if dados.get("perguntas_orientadoras"):
        return dados["perguntas_orientadoras"][0].strip()

    return "Não encontrei essa informação na base."


def _rodar_tutor_crew(pergunta: str) -> tuple[list[str], str, dict]:
    from calma_ai.crew import CalmaAI
    from calma_ai.reflexo import decidir

    contexts, meta = _extrair_contextos_e_metadados(pergunta)
    reflexo = decidir(
        pergunta,
        assunto_curricular="conteúdo curricular",
        contexto_cotidiano="contas domésticas, salário ou rotina de trabalho",
    )
    if not reflexo.encaminhar_para_equipe:
        return contexts, (reflexo.resposta or "Sem resposta reflexa."), meta

    inputs = {
        "student_id": "avaliacao_ragas",
        "student_question": pergunta,
        "subject": "conteúdo curricular",
        "everyday_context": "contas domésticas, salário ou rotina de trabalho",
        "reflex_analysis": reflexo.resumo(),
        "reflex_response": reflexo.resposta or "",
    }
    try:
        resultado = CalmaAI().crew().kickoff(inputs=inputs)
    except Exception as e:
        return contexts, f"Erro no kickoff: {e}", meta

    resposta = resultado.pydantic
    if resposta is not None and hasattr(resposta, "mensagem"):
        return contexts, resposta.mensagem, meta
    return contexts, str(getattr(resultado, "raw", "Sem resposta do crew.")), meta


# =============================================================================
# Sanity
# =============================================================================
def sanity_check() -> bool:
    total = len(GABARITO["question"])
    problemas = 0

    print(f"[SANITY] Testando relevância de {total} perguntas...\n")
    for i, pergunta in enumerate(GABARITO["question"], 1):
        contexts, meta = _extrair_contextos_e_metadados(pergunta)
        fonte = meta["fonte"]

        if fonte in ("vazio", "erro"):
            status = "FALHA"
            problemas += 1
        else:
            status = "OK"

        aviso = ""
        if meta.get("aviso"):
            aviso = f" | aviso: {meta['aviso'][:60]}"

        print(
            f"[{i:02d}/{total}] {status:5s} | fonte={fonte:12s} "
            f"| topico={str(meta['topico']):20s} "
            f"| FAQ={meta['faqs_aceitas']}+/{meta['faqs_descartadas']}- "
            f"| n_ctx={len(contexts)} | {pergunta[:55]}{aviso}"
        )

    print(f"\n[SANITY] {total - problemas}/{total} perguntas com contexto relevante")
    if problemas:
        print(f"[SANITY] ⚠️  {problemas} pergunta(s) sem contexto relevante")
        return False
    print("[SANITY] ✅ Todas as perguntas têm contexto relevante. Pode rodar RAGAS.")
    return True


# =============================================================================
# Geração do dataset
# =============================================================================
def gerar_dataset(modo: str, max_perguntas: int | None) -> dict:
    questions = GABARITO["question"]
    truths = GABARITO["ground_truth"]
    if max_perguntas is not None:
        questions = questions[:max_perguntas]
        truths = truths[:max_perguntas]

    total = len(questions)
    print(f"[{modo.upper()}] Gerando dataset para {total} pergunta(s)...\n")

    dados = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    metas = []

    for i, pergunta in enumerate(questions, 1):
        if modo == "crew":
            contexts, answer, meta = _rodar_tutor_crew(pergunta)
        else:
            contexts, meta = _extrair_contextos_e_metadados(pergunta)
            answer = _resposta_extrativa_rag(pergunta, contexts, meta)

        dados["question"].append(pergunta)
        dados["answer"].append(answer)
        dados["contexts"].append(contexts)
        dados["ground_truth"].append(truths[i - 1])
        metas.append(meta)

        preview = answer[:110].replace("\n", " ")
        print(f"[{i}/{total}] fonte={meta['fonte']:12s} | n_ctx={len(contexts)} | {pergunta[:55]}")
        print(f"    → answer: {preview}{'...' if len(answer) > 110 else ''}\n")

    with open("dataset_ragas_real.json", "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)
    print("Dataset salvo em dataset_ragas_real.json")

    from collections import Counter
    contagem = Counter(m["fonte"] for m in metas)
    print("\n[RESUMO] Distribuição das fontes de contexto:")
    for fonte, n in contagem.most_common():
        print(f"  {fonte:15s}: {n}")

    return dados


# =============================================================================
# RAGAS
# =============================================================================
def avaliar_com_ragas():
    import pandas as pd
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    arquivo = Path("dataset_ragas_real.json")
    if not arquivo.exists():
        raise FileNotFoundError(
            "dataset_ragas_real.json não existe. Rode antes sem --so-dataset."
        )

    with open(arquivo, encoding="utf-8") as f:
        dados = json.load(f)

    dataset_ragas = Dataset.from_dict(dados)
    print(f"\n[RAGAS] Avaliando {len(dataset_ragas)} pergunta(s)...\n")

    juiz_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:1.5b", temperature=0))
    juiz_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    run_config = RunConfig(timeout=300, max_workers=1, max_retries=2)

    resultado = evaluate(
        dataset=dataset_ragas,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=juiz_llm,
        embeddings=juiz_emb,
        run_config=run_config,
    )

    df_metricas = resultado.to_pandas().reset_index(drop=True)
    df_dataset  = dataset_ragas.to_pandas().reset_index(drop=True)
    df_final    = pd.concat([df_dataset, df_metricas], axis=1)

    print("--- RESULTADO FINAL ---")
    print(resultado)

    print("\n--- DETALHAMENTO POR PERGUNTA ---")
    cols = ["question", "faithfulness", "answer_relevancy",
            "context_precision", "context_recall"]
    print(df_final[cols].to_string(index=False))

    df_final.to_csv("resultados_ragas_real.csv", index=False)
    print("\nSalvo em resultados_ragas_real.csv")

    if "answer_relevancy" in df_final.columns:
        print("\n--- TOP 5 PIORES answer_relevancy ---")
        piores = df_final.sort_values("answer_relevancy").head(5)
        for i, row in piores.iterrows():
            print(f"\n[{i+1}] answer_relevancy = {row['answer_relevancy']:.3f}")
            print(f"    P: {row.get('question', '')}")
            print(f"    R: {str(row.get('answer', ''))[:200]}")


# =============================================================================
# Log
# =============================================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", choices=["rag", "crew"], default="rag")
    parser.add_argument("--so-dataset", action="store_true")
    parser.add_argument("--max-perguntas", type=int, default=None)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--limpar", action="store_true")
    args = parser.parse_args()

    log = open("log_avaliacao_real.txt", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)

    try:
        if args.limpar:
            for f in ["dataset_ragas_real.json", "resultados_ragas_real.csv"]:
                p = Path(f)
                if p.exists():
                    p.unlink()
                    print(f"[limpar] removido {f}")

        if args.sanity:
            ok = sanity_check()
            return 0 if ok else 1

        gerar_dataset(args.modo, args.max_perguntas)
        if not args.so_dataset:
            avaliar_com_ragas()
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())