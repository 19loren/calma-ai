"""
Avaliação do Calma AI com RAGAS usando Google Gemini.
Versão corrigida: lida com argumentos dinâmicos das métricas,
limites de requisição da API (429) e diferentes versões do RAGAS.
"""

import argparse
import asyncio
import inspect
import json
import math
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIMILARIDADE_MINIMA = 0.30
# gemini-1.5-flash tem cota gratuita generosa (~1500 req/dia).
# gemini-2.0-flash também é uma opção, mas pode ter limite diário menor.
MODELO = "gemini-4-hydrogen"

GABARITO = {
    "question": [
        "A conta de luz era R$ 110 e teve aumento de 20%. Qual foi o valor do aumento?",
        "Uma loja dá 15% de desconto em uma geladeira de R$ 2.400. Quanto você economiza?",
        "O salário de R$ 1.800 teve reajuste de 5%. Qual é o novo salário?",
        "Em uma receita proporcional, com 2 kg de farinha você faz 40 pães. Quantos pães faz com 5 kg?",
        "Em uma proporção inversa, 4 pedreiros fazem um muro em 6 dias. Em quantos dias 3 pedreiros fariam o mesmo muro?",
        "O carro roda 12 km com 1 litro. Quantos litros gasta em uma viagem de 300 km, usando regra de três?",
        "Como calcular a fração 1/4 do salário de R$ 2.000 que vai pro aluguel?",
        "Uma pizza foi dividida em 8 pedaços e você comeu 3. Que fração da pizza sobrou?",
        "Como somar duas frações com denominadores diferentes, como 1/2 com 1/3?",
        "Você pegou R$ 500 emprestado a 3% ao mês de juros simples. Quanto pagará de juros em 4 meses?",
        "Uma TV de R$ 1.200 foi parcelada com juros simples de 2% ao mês por 6 meses. Qual é o valor total pago?",
        "Uma sala tem 4 m por 5 m. Quantos m² de piso você precisa comprar?",
        "Um terreno retangular mede 10 m por 25 m. Quantos metros de cerca são necessários para contorná-lo?",
        "Como encontro a ideia principal de um texto?",
        "Não sei por onde começar a interpretar este parágrafo.",
        "O botijão de gás custava R$ 110 e teve aumento de 20%. Qual foi o valor do aumento?",
        "O que significa 25%?",
        "Por que 50% é a metade?",
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
        "O aumento foi de R$ 22,00 (110 × 20 / 100).",
        "Você economiza R$ 360,00 (2.400 × 15 / 100).",
        "O novo salário é R$ 1.890,00 (1.800 + 5% de 1.800).",
        "Faz 100 pães (40 ÷ 2 × 5).",
        "3 pedreiros levam 8 dias (relação inversamente proporcional).",
        "Gasta 25 litros (300 ÷ 12).",
        "Paga R$ 500,00 de aluguel (2.000 ÷ 4).",
        "Sobrou 5/8 da pizza (8/8 − 3/8).",
        "É preciso igualar os denominadores: 1/2 + 1/3 = 3/6 + 2/6 = 5/6.",
        "Juros = 500 × 0,03 × 4 = R$ 60,00.",
        "Total = 1.200 + (1.200 × 0,02 × 6) = R$ 1.344,00.",
        "Precisa de 20 m² de piso (4 × 5).",
        "Precisa de 70 m de cerca (2 × (10 + 25)).",
        "A ideia principal resume o assunto mais importante do texto; ajuda olhar o título e o início de cada parágrafo.",
        "Comece pelo título e pelo primeiro parágrafo; identifique sobre o que o texto fala antes dos detalhes.",
        "O aumento foi de R$ 22,00 (110 × 20 / 100).",
        "25% é 25 partes de 100, ou 1/4.",
        "50% é 50/100, que simplificado é 1/2.",
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


def _consultar_base(pergunta: str) -> dict:
    from calma_ai.tools.conteudo import ConsultarConteudoTool

    try:
        bruto = ConsultarConteudoTool()._run(pergunta)
        dados = json.loads(bruto)
        return dados if isinstance(dados, dict) else {}
    except Exception:
        return {}


def _extrair_contextos(pergunta: str) -> list[str]:
    dados = _consultar_base(pergunta)
    topico = dados.get("topico")
    contexts = []

    if topico and topico != "outro":
        for c in dados.get("conceitos") or []:
            contexts.append(c.strip())

    for ref in dados.get("referencias_faq") or []:
        sim = ref.get("similaridade") or 0.0
        if sim >= SIMILARIDADE_MINIMA:
            contexts.append(
                f"[FAQ {ref.get('id', '?')} sim={sim:.2f}] {ref.get('resposta', '').strip()}"
            )

    return contexts


def _chamar_gemini_com_retry(client, prompt: str, max_tentativas: int = 4) -> str:
    """Chama a API do Gemini com backoff exponencial para erros 429."""
    for tentativa in range(max_tentativas):
        try:
            return client.models.generate_content(
                model=MODELO,
                contents=prompt,
                config={"temperature": 0.0},
            ).text.strip()
        except Exception as e:
            erro_str = str(e).lower()
            if "429" in erro_str or "resource exhausted" in erro_str or "quota" in erro_str:
                if tentativa < max_tentativas - 1:
                    delay = 2 ** tentativa  # 1s, 2s, 4s...
                    print(f"    [429] Limite atingido. Tentando novamente em {delay}s...")
                    import time
                    time.sleep(delay)
                    continue
            raise
    raise RuntimeError("Máximo de tentativas atingido")


def gerar_dataset_rag(max_perguntas=None):
    from google import genai

    questions = GABARITO["question"][:max_perguntas]
    truths = GABARITO["ground_truth"][:max_perguntas]

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    dados = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    total = len(questions)

    print(f"[RAG] Gerando respostas via Gemini ({MODELO}) para {total} pergunta(s)...\n")

    for i, pergunta in enumerate(questions, 1):
        contexts = _extrair_contextos(pergunta)
        ctx_texto = "\n".join(contexts)

        prompt = f"""Você é um tutor socrático. Baseado APENAS no contexto abaixo, responda à pergunta do aluno sem entregar a resposta final, guiando-o com uma pergunta.

Contexto: {ctx_texto}
Pergunta: {pergunta}
Resposta:"""

        try:
            resposta = _chamar_gemini_com_retry(client, prompt)
        except Exception as e:
            resposta = f"Erro na API Gemini: {e}"

        dados["question"].append(pergunta)
        dados["answer"].append(resposta)
        dados["contexts"].append(contexts)
        dados["ground_truth"].append(truths[i - 1])

        print(f"[{i}/{total}] {pergunta[:50]}")
        print(f"    → answer: {resposta[:100]}...\n")

    with open("dataset_ragas_gemini.json", "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)
    print("Dataset salvo em dataset_ragas_gemini.json\n")
    return dados


def avaliar_com_ragas():
    import pandas as pd
    from google import genai
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        Faithfulness,
        ContextPrecisionWithReference,
        ContextRecall,
    )

    arquivo = Path("dataset_ragas_gemini.json")
    if not arquivo.exists():
        raise FileNotFoundError(
            "Dataset não encontrado. Rode sem --so-avaliar primeiro."
        )

    with open(arquivo, encoding="utf-8") as f:
        dados = json.load(f)

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    juiz_llm = llm_factory(
        MODELO,
        provider="google",
        client=client,
        adapter="litellm",  # Força o uso do adaptador LiteLLM
    )

    metrics = [
        Faithfulness(llm=juiz_llm),
        ContextPrecisionWithReference(llm=juiz_llm),
        ContextRecall(llm=juiz_llm),
    ]

    print(
        f"[RAGAS] Avaliando {len(dados['question'])} pergunta(s) usando Gemini como Juiz...\n"
    )

    # Mapeamento explícito: JSON original -> nomes esperados pelo RAGAS 0.4
    MAPA_CAMPOS = {
        "question": "user_input",
        "answer": "response",
        "contexts": "retrieved_contexts",
        "ground_truth": "reference",
    }

    async def score_sample(sample_json, semaforo):
        sample = {MAPA_CAMPOS.get(k, k): v for k, v in sample_json.items()}
        row = {"question": sample_json["question"]}

        for metric in metrics:
            # Inspeciona a assinatura de ascore() para passar exatamente
            # os argumentos que a métrica espera. Funciona para qualquer métrica.
            parametros_esperados = inspect.signature(metric.ascore).parameters
            kwargs = {
                nome: sample[nome]
                for nome in parametros_esperados
                if nome in sample
            }

            async with semaforo:
                try:
                    result = await metric.ascore(**kwargs)
                    row[metric.name] = result.value
                except Exception as error:
                    print(f"{metric.name} falhou em {sample_json['question']!r}: {error}")
                    row[metric.name] = math.nan
        return row

    async def score_dataset(max_concurrency=2):
        # Concorrência baixa para respeitar o limite de requisições por minuto
        semaforo = asyncio.Semaphore(max_concurrency)
        tarefas = [
            score_sample(dict(zip(dados.keys(), valores)), semaforo)
            for valores in zip(*dados.values())
        ]
        return await asyncio.gather(*tarefas)

    resultados = asyncio.run(score_dataset())

    df_metricas = pd.DataFrame(resultados)
    df_dataset = pd.DataFrame(dados)
    df_final = pd.concat([df_dataset, df_metricas], axis=1)

    print("\n--- RESULTADO FINAL ---")
    colunas_metricas = [
        c for c in df_final.columns
        if c not in {"question", "answer", "contexts", "ground_truth"}
    ]
    if colunas_metricas:
        print(df_final[["question"] + colunas_metricas].describe())
    else:
        print("Nenhuma métrica foi calculada com sucesso.")

    df_final.to_csv("resultados_ragas_gemini.csv", index=False)
    print("\nSalvo em resultados_ragas_gemini.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--so-avaliar", action="store_true")
    parser.add_argument("--max-perguntas", type=int, default=10)
    args = parser.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERRO: Configure a variável GOOGLE_API_KEY antes de rodar.")
        sys.exit(1)

    if not args.so_avaliar:
        gerar_dataset_rag(args.max_perguntas)

    avaliar_com_ragas()


if __name__ == "__main__":
    main()