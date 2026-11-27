"""
DetectEmocaoTool - Calma.AI

Reconhecimento de possíveis sinais emocionais nas mensagens dos
estudantes, usando o modelo pysentimiento/bert-pt-emotion
(BERTimbau ajustado nas 28 emoções do GoEmotions, multirrótulo).

Fluxo:
    mensagem -> BERT (28 emoções, sigmoid) -> emoções em português
             -> sinais pedagógicos (confusão, frustração, ...) -> diagnosticador

A classificação é uma estimativa textual, não um diagnóstico psicológico.
Ela complementa, mas nunca substitui, as regras de segurança do reflexo.py.

Esta ferramenta só detecta e devolve o resultado. Quem salva a emoção no
histórico é o progresso.py (registrar_interacao), a partir do Diagnostico.
"""

import json
import logging

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# -------------------------------------------------
# 1. CONFIGURAÇÃO
# -------------------------------------------------

MODELO = "pysentimiento/bert-pt-emotion"

# O modelo é multirrótulo (sigmoid): cada emoção tem sua própria
# probabilidade e elas NÃO somam 1. Com 28 classes, 0.50 quase nunca é
# atingido em mensagens curtas; 0.30 é um ponto de partida comum para
# GoEmotions. Ajuste com mensagens reais dos estudantes.
THRESHOLD = 0.30

# Rótulos do modelo (inglês) -> português
TRADUCAO = {
    "admiration": "admiração", "amusement": "diversão", "anger": "raiva",
    "annoyance": "irritação", "approval": "aprovação", "caring": "cuidado",
    "confusion": "confusão", "curiosity": "curiosidade", "desire": "desejo",
    "disappointment": "decepção", "disapproval": "desaprovação",
    "disgust": "nojo", "embarrassment": "constrangimento",
    "excitement": "empolgação", "fear": "medo", "gratitude": "gratidão",
    "grief": "luto", "joy": "alegria", "love": "amor",
    "nervousness": "nervosismo", "optimism": "otimismo", "pride": "orgulho",
    "realization": "compreensão", "relief": "alívio", "remorse": "remorso",
    "sadness": "tristeza", "surprise": "surpresa", "neutral": "neutro",
}

# Emoções do modelo -> sinais que interessam ao diagnóstico pedagógico.
# É aqui que o "decifrador" (BERT) vira informação para o diagnosticador.
SINAIS_PEDAGOGICOS = {
    "confusao": ["confusion"],
    "frustracao": ["annoyance", "disappointment", "anger", "disapproval"],
    "ansiedade": ["nervousness", "fear", "embarrassment"],
    "desanimo": ["sadness", "grief", "remorse"],
    "curiosidade": ["curiosity", "desire", "surprise"],
    "engajamento_positivo": [
        "joy", "excitement", "gratitude", "admiration", "approval",
        "optimism", "pride", "relief", "amusement", "realization",
    ],
}

# Orientação curta que o diagnosticador pode usar diretamente
ORIENTACAO = {
    "confusao": "Reexplicar com outro exemplo, em passos menores.",
    "frustracao": "Validar a dificuldade, reduzir o nível e propor uma vitória rápida.",
    "ansiedade": "Tranquilizar, lembrar que errar faz parte e ir no ritmo do estudante.",
    "desanimo": "Acolher, reconhecer o esforço e retomar algo que ele já domina.",
    "curiosidade": "Aproveitar o interesse e aprofundar o tema.",
    "engajamento_positivo": "Manter o ritmo e propor o próximo desafio.",
    "neutro": "Seguir o plano normalmente.",
}

# -------------------------------------------------
# 2. CARREGAMENTO DO MODELO (uma única vez)
# -------------------------------------------------

_analyzer = None
_backend = None


def carregar_analisador():
    """
    Tenta o pysentimiento (que também faz o pré-processamento do texto).
    Se ele falhar na instalação/importação, cai para o transformers puro,
    que usa exatamente o mesmo modelo BERT.
    """
    global _analyzer, _backend

    if _analyzer is not None:
        return _analyzer

    try:
        from pysentimiento import create_analyzer
        _analyzer = create_analyzer(task="emotion", lang="pt")
        _backend = "pysentimiento"
    except Exception as erro:
        logger.warning("pysentimiento indisponível (%s); usando transformers.", erro)
        from transformers import pipeline
        _analyzer = pipeline(
            "text-classification",
            model=MODELO,
            top_k=None,                   # devolve as 28 emoções
            function_to_apply="sigmoid",  # multirrótulo
        )
        _backend = "transformers"

    return _analyzer


def aquecer_modelo():
    """Chame no início da aplicação para não travar o 1º atendimento."""
    carregar_analisador()


def _probabilidades_brutas(texto: str) -> dict:
    analyzer = carregar_analisador()

    if _backend == "pysentimiento":
        return dict(analyzer.predict(texto).probas)

    saida = analyzer(texto)
    if saida and isinstance(saida[0], list):  # algumas versões aninham a lista
        saida = saida[0]
    return {item["label"]: item["score"] for item in saida}


# -------------------------------------------------
# 3. IDENTIFICAÇÃO DAS EMOÇÕES
# -------------------------------------------------

def detectar_emocao(texto: str) -> dict:
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("A mensagem do estudante não pode estar vazia.")

    brutas = _probabilidades_brutas(texto)

    probabilidades = dict(sorted(
        ((TRADUCAO.get(k, k), round(float(v), 4)) for k, v in brutas.items()),
        key=lambda item: item[1],
        reverse=True,
    ))

    # Emoções acima do limiar, ignorando "neutro"
    emocoes_detectadas = {
        emocao: p for emocao, p in probabilidades.items()
        if p >= THRESHOLD and emocao != "neutro"
    }

    # Pontuação de cada sinal pedagógico = maior emoção do grupo
    sinais = {
        sinal: round(max(float(brutas.get(r, 0.0)) for r in rotulos), 4)
        for sinal, rotulos in SINAIS_PEDAGOGICOS.items()
    }
    sinais = dict(sorted(sinais.items(), key=lambda i: i[1], reverse=True))

    sinal_principal, pontuacao = next(iter(sinais.items()))
    if pontuacao < THRESHOLD:
        sinal_principal = "neutro"

    emocao_principal = next(iter(emocoes_detectadas), None)

    return {
        "emocao_principal": emocao_principal,
        "pontuacao_principal": emocoes_detectadas.get(emocao_principal),
        "emocoes_detectadas": emocoes_detectadas,
        "sinal_pedagogico": sinal_principal,
        "pontuacao_sinais": sinais,
        "orientacao_para_diagnostico": ORIENTACAO[sinal_principal],
        "top5_probabilidades": dict(list(probabilidades.items())[:5]),
        "limiar_usado": THRESHOLD,
        "status": "sinais_detectados" if emocoes_detectadas else "inconclusivo",
        "aviso": "Estimativa feita a partir do texto; não é diagnóstico psicológico.",
    }


# -------------------------------------------------
# 4. FERRAMENTA CREWAI
# -------------------------------------------------

class EmotionInput(BaseModel):
    estudante_id: str = Field(description="Identificador do estudante.")
    mensagem: str = Field(description="Última mensagem enviada pelo estudante.")


class DetectEmocaoTool(BaseTool):
    name: str = "detectar_emocao"
    description: str = (
        "Analisa a mensagem do estudante com um modelo BERT em português "
        "(28 emoções do GoEmotions) e devolve: as emoções detectadas, o "
        "sinal pedagógico principal (confusao, frustracao, ansiedade, "
        "desanimo, curiosidade, engajamento_positivo ou neutro) e uma "
        "orientação de como adaptar a explicação. Use durante o diagnóstico "
        "pedagógico. Os resultados são estimativas, não diagnósticos psicológicos."
    )
    args_schema: type[BaseModel] = EmotionInput

    def _run(self, estudante_id: str, mensagem: str) -> str:
        try:
            resposta = {"estudante_id": estudante_id, **detectar_emocao(mensagem)}
        except Exception as erro:
            resposta = {
                "estudante_id": estudante_id,
                "status": "erro",
                "erro": f"{type(erro).__name__}: {erro}",
                "emocao_principal": None,
                "sinal_pedagogico": "neutro",
                "orientacao_para_diagnostico": ORIENTACAO["neutro"],
            }
        return json.dumps(resposta, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Teste rápido: uv run python src/calma_ai/tools/emocao.py
    for frase in [
        "não entendi nada dessa conta, to perdido",
        "já tentei três vezes e continua dando errado, que raiva",
        "nossa, que legal! e como faz com números maiores?",
    ]:
        print(frase)
        print(DetectEmocaoTool()._run("teste", frase), "\n")