"""Testes do detector de emoções sem baixar o modelo BERT (usa um modelo falso)."""

import json

import calma_ai.tools.emocao as emocao
from calma_ai.models import Diagnostico, ProximaAcao
from calma_ai.tools import DetectEmocaoTool
from calma_ai.tools.progresso import carregar_perfil, registrar_interacao


class _SaidaFalsa:
    def __init__(self, probas):
        self.probas = probas


class _AnalisadorFalso:
    """Finge ser o pysentimiento: devolve probabilidades fixas."""

    def __init__(self, probas):
        self._probas = probas

    def predict(self, texto):
        return _SaidaFalsa(self._probas)


def _usar_modelo_falso(monkeypatch, probas):
    monkeypatch.setattr(emocao, "_analyzer", _AnalisadorFalso(probas))
    monkeypatch.setattr(emocao, "_backend", "pysentimiento")


def test_confusao_vira_sinal_pedagogico(monkeypatch):
    _usar_modelo_falso(monkeypatch, {"confusion": 0.82, "neutral": 0.40, "joy": 0.02})

    resultado = emocao.detectar_emocao("não entendi nada dessa conta")

    assert resultado["emocao_principal"] == "confusão"
    assert resultado["sinal_pedagogico"] == "confusao"
    assert resultado["status"] == "sinais_detectados"


def test_frustracao_agrupa_irritacao_e_decepcao(monkeypatch):
    _usar_modelo_falso(monkeypatch, {"annoyance": 0.55, "disappointment": 0.45})

    resultado = emocao.detectar_emocao("já tentei três vezes e não dá certo")

    assert resultado["sinal_pedagogico"] == "frustracao"


def test_mensagem_sem_sinal_fica_neutra(monkeypatch):
    _usar_modelo_falso(monkeypatch, {"neutral": 0.90, "joy": 0.05})

    resultado = emocao.detectar_emocao("quanto é 10% de 200?")

    assert resultado["emocao_principal"] is None
    assert resultado["sinal_pedagogico"] == "neutro"
    assert resultado["status"] == "inconclusivo"


def test_ferramenta_nao_quebra_com_mensagem_vazia():
    saida = json.loads(DetectEmocaoTool()._run("ana", "   "))

    assert saida["status"] == "erro"
    assert saida["sinal_pedagogico"] == "neutro"


def test_emocao_e_salva_no_historico(monkeypatch, tmp_path):
    monkeypatch.setenv("CALMA_DATA_DIR", str(tmp_path))
    diagnostico = Diagnostico(
        topico="porcentagem",
        dificuldade_principal="nenhuma",
        resposta_esperada="20",
        proxima_acao=ProximaAcao.DAR_DICA,
        justificativa="teste",
        emocao="confusão",
        sinal_pedagogico="confusao",
    )

    registrar_interacao("ana", "não entendi", diagnostico)

    ultimo = carregar_perfil("ana")["historico"][-1]
    assert ultimo["emocao"] == "confusão"
    assert ultimo["sinal_pedagogico"] == "confusao"


def test_diagnostico_continua_valido_sem_emocao():
    diagnostico = Diagnostico(
        topico="outro",
        dificuldade_principal="nenhuma",
        resposta_esperada="-",
        proxima_acao=ProximaAcao.FAZER_PERGUNTA,
        justificativa="teste",
    )

    assert diagnostico.emocao is None