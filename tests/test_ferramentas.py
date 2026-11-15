import json
from types import SimpleNamespace

import pytest
from crewai.tasks.task_output import TaskOutput

from calma_ai.crew import CalmaAI, validar_mensagem_final
from calma_ai.models import Diagnostico, ProximaAcao, RespostaRevisada
from calma_ai.tools import CalculadoraTool, ConsultarConteudoTool, PerfilEstudanteTool
from calma_ai.tools.calculadora import calcular
import calma_ai.tools.progresso as progresso
from calma_ai.tools.progresso import registrar_interacao


@pytest.mark.parametrize(
    ("expressao", "esperado"),
    [("180 * 10 / 100", 18), ("180 + 180 * 0,1", 198), ("(1200 - 300) / 3", 300), ("3 x 4", 12)],
)
def test_calculadora(expressao, esperado):
    assert calcular(expressao) == pytest.approx(esperado)


@pytest.mark.parametrize("expressao", ["__import__('os')", "open('x')", "2 ** 1000", "1/0"])
def test_calculadora_rejeita_expressoes_inseguras(expressao):
    assert CalculadoraTool()._run(expressao).startswith("Erro")


def test_consulta_conteudo_encontra_topico_pela_duvida(tmp_path, monkeypatch):
    monkeypatch.setenv("CALMA_FAQ_PATH", str(tmp_path / "faq_indisponivel.md"))
    resultado = ConsultarConteudoTool()._run("A conta de luz aumentou 10%, como calculo?")
    assert json.loads(resultado)["topico"] == "porcentagem"


def test_consulta_conteudo_sem_topico_lista_disponiveis(tmp_path, monkeypatch):
    monkeypatch.setenv("CALMA_FAQ_PATH", str(tmp_path / "faq_indisponivel.md"))
    assert "Tópicos disponíveis" in ConsultarConteudoTool()._run("verbos no passado")


def test_progresso_salva_dentro_da_pasta_de_dados(tmp_path, monkeypatch):
    monkeypatch.setenv("CALMA_DATA_DIR", str(tmp_path))
    diagnostico = Diagnostico(
        topico="porcentagem",
        dificuldade_principal="dividir por 100",
        raciocinio_do_estudante_correto=False,
        resposta_esperada="18",
        proxima_acao=ProximaAcao.DAR_DICA,
        justificativa="errou a divisão",
    )
    arquivo = registrar_interacao("../Maria Silva", "pergunta", diagnostico)
    registrar_interacao("../Maria Silva", "pergunta 2", diagnostico)

    assert arquivo.parent == tmp_path
    perfil = json.loads(PerfilEstudanteTool()._run("../Maria Silva"))
    assert perfil["dificuldades"] == {"dividir por 100": 2}
    assert len(perfil["historico"]) == 2


def test_progresso_escrita_atomica_preserva_arquivo_se_substituicao_falha(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CALMA_DATA_DIR", str(tmp_path))
    diagnostico = Diagnostico(
        topico="porcentagem",
        dificuldade_principal="nenhuma",
        resposta_esperada="18",
        proxima_acao=ProximaAcao.FAZER_PERGUNTA,
        justificativa="primeiro passo",
    )
    arquivo = registrar_interacao("Maria", "primeira pergunta", diagnostico)
    conteudo_original = arquivo.read_bytes()

    def falhar_substituicao(_origem, _destino):
        raise OSError("disco indisponível")

    monkeypatch.setattr(progresso.os, "replace", falhar_substituicao)
    with pytest.raises(OSError, match="disco indisponível"):
        registrar_interacao("Maria", "segunda pergunta", diagnostico)

    assert arquivo.read_bytes() == conteudo_original
    assert list(tmp_path.iterdir()) == [arquivo]


def _saida(mensagem: str) -> TaskOutput:
    resposta = RespostaRevisada(mensagem=mensagem, revelava_resposta=False)
    return TaskOutput(description="d", agent="a", raw=resposta.model_dump_json(), pydantic=resposta)


def test_guardrail_aceita_mensagem_com_pergunta():
    assert validar_mensagem_final(_saida("Quanto seria 1% de 180?"))[0]


@pytest.mark.parametrize("mensagem", ["O aumento foi de 18 reais.", "   ", "Pergunta? " * 200])
def test_guardrail_rejeita_mensagem_invalida(mensagem):
    assert not validar_mensagem_final(_saida(mensagem))[0]


def test_guardrail_aceita_pista_parcial_com_exemplo():
    assert validar_mensagem_final(
        _saida("Se o valor fosse R$ 100, como você descobriria quanto representa 10%?")
    )[0]


def test_guardrail_rejeita_conta_ja_resolvida():
    mensagem = "100 × 23 / 100 = 23. Como chegou a esse resultado?"
    valido, feedback = validar_mensagem_final(_saida(mensagem))
    assert not valido
    assert "conta já resolvida" in feedback


@pytest.mark.parametrize(
    ("mensagem", "esperado"),
    [
        ("O resultado é 18 reais. Como você chegou a esse valor?", "18"),
        ("A resposta foi R$ 18,00; qual etapa você faria primeiro?", "18"),
        (
            "A socialização é o processo de aprendizagem de normas sociais. O que percebe?",
            "A socialização é o processo de aprendizagem de normas sociais.",
        ),
        (
            "A socialização, processo de aprendizagem e construção cultural, é tema da aula. O que percebe?",
            "A socialização é o processo de aprendizagem de normas sociais.",
        ),
    ],
)
def test_guardrail_rejeita_vazamento_numerico_ou_textual(mensagem, esperado):
    saida = _saida(mensagem)
    valido, feedback = validar_mensagem_final(saida, resposta_esperada=esperado)
    assert not valido
    assert "revela a resposta esperada" in feedback


def test_guardrail_aceita_pergunta_sem_resposta_esperada():
    saida = _saida("O que você já identificou no enunciado?")
    assert validar_mensagem_final(saida, resposta_esperada="18")[0]


def test_guardrail_da_crew_usa_resposta_do_diagnostico():
    crew = CalmaAI().crew()
    crew.tasks[0].output = SimpleNamespace(
        pydantic=Diagnostico(
            topico="porcentagem",
            dificuldade_principal="nenhuma",
            resposta_esperada="18",
            proxima_acao=ProximaAcao.FAZER_PERGUNTA,
            justificativa="validar saída",
        )
    )

    valido, feedback = crew.tasks[-1].guardrail(
        _saida("O resultado é 18 reais. Como você chegou a esse valor?")
    )

    assert not valido
    assert "resposta esperada" in feedback
