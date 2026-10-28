from types import SimpleNamespace
from unittest.mock import Mock

import calma_ai.main as main
from calma_ai.crew import CalmaAI
from calma_ai.models import RespostaRevisada


def test_equipe_usa_llama_local_via_ollama(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    agentes = CalmaAI().crew().agents

    assert len(agentes) == 3
    for agente in agentes:
        assert agente.llm.model == "llama3.2"
        assert agente.llm.base_url == "http://localhost:11434/v1"


def test_executar_equipe_encaminha_contexto_para_crew(monkeypatch):
    resultado = object()
    kickoff = Mock(return_value=resultado)

    class CrewStub:
        def kickoff(self, *, inputs):
            return kickoff(inputs=inputs)

    class CalmaAIStub:
        def crew(self):
            return CrewStub()

    monkeypatch.setattr(main, "CalmaAI", CalmaAIStub)
    inputs = {
        "student_question": "Minha conta aumentou 10%.",
        "subject": "porcentagem",
        "everyday_context": "conta de luz",
        "reflex_analysis": "R6 -> encaminhar_equipe",
    }

    assert main.executar_equipe(inputs) is resultado
    kickoff.assert_called_once_with(inputs=inputs)


def test_run_processa_varios_turnos_ate_comando_sair(monkeypatch):
    respostas = iter(
        [
            "Maximus",
            "porcentagem",
            "conta de água",
            "Como calculo 10%?",
            "Eu tentei dividir por 10.",
            "SAIR",
        ]
    )
    processar_turno = Mock()
    monkeypatch.setattr(main, "load_dotenv", Mock())
    monkeypatch.setattr(main, "_perguntar", lambda *_args: next(respostas))
    monkeypatch.setattr(main, "processar_turno", processar_turno)

    main.run()

    assert processar_turno.call_count == 2
    assert processar_turno.call_args_list[0].args[0] == {
        "student_id": "Maximus",
        "subject": "porcentagem",
        "everyday_context": "conta de água",
        "student_question": "Como calculo 10%?",
    }
    assert processar_turno.call_args_list[1].args[0]["student_question"] == (
        "Eu tentei dividir por 10."
    )


def test_processar_turno_incorpora_resposta_reflexa_sem_exibir_frase_pronta(
    monkeypatch, capsys
):
    executar_equipe = Mock(
        return_value=SimpleNamespace(
            tasks_output=[SimpleNamespace(pydantic=None)],
            pydantic=RespostaRevisada(
                mensagem="Vamos olhar só para o primeiro passo. O que o exercício quer descobrir?",
                revelava_resposta=False,
            ),
            raw="",
        )
    )
    monkeypatch.setattr(main, "executar_equipe", executar_equipe)
    inputs = {
        "student_id": "Max",
        "student_question": "Me dá a resposta logo: quanto é 10% de 180?",
        "subject": "porcentagem",
        "everyday_context": "conta de luz",
    }

    main.processar_turno(inputs)

    texto = capsys.readouterr().out
    assert texto.count("Tutor:") == 1
    assert "[reflexo]" not in texto
    assert "[agente]" not in texto
    assert "Eu não vou te entregar o resultado pronto" not in texto
    assert "Vamos olhar só para o primeiro passo." in texto
    assert inputs["reflex_response"].startswith("Eu não vou te entregar")
    assert "reflex_response" in executar_equipe.call_args.args[0]


def test_processar_turno_oculta_diagnostico_no_modo_normal(monkeypatch, capsys):
    executar_equipe = Mock(
        return_value=SimpleNamespace(
            tasks_output=[SimpleNamespace(pydantic=None)],
            pydantic=RespostaRevisada(
                mensagem="O que você já percebeu?",
                revelava_resposta=False,
            ),
            raw="",
        )
    )
    monkeypatch.setattr(main, "executar_equipe", executar_equipe)
    monkeypatch.delenv("CALMA_DEBUG", raising=False)

    main.processar_turno(
        {
            "student_id": "Max",
            "student_question": "Como interpreto um texto?",
            "subject": "Interpretação de texto",
            "everyday_context": "rotina",
        }
    )

    texto = capsys.readouterr().out
    assert "[debug]" not in texto
    assert "Tutor:\nO que você já percebeu?" in texto
