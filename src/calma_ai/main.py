import logging
import os
import sys

from crewai import CrewOutput
from crewai.types.streaming import CrewStreamingOutput
from dotenv import load_dotenv

from calma_ai.crew import CalmaAI, validar_mensagem_final
from calma_ai.knowledge.exemplos_diagnosticador import (
    formatar_exemplos_diagnosticador,
)
from calma_ai.models import Diagnostico, RespostaRevisada
from calma_ai.reflexo import decidir, normalizar
from calma_ai.tools.progresso import registrar_interacao

LOGGER = logging.getLogger(__name__)
RESPOSTA_SEGURA_FALHA = (
    "Tive um problema técnico para validar a resposta agora. "
    "Você pode tentar novamente em instantes?"
)

PADROES = {
    "student_id": "estudante",
    "subject": "conteúdo curricular",
    "everyday_context": "contas domésticas, salário, horas extras ou rotina de trabalho",
}


def _debug_ativo() -> bool:
    return os.getenv("CALMA_DEBUG", "").strip().lower() in {"1", "true", "sim"}


def _perguntar(rotulo: str, padrao: str | None = None) -> str:
    sufixo = f" [{padrao}]" if padrao else ""
    while True:
        valor = input(f"{rotulo}{sufixo}: ").strip()
        if valor:
            return valor
        if padrao is not None:
            return padrao
        print("Este campo é obrigatório.")


def coletar_inputs() -> dict[str, str]:
    return {
        "student_id": _perguntar("Nome ou identificador do estudante", PADROES["student_id"]),
        "subject": _perguntar("Conteúdo curricular", PADROES["subject"]),
        "everyday_context": _perguntar("Contexto do dia a dia", PADROES["everyday_context"]),
    }


def executar_equipe(inputs: dict[str, str]) -> CrewOutput | CrewStreamingOutput:
    return CalmaAI().crew().kickoff(inputs=inputs)


def processar_turno(inputs: dict[str, str]) -> None:
    reflexo = decidir(
        inputs["student_question"],
        assunto_curricular=inputs["subject"],
        contexto_cotidiano=inputs["everyday_context"],
    )
    if _debug_ativo():
        print(f"\n[debug] {reflexo.resumo()}")
    if reflexo.resposta and not reflexo.encaminhar_para_equipe:
        print(f"\nTutor:\n{reflexo.resposta}\n")
    if not reflexo.encaminhar_para_equipe:
        return
    inputs["reflex_analysis"] = reflexo.resumo()
    inputs["reflex_response"] = reflexo.resposta or ""
    inputs["eja_fewshot_examples"] = formatar_exemplos_diagnosticador(
        inputs["student_question"]
    )

    try:
        resultado = executar_equipe(inputs)
    except Exception:
        LOGGER.exception("Falha ao executar a equipe de tutoria.")
        print(f"\nTutor:\n{RESPOSTA_SEGURA_FALHA}\n")
        return

    task_outputs = getattr(resultado, "tasks_output", ())
    resposta = getattr(resultado, "pydantic", None)
    if not task_outputs:
        LOGGER.error("A equipe não retornou resultados de tarefas.")
        print(f"\nTutor:\n{RESPOSTA_SEGURA_FALHA}\n")
        return

    diagnostico = getattr(task_outputs[0], "pydantic", None)
    resposta_final = getattr(task_outputs[-1], "pydantic", None)
    if (
        not isinstance(diagnostico, Diagnostico)
        or not isinstance(resposta, RespostaRevisada)
        or not isinstance(resposta_final, RespostaRevisada)
        or resposta != resposta_final
    ):
        LOGGER.error("A equipe retornou diagnóstico ou resposta final fora do formato esperado.")
        print(f"\nTutor:\n{RESPOSTA_SEGURA_FALHA}\n")
        return

    valido, motivo = validar_mensagem_final(
        task_outputs[-1],
        resposta_esperada=diagnostico.resposta_esperada,
    )
    if not valido:
        LOGGER.error("Resposta do tutor bloqueada pelo guardrail: %s", motivo)
        print(f"\nTutor:\n{RESPOSTA_SEGURA_FALHA}\n")
        return

    try:
        arquivo = registrar_interacao(
            inputs["student_id"], inputs["student_question"], diagnostico
        )
    except OSError:
        LOGGER.exception("Não foi possível salvar o progresso do estudante.")
    else:
        if _debug_ativo():
            print(
                f"[debug] tópico: {diagnostico.topico} | ação: "
                f"{diagnostico.proxima_acao.value} | motivo: {diagnostico.justificativa}"
            )
            print(f"[debug] progresso salvo em {arquivo}")

    if _debug_ativo() and resposta.revelava_resposta:
        print("[debug] o revisor ajustou uma mensagem que revelava a resposta.")
    print(f"\nTutor:\n{resposta.mensagem}\n")


def run() -> None:
    load_dotenv()
    try:
        contexto = coletar_inputs()
        print("\nConversa iniciada. Digite 'sair' para encerrar.\n")
        while True:
            pergunta = _perguntar("Dúvida ou tentativa do estudante")
            if normalizar(pergunta) in {"sair", "exit", "quit"}:
                print("Conversa encerrada.")
                return
            processar_turno({**contexto, "student_question": pergunta})
    except (EOFError, KeyboardInterrupt):
        print("\nConversa encerrada.")


if __name__ == "__main__":
    sys.exit(run())
