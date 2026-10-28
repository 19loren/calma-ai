import os
import sys

from crewai import CrewOutput
from crewai.types.streaming import CrewStreamingOutput
from dotenv import load_dotenv

from calma_ai.crew import CalmaAI
from calma_ai.models import Diagnostico, RespostaRevisada
from calma_ai.reflexo import decidir, normalizar
from calma_ai.tools.progresso import registrar_interacao

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

    resultado = executar_equipe(inputs)

    diagnostico = resultado.tasks_output[0].pydantic
    resposta = resultado.pydantic

    if isinstance(diagnostico, Diagnostico):
        arquivo = registrar_interacao(inputs["student_id"], inputs["student_question"], diagnostico)
        if _debug_ativo():
            print(
                f"[debug] tópico: {diagnostico.topico} | ação: "
                f"{diagnostico.proxima_acao.value} | motivo: {diagnostico.justificativa}"
            )
            print(f"[debug] progresso salvo em {arquivo}")

    if isinstance(resposta, RespostaRevisada):
        if _debug_ativo() and resposta.revelava_resposta:
            print("[debug] o revisor ajustou uma mensagem que revelava a resposta.")
        print(f"\nTutor:\n{resposta.mensagem}\n")
    else:
        print(f"\nTutor:\n{resultado.raw}\n")


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
