import sys

from dotenv import load_dotenv

from calma_ai.crew import CalmaAI
from calma_ai.models import Diagnostico, RespostaRevisada
from calma_ai.reflexo import decidir
from calma_ai.tools.progresso import registrar_interacao

PADROES = {
    "student_id": "estudante",
    "subject": "conteúdo curricular",
    "everyday_context": "contas domésticas, salário, horas extras ou rotina de trabalho",
}


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
        "student_question": _perguntar("Dúvida ou tentativa do estudante"),
        "subject": _perguntar("Conteúdo curricular", PADROES["subject"]),
        "everyday_context": _perguntar("Contexto do dia a dia", PADROES["everyday_context"]),
    }


def run() -> None:
    load_dotenv()
    inputs = coletar_inputs()

    reflexo = decidir(inputs["student_question"])
    print(f"\n[reflexo] {reflexo.resumo()}")
    if reflexo.resposta:
        print(f"Tutor: {reflexo.resposta}")
    if not reflexo.encaminhar_para_equipe:
        return
    inputs["reflex_analysis"] = reflexo.resumo()

    resultado = CalmaAI().crew().kickoff(inputs=inputs)

    diagnostico = resultado.tasks_output[0].pydantic
    resposta = resultado.pydantic

    if isinstance(diagnostico, Diagnostico):
        print(
            f"\n[agente] tópico: {diagnostico.topico} | ação: {diagnostico.proxima_acao.value}"
            f" | motivo: {diagnostico.justificativa}"
        )
        arquivo = registrar_interacao(inputs["student_id"], inputs["student_question"], diagnostico)
        print(f"[agente] progresso salvo em {arquivo}")

    if isinstance(resposta, RespostaRevisada):
        if resposta.revelava_resposta:
            print("[agente] o revisor corrigiu uma mensagem que revelava a resposta.")
        print(f"\nTutor: {resposta.mensagem}")
    else:
        print(f"\nTutor: {resultado.raw}")


if __name__ == "__main__":
    sys.exit(run())
