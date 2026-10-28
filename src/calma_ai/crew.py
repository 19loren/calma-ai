from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tasks.task_output import TaskOutput

from calma_ai.models import Diagnostico, RespostaRevisada
from calma_ai.tools import CalculadoraTool, ConsultarConteudoTool, PerfilEstudanteTool

LIMITE_CARACTERES = 1000


def validar_mensagem_final(output: TaskOutput) -> tuple[bool, Any]:
    """Guardrail determinístico aplicado depois do revisor."""
    resposta = output.pydantic
    if not isinstance(resposta, RespostaRevisada):
        try:
            resposta = RespostaRevisada.model_validate_json(output.raw)
        except ValueError:
            return False, "Retorne um JSON válido com os campos mensagem, revelava_resposta e ajustes."

    mensagem = resposta.mensagem.strip()
    if not mensagem:
        return False, "A mensagem final está vazia."
    if "?" not in mensagem:
        return False, "A mensagem final precisa conter uma pergunta orientadora para o estudante."
    if len(mensagem) > LIMITE_CARACTERES:
        return False, f"A mensagem final está longa demais; use no máximo {LIMITE_CARACTERES} caracteres."
    return True, output


@CrewBase
class CalmaAI:
    """Equipe de tutoria socrática da EJA: diagnosticador, tutor e revisor."""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def diagnosticador(self) -> Agent:
        return Agent(
            config=self.agents_config["diagnosticador"],  # type: ignore[index]
            tools=[PerfilEstudanteTool(), ConsultarConteudoTool(), CalculadoraTool()],
        )

    @agent
    def tutor(self) -> Agent:
        return Agent(
            config=self.agents_config["tutor"],  # type: ignore[index]
            tools=[ConsultarConteudoTool()],
        )

    @agent
    def revisor(self) -> Agent:
        return Agent(config=self.agents_config["revisor"])  # type: ignore[index]

    @task
    def diagnosticar(self) -> Task:
        return Task(
            config=self.tasks_config["diagnosticar"],  # type: ignore[index]
            output_pydantic=Diagnostico,
        )

    @task
    def orientar(self) -> Task:
        return Task(config=self.tasks_config["orientar"])  # type: ignore[index]

    @task
    def revisar(self) -> Task:
        return Task(
            config=self.tasks_config["revisar"],  # type: ignore[index]
            output_pydantic=RespostaRevisada,
            guardrail=validar_mensagem_final,
            guardrail_max_retries=2,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
        )
