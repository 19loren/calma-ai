import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tasks.task_output import TaskOutput

from calma_ai.models import Diagnostico, RespostaRevisada
from calma_ai.tools import CalculadoraTool, ConsultarConteudoTool, PerfilEstudanteTool

LIMITE_CARACTERES = 1000
PADRAO_CONTA_RESOLVIDA = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*[x×*/÷+]\s*\d+(?:[.,]\d+)?"
    r"(?:\s*[x×*/÷+]\s*\d+(?:[.,]\d+)?)*\s*=\s*-?\d+(?:[.,]\d+)?"
)
PADRAO_NUMERO = re.compile(
    r"(?<![\w])[-+]?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?![\w])"
)
PALAVRAS_VAZIAS_RESPOSTA = frozenset(
    {
        "a", "as", "ao", "aos", "com", "da", "das", "de", "do", "dos",
        "e", "em", "na", "nas", "no", "nos", "o", "os", "para", "pela",
        "pelo", "por", "que", "um", "uma", "uns", "umas",
    }
)


def _normalizar_texto(texto: str) -> str:
    decomposed = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", sem_acento.lower())).strip()


def _numeros(texto: str) -> set[str]:
    valores: set[str] = set()
    for match in PADRAO_NUMERO.findall(texto):
        numero = match
        if "," in numero:
            numero = numero.replace(".", "").replace(",", ".")
        try:
            valor = Decimal(numero).normalize()
            valores.add("0" if valor == 0 else format(valor, "f"))
        except (InvalidOperation, ValueError):
            continue
    return valores


def resposta_esperada_vazada(mensagem: str, resposta_esperada: str) -> bool:
    """Detecta resposta esperada explícita, incluindo formatação numérica comum."""
    mensagem_normalizada = _normalizar_texto(mensagem)
    resposta_normalizada = _normalizar_texto(resposta_esperada)
    if resposta_normalizada and f" {resposta_normalizada} " in f" {mensagem_normalizada} ":
        return True
    palavras_esperadas = [
        palavra
        for palavra in resposta_normalizada.split()
        if palavra not in PALAVRAS_VAZIAS_RESPOSTA and len(palavra) > 2
    ]
    palavras_mensagem = mensagem_normalizada.split()

    def contem_em_ordem(frase: list[str]) -> bool:
        indice = 0
        for palavra in frase:
            while indice < len(palavras_mensagem) and palavras_mensagem[indice] != palavra:
                indice += 1
            if indice == len(palavras_mensagem):
                return False
            indice += 1
        return True

    if len(palavras_esperadas) >= 3:
        if any(
            contem_em_ordem(palavras_esperadas[inicio : inicio + 3])
            for inicio in range(len(palavras_esperadas) - 2)
        ):
            return True
    numeros_esperados = _numeros(resposta_esperada)
    return bool(numeros_esperados.intersection(_numeros(mensagem)))


def validar_mensagem_final(
    output: TaskOutput,
    resposta_esperada: str | None = None,
) -> tuple[bool, Any]:
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
    if resposta_esperada and resposta_esperada_vazada(mensagem, resposta_esperada):
        return (
            False,
            "A mensagem revela a resposta esperada. Reformule sem informar o resultado "
            "numérico ou textual.",
        )
    if PADRAO_CONTA_RESOLVIDA.search(mensagem.lower()):
        return (
            False,
            "Não apresente uma conta já resolvida. Dê uma pista parcial e faça uma "
            "pergunta orientadora.",
        )
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
        tarefa_diagnostico = self.diagnosticar()

        def validar_com_diagnostico(output: TaskOutput) -> tuple[bool, Any]:
            diagnostico = getattr(tarefa_diagnostico.output, "pydantic", None)
            if not isinstance(diagnostico, Diagnostico):
                return False, "Não foi possível validar a resposta esperada do diagnóstico."
            return validar_mensagem_final(output, diagnostico.resposta_esperada)

        return Task(
            config=self.tasks_config["revisar"],  # type: ignore[index]
            output_pydantic=RespostaRevisada,
            guardrail=validar_com_diagnostico,
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
