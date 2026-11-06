from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# importa a ferramenta de pesquisa do RAG
from tools.conteudo import pesquisar_material_eja

@CrewBase
class CalmaAiCrew():
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def agente_eja(self) -> Agent:
        return Agent(
            config=self.agents_config['agente_eja'],
            tools=[pesquisar_material_eja],
            verbose=True,
            allow_delegation=False
        )

