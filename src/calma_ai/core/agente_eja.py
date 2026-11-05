import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# carrega a GEMINI_API_KEY diretamente do .env
load_dotenv()

# inicializa o LLM + versao exata do modelo
llm_gemini = LLM(
    model="gemini/gemini-3.8-flash", 
    api_key=os.getenv("GEMINI_API_KEY")
)

# definir o agente
tradutor_andragogico = Agent(
    role='Tradutor Andragógico para EJA',
    goal='Adaptar problemas escolares infantis para a realidade prática de trabalhadores adultos.',
    backstory='É um professor de adultos com muita empatia. Sabe que os alunos chegam cansados do trabalho e que a matemática deve fazer sentido no dia a dia, como calcular descontos no supermercado ou no salário.',
    llm=llm_gemini, 
    verbose=True,
    allow_delegation=False
)

# definir a tarefa 
problema_infantil = "O Marcinho tinha 50 berlindes e perdeu 20%. Com quantos ficou?"

tarefa_adaptacao = Task(
    description=f'''
    Reescreva o seguinte problema matemático escolar: "{problema_infantil}". 
    Remova a linguagem infantil (por ex: berlindes, Pedrinho). 
    Adapte o contexto para a vida de um adulto (por exemplo: calcular o desconto numa fatura de eletricidade, ou juros de um cartão de crédito).
    A lógica matemática deve manter-se (calcular 20% de 50 e subtrair).
    ''',
    expected_output='Um texto curto, direto e acolhedor, com o novo problema matemático focado na realidade de um trabalhador.',
    agent=tradutor_andragogico
)

# montar o crew e executar
crew = Crew(
    agents=[tradutor_andragogico],
    tasks=[tarefa_adaptacao],
    process=Process.sequential
)

print("iniciando o agente do EJA...\n")
resultado = crew.kickoff()

print("\nPROBLEMA ADAPTADO PARA O ALUNO: ")
print(resultado)