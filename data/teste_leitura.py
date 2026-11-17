from database import alunos_collection

print("Buscando perfil do aluno no MongoDB Atlas...\n")

# query pelo nome ou qualquer outro campo
aluno = alunos_collection.find_one({"nome_preferido": "José"})

# verificamos se o banco encontrou o registro
if aluno:
    print("Perfil encontrado com sucesso!\n")
    print(f"Nome: {aluno['nome_preferido']}")
    print(f"Profissão: {aluno['profissao']}")
    print(f"Objetivo: {aluno['observacao']}")
    
    # simulacao da montagem do prompt para o seu agente do CrewAI
    prompt_contexto = (
        f"Você é um tutor socrático paciente. "
        f"Seu aluno se chama {aluno['nome_preferido']}, ele é {aluno['profissao']} "
        f"e {aluno['observacao'].lower()} "
        f"Use exemplos do dia a dia dele."
    )
    print("COMO O AGENTE VAI RECEBER A INFORMAÇÃO:")
    print(prompt_contexto)

else:
    print("Nenhum aluno encontrado com esse nome.")
