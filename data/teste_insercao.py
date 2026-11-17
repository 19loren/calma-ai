from database import alunos_collection

# perfil de teste
aluno_teste = {
    "nome_preferido": "José",
    "idade": 52,
    "profissao": "Pedreiro",
    "foco_interesse": ["Construção Civil", "Orçamento Familiar"],
    "observacao": "Quer aprender matemática para não ser enganado em compras de material."
}

print("Enviando dados para o MongoDB Atlas...")
resultado = alunos_collection.insert_one(aluno_teste)

print(f"Sucesso! Aluno salvo com o ID: {resultado.inserted_id}")