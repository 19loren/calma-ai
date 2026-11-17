import os
from pymongo import MongoClient
from dotenv import load_dotenv

# carrega as variaveis de ambiente do .env
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")

if not MONGO_URI:
    raise ValueError("MONGODB_URI não encontrada no arquivo .env")

client = MongoClient(MONGO_URI)

db = client["calma_ai_db"]

# define as coleçoes estruturais do projeto
alunos_collection = db["alunos"]
sessoes_collection = db["sessoes_tutoria"]
progresso_collection = db["progresso"]
logs_collection = db["logs_reflexo"]

# busca o perfil do aluno no mongo e formata o contexto para o crewai
def obter_contexto_aluno(nome_aluno: str) -> str:
    
    aluno = alunos_collection.find_one({"nome_preferido": nome_aluno})
    
    if aluno:
        return (
            f"O aluno se chama {aluno.get('nome_preferido', 'Estudante')}, "
            f"trabalha como {aluno.get('profissao', 'profissional')} "
            f"e tem o seguinte objetivo: {aluno.get('observacao', '')}. "
            f"Use exemplos práticos do dia a dia dessa profissão."
        )
    return "O aluno é um estudante da EJA. Adapte a linguagem para adultos."

if __name__ == "__main__":
    try:
        client.admin.command('ping')
        print("Conexão segura com o MongoDB estabelecida com sucesso!")
    except Exception as e:
        print(f"Erro ao conectar: {e}")