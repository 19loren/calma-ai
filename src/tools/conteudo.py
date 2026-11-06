from crewai.tools import tool
import chromadb
import os

@tool("Pesquisar Material do EJA")
def pesquisar_material_eja(pergunta: str) -> str:
    """
    Use esta ferramenta para pesquisar respostas e conceitos no material didático oficial do EJA.
    Sempre passe uma pergunta clara e objetiva como argumento.
    """
    # Aponta para a pasta onde salvamos o banco do ChromaDB
    diretorio_atual = os.path.dirname(__file__)
    caminho_db = os.path.abspath(os.path.join(diretorio_atual, "../../../dados/vetores"))
    
    try:
        cliente_chroma = chromadb.PersistentClient(path=caminho_db)
        colecao = cliente_chroma.get_collection(name="eja_ciencias")
        
        # Traz os 2 trechos mais relevantes para dar mais contexto ao agente
        resultados = colecao.query(
            query_texts=[pergunta],
            n_results=2
        )
        
        if resultados['documents'] and resultados['documents'][0]:
            resposta = "Contexto encontrado no material oficial:\n\n"
            for i, doc in enumerate(resultados['documents'][0]):
                pagina = resultados['metadatas'][0][i]['pagina']
                resposta += f"[Página {pagina}]: {doc}\n\n"
            return resposta
            
        return "Não encontrei nenhuma informação no material sobre este tópico."
        
    except Exception as e:
        return f"Erro ao acessar o banco de dados de conhecimento: {str(e)}"