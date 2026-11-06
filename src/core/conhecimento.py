import pymupdf
import chromadb
import os

def ingerir_pdf_para_chromadb(nome_ficheiro: str):
    # 1. Configurar os caminhos respeitando a nossa arquitetura
    diretorio_atual = os.path.dirname(__file__)
    caminho_base = os.path.abspath(os.path.join(diretorio_atual, "../../../dados"))
    caminho_pdf = os.path.join(caminho_base, "brutos", nome_ficheiro)
    caminho_db = os.path.join(caminho_base, "vetores")
    
    print(f"A preparar o banco de dados em: {caminho_db}")
    
    # 2. Iniciar o ChromaDB (Modo Persistente para gravar no disco)
    cliente_chroma = chromadb.PersistentClient(path=caminho_db)
    
    # Limpa a coleção antiga para garantirmos um teste fresco
    nome_colecao = "eja_ciencias"
    try:
        cliente_chroma.delete_collection(name=nome_colecao)
    except Exception:
        pass
    
    colecao = cliente_chroma.create_collection(name=nome_colecao)
    
    # 3. Processar e "Fatiar" o PDF
    print("A fatiar as páginas e a gerar as representações vetoriais (pode demorar na 1ª vez)...")
    documento = pymupdf.open(caminho_pdf)
    
    documentos = []
    metadados = []
    ids = []

    # vamos processar apenas 50 páginas para validar o conceito rapidamente
    limite_paginas = min(50, len(documento))
    
    for num_pagina in range(limite_paginas):
        texto = documento[num_pagina].get_text().strip()
        
        # Ignora páginas em branco ou só com números
        if len(texto) > 50:
            # Limpa quebras de linha estranhas
            texto_limpo = " ".join(texto.split())
            
            documentos.append(texto_limpo)
            metadados.append({"pagina": num_pagina, "origem": nome_ficheiro})
            ids.append(f"pag_{num_pagina}")

    # 4. Guardar no Banco Vetorial
    colecao.add(
        documents=documentos,
        metadatas=metadados,
        ids=ids
    )
    print(f"Sucesso! {len(documentos)} páginas guardadas no banco.")

    # pesquisa semantica
    # em vez de procurar por palavras exatas, o banco procura pelo "sentido" da frase
    pergunta = "qual é o objetivo do ensino de ciencias?"
    print(f"\n--- TESTE DE PESQUISA ---")
    print(f"Pergunta simulada do utilizador: '{pergunta}'")
    
    resultados = colecao.query(
        query_texts=[pergunta],
        n_results=1  # traz apenas o trecho mais relevante
    )
    
    if resultados['documents'][0]:
        print("\nO RAG encontrou isto:")
        print(f"-> Na Página: {resultados['metadatas'][0][0]['pagina']}")
        print(f"-> Trecho: {resultados['documents'][0][0][:400]}...")
    else:
        print("Nenhum resultado encontrado.")

if __name__ == "__main__":
    ingerir_pdf_para_chromadb("Material EJA - Ciência da Natureza - Professor.pdf")