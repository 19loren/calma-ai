import os
import socket
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

# força o python a usar apenas IPv4
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print(f"URL carregada: {QDRANT_URL}")
print(f"API Key carregada? {'Sim' if QDRANT_API_KEY else 'NÃO (None)'}")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("As variáveis QDRANT_URL ou QDRANT_API_KEY não foram encontradas no .env!")

COLLECTION_NAME = "eja_conhecimento"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    port=443,
    timeout=60,
    prefer_grpc=False,
)

# cria a collection se ela ainda nao existir
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        ),
    )
    print(f"Collection '{COLLECTION_NAME}' criada com sucesso!")
else:
    print(f"Collection '{COLLECTION_NAME}' já existe.")

# cria indices de payload
campos_indexados = ["materia", "etapa_eja", "termo"]
for campo in campos_indexados:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=campo,
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"Índice de filtro criado para o campo: '{campo}'")

print("\nSetup do Qdrant finalizado com sucesso!")