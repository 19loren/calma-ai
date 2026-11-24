from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from calma_ai.knowledge.faqs import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QDRANT_COLLECTION,
    FAQ,
    _encode,
    _faq_text,
    _load_model,
    parse_faqs,
)

load_dotenv(dotenv_path=ROOT_DIR / ".env")


def force_ipv4() -> None:
    old_getaddrinfo = socket.getaddrinfo
    if getattr(old_getaddrinfo, "_ipv4_forced", False):
        return

    def new_getaddrinfo(*args: Any, **kwargs: Any) -> list[Any]:
        responses = old_getaddrinfo(*args, **kwargs)
        return [r for r in responses if r[0] == socket.AF_INET]

    new_getaddrinfo._ipv4_forced = True  # type: ignore[attr-defined]
    socket.getaddrinfo = new_getaddrinfo


def faq_uuid(faq_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"calma-ai-faq:{faq_id.strip()}"))


def inferir_materia(faq: FAQ) -> str:
    if faq.materia:
        return faq.materia.strip().lower()
    texto = f"{faq.id} {faq.topic} {faq.document}".lower()
    if any(k in texto for k in ("mat-", "juros", "porcent", "equa", "fra", "geometr", "matem")):
        return "matematica"
    if any(k in texto for k in ("por-", "portug", "texto", "gramat", "reda", "leitura")):
        return "portugues"
    if any(k in texto for k in ("cie-", "biolog", "fisic", "quim", "energia", "saude", "radiac")):
        return "ciencias"
    if any(k in texto for k in ("hum-", "histor", "geograf", "cidadan", "trabalho")):
        return "humanas"
    return "geral"


def coletar_faqs() -> list[FAQ]:
    arquivos: list[Path] = []
    consolidado = ROOT_DIR / "data" / "faq" / "FAQ_CONSOLIDADO.md"
    if consolidado.exists():
        arquivos.append(consolidado)

    pasta_lotes = ROOT_DIR / "data" / "faq" / "lotes"
    if pasta_lotes.exists():
        arquivos.extend(sorted(pasta_lotes.rglob("*.md")))

    if not arquivos:
        raise FileNotFoundError("Nenhum arquivo .md de FAQ encontrado.")

    todas: list[FAQ] = []
    ids_vistos: dict[str, str] = {}
    for arquivo in arquivos:
        faqs_arquivo = parse_faqs(arquivo.read_text(encoding="utf-8"))
        for faq in faqs_arquivo:
            if faq.id in ids_vistos:
                raise ValueError(
                    f"ID duplicado '{faq.id}' em '{arquivo.name}' (já existe em '{ids_vistos[faq.id]}')."
                )
            ids_vistos[faq.id] = arquivo.name
            todas.append(faq)
        print(f"[FAQ OK] {arquivo.relative_to(ROOT_DIR)}: {len(faqs_arquivo)} registros")
    return todas


def coletar_trilhas() -> list[dict[str, Any]]:
    pasta_lotes = ROOT_DIR / "data" / "faq" / "lotes"
    if not pasta_lotes.exists():
        return []
    aulas: list[dict[str, Any]] = []
    ids_vistos: set[str] = set()
    for arquivo in sorted(pasta_lotes.rglob("exercicios.json")):
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        for aula in dados.get("aulas", []):
            aula_id = aula["aula_id"]
            if aula_id in ids_vistos:
                raise ValueError(f"Aula ID duplicado '{aula_id}' em '{arquivo}'.")
            ids_vistos.add(aula_id)
            aulas.append(aula)
        print(f"[TRILHA OK] {arquivo.relative_to(ROOT_DIR)}: {len(dados.get('aulas', []))} aulas")
    return aulas


def sincronizar_qdrant(faqs: list[FAQ], batch_size: int) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not qdrant_url or not qdrant_key:
        raise ValueError("QDRANT_URL e QDRANT_API_KEY precisam estar configurados no .env.")

    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION).strip()
    model_name = os.getenv("CALMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()

    print(f"\n[Qdrant] Gerando embeddings com '{model_name}'...")
    model = _load_model(model_name)
    vetores = _encode(model, [_faq_text(faq) for faq in faqs])

    force_ipv4()
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, port=443, timeout=30, prefer_grpc=False)
    existentes = {c.name for c in client.get_collections().collections}
    if collection not in existentes:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vetores.shape[1], distance=Distance.COSINE),
        )
    for campo in ("materia", "tema", "aula_id", "pdf"):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=campo,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    pontos = [
        PointStruct(
            id=faq_uuid(faq.id),
            vector=vetores[i].tolist(),
            payload={
                "id_faq": faq.id,
                "pergunta": faq.question,
                "resposta": faq.answer,
                "tema": faq.topic,
                "materia": inferir_materia(faq),
                "modulo": faq.modulo,
                "aula": faq.aula,
                "aula_id": faq.aula_id,
                "fonte": faq.source,
                "pdf": faq.pdf,
                "paginas": faq.pages,
                "origem": faq.origin,
                "documento": faq.document,
            },
        )
        for i, faq in enumerate(faqs)
    ]

    for inicio in range(0, len(pontos), batch_size):
        lote = pontos[inicio : inicio + batch_size]
        client.upsert(collection_name=collection, points=lote, wait=True)
        print(f"[Qdrant] Lote {inicio + 1}..{inicio + len(lote)} sincronizado.")

    info = client.get_collection(collection)
    print(f"[Qdrant] Concluído! Pontos ativos em '{collection}': {info.points_count}")


def sincronizar_mongo(aulas: list[dict[str, Any]]) -> None:
    from pymongo import ASCENDING, MongoClient

    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise ValueError("MONGODB_URI precisa estar configurada no .env.")

    db_name = os.getenv("MONGODB_DB_NAME", "calma_ai_db").strip()
    force_ipv4()
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    db = client[db_name]
    colecao = db["trilhas"]
    colecao.create_index([("aula_id", ASCENDING)], unique=True)
    colecao.create_index([("materia", ASCENDING), ("modulo_ordem", ASCENDING), ("aula_ordem", ASCENDING)])

    for aula in aulas:
        colecao.update_one({"aula_id": aula["aula_id"]}, {"$set": aula}, upsert=True)
    print(f"[MongoDB] Concluído! {len(aulas)} aulas sincronizadas na coleção '{db_name}.trilhas'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sincroniza FAQs (Qdrant) e Trilhas (MongoDB).")
    parser.add_argument("--dry-run", action="store_true", help="Apenas valida os arquivos sem enviar à nuvem.")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do lote no Qdrant.")
    args = parser.parse_args()

    faqs = coletar_faqs()
    aulas = coletar_trilhas()
    print(f"\nResumo validado: {len(faqs)} FAQs e {len(aulas)} aulas ordenadas.")

    if args.dry_run:
        print("Modo --dry-run finalizado com sucesso.")
        return

    sincronizar_qdrant(faqs, batch_size=args.batch_size)
    if aulas:
        sincronizar_mongo(aulas)


if __name__ == "__main__":
    main()