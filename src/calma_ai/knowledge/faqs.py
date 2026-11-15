"""Leitura e recuperação semântica das FAQs usadas para identificar materiais."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FAQ_PATH = ROOT_DIR / "data" / "faq" / "FAQ_CONSOLIDADO.md"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SIMILARITY = 0.625
AMBIGUITY_MARGIN_RATIO = 0.0


@dataclass(frozen=True)
class FAQ:
    id: str
    question: str
    answer: str
    topic: str
    source: str
    pdf: str
    pages: str
    origin: str
    document: str


@dataclass(frozen=True)
class FAQMatch:
    faq: FAQ
    score: float


@dataclass(frozen=True)
class FAQSearch:
    matches: tuple[FAQMatch, ...]
    ambiguous: bool


_DOCUMENT_HEADER = re.compile(r"^# Documento \d+:\s*(.+?)\s*$")
_QUESTION_HEADER = re.compile(r"^##\s+(.+?)\s*$")
_METADATA_LINE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")


def faq_path() -> Path:
    """Resolve o arquivo local, permitindo apontar para outra base por ambiente."""
    return Path(os.getenv("CALMA_FAQ_PATH", str(DEFAULT_FAQ_PATH))).expanduser()


def parse_faqs(markdown: str) -> tuple[FAQ, ...]:
    """Lê cada seção pergunta–resposta como um registro indivisível."""
    lines = markdown.splitlines()
    documents: list[FAQ] = []
    document_title = ""
    index = 0

    while index < len(lines):
        document_match = _DOCUMENT_HEADER.match(lines[index])
        if document_match:
            document_title = document_match.group(1).strip()
            index += 1
            continue

        question_match = _QUESTION_HEADER.match(lines[index])
        if not question_match:
            index += 1
            continue

        question = question_match.group(1).strip()
        index += 1
        section: list[str] = []
        while index < len(lines):
            if _QUESTION_HEADER.match(lines[index]) or _DOCUMENT_HEADER.match(lines[index]):
                break
            section.append(lines[index])
            index += 1

        answer_lines: list[str] = []
        metadata: dict[str, str] = {}
        reading_metadata = False
        for line in section:
            metadata_match = _METADATA_LINE.match(line)
            if metadata_match:
                reading_metadata = True
                metadata[metadata_match.group(1).strip()] = metadata_match.group(2).strip()
            elif reading_metadata:
                if line.strip():
                    raise ValueError(f"Metadado inesperado na FAQ '{question}'.")
            else:
                answer_lines.append(line)

        answer = "\n".join(answer_lines).strip()
        required = ("ID", "Tema", "Fonte", "PDF", "Páginas do PDF / slides", "Origem")
        missing = [field for field in required if not metadata.get(field)]
        if not answer or missing:
            raise ValueError(
                f"FAQ '{question}' inválida; resposta vazia ou metadados ausentes: "
                + ", ".join(missing)
            )

        origin_match = re.search(r"\[([^\]]+)\]\(([^)]+)\)", metadata["Origem"])
        origin = (
            f"{origin_match.group(1)} ({origin_match.group(2)})"
            if origin_match
            else metadata["Origem"]
        )
        documents.append(
            FAQ(
                id=metadata["ID"],
                question=question,
                answer=answer,
                topic=metadata["Tema"],
                source=metadata["Fonte"],
                pdf=metadata["PDF"],
                pages=metadata["Páginas do PDF / slides"],
                origin=origin,
                document=document_title,
            )
        )

    if not documents:
        raise ValueError("A base FAQ não contém seções de pergunta e resposta.")
    ids = [faq.id for faq in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("A base FAQ contém identificadores duplicados.")
    return tuple(documents)


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    cache_folder = os.getenv("HF_HUB_CACHE")
    if cache_folder is None:
        hf_home = os.getenv("HF_HOME")
        if hf_home is None and sys.prefix != sys.base_prefix:
            hf_home = str(Path(sys.prefix) / "hf-cache")
        if hf_home is not None:
            cache_folder = str(Path(hf_home) / "hub")
    return SentenceTransformer(model_name, cache_folder=cache_folder)


def _faq_text(faq: FAQ) -> str:
    return (
        f"Pergunta: {faq.question}\n"
        f"Resposta: {faq.answer}\n"
        f"Tema: {faq.topic}\n"
        f"Material: {faq.document}"
    )


def _encode(model: Any, texts: Sequence[str]) -> np.ndarray:
    embeddings = np.asarray(
        model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("O modelo retornou uma matriz de embeddings inválida.")
    if not np.isfinite(embeddings).all():
        raise ValueError("O modelo retornou embeddings não finitos.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)


def _float_setting(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as erro:
        raise ValueError(f"{name} precisa ser um número entre {minimum} e {maximum}.") from erro
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} precisa ser um número entre {minimum} e {maximum}.")
    return value


@lru_cache(maxsize=4)
def _load_index(
    path: Path,
    modified_ns: int,
    size: int,
    model_name: str,
) -> tuple[tuple[FAQ, ...], np.ndarray]:
    del modified_ns, size
    faqs = parse_faqs(path.read_text(encoding="utf-8"))
    embeddings = _encode(_load_model(model_name), [_faq_text(faq) for faq in faqs])
    return faqs, embeddings


def _model_name() -> str:
    return os.getenv("CALMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()


def _is_ambiguous(matches: Sequence[FAQMatch], margin_ratio: float) -> bool:
    return (
        len(matches) > 1
        and matches[0].score - matches[1].score
        <= matches[0].score * margin_ratio
    )


def search_many_faqs(
    queries: Sequence[str],
    *,
    limit: int = 3,
    path: Path | None = None,
    min_similarity: float | None = None,
    ambiguity_margin_ratio: float | None = None,
) -> tuple[FAQSearch, ...]:
    """Busca várias consultas em lote para avaliação, com o mesmo índice neural."""
    if limit < 1:
        raise ValueError("O limite de resultados precisa ser positivo.")
    if not queries:
        return ()
    if any(not isinstance(query, str) for query in queries):
        raise TypeError("Todas as consultas precisam ser textos.")

    threshold = (
        min_similarity
        if min_similarity is not None
        else _float_setting(
            "CALMA_FAQ_MIN_SIMILARITY", MIN_SIMILARITY, minimum=0.0, maximum=1.0
        )
    )
    margin_ratio = (
        ambiguity_margin_ratio
        if ambiguity_margin_ratio is not None
        else _float_setting(
            "CALMA_FAQ_AMBIGUITY_MARGIN_RATIO",
            AMBIGUITY_MARGIN_RATIO,
            minimum=0.0,
            maximum=1.0,
        )
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("O limiar de similaridade precisa estar entre 0 e 1.")
    if not 0.0 <= margin_ratio <= 1.0:
        raise ValueError("A margem de ambiguidade precisa estar entre 0 e 1.")
    if all(not query.strip() for query in queries):
        return tuple(FAQSearch(matches=(), ambiguous=False) for _ in queries)

    corpus_path = (path or faq_path()).resolve()
    stat = corpus_path.stat()
    model_name = _model_name()
    faqs, faq_embeddings = _load_index(
        corpus_path, stat.st_mtime_ns, stat.st_size, model_name
    )
    nonempty_indices = [index for index, query in enumerate(queries) if query.strip()]
    query_embeddings = _encode(
        _load_model(model_name), [queries[index] for index in nonempty_indices]
    )
    if query_embeddings.shape[1] != faq_embeddings.shape[1]:
        raise ValueError("As dimensões dos embeddings de consulta e FAQ não coincidem.")

    scores = query_embeddings @ faq_embeddings.T
    results = [FAQSearch(matches=(), ambiguous=False) for _ in queries]
    for row, query_index in enumerate(nonempty_indices):
        ranked_indices = np.argsort(scores[row])[::-1][:limit]
        matches = tuple(
            FAQMatch(faq=faqs[index], score=float(scores[row, index]))
            for index in ranked_indices
            if scores[row, index] > 0 and scores[row, index] >= threshold
        )
        results[query_index] = FAQSearch(
            matches=matches,
            ambiguous=_is_ambiguous(matches, margin_ratio),
        )
    return tuple(results)


def search_faqs(
    query: str,
    *,
    limit: int = 3,
    path: Path | None = None,
    min_similarity: float | None = None,
    ambiguity_margin_ratio: float | None = None,
) -> FAQSearch:
    """Ranqueia FAQs por similaridade de cosseno entre embeddings normalizados."""
    return search_many_faqs(
        [query],
        limit=limit,
        path=path,
        min_similarity=min_similarity,
        ambiguity_margin_ratio=ambiguity_margin_ratio,
    )[0]
