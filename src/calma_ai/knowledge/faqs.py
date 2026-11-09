"""Leitura e recuperação lexical das FAQs usadas para identificar materiais."""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FAQ_PATH = ROOT_DIR / "data" / "faq" / "FAQ_CONSOLIDADO.md"
AMBIGUITY_MARGIN_RATIO = 0.25

_DOCUMENT_HEADER = re.compile(r"^# Documento \d+:\s*(.+?)\s*$")
_QUESTION_HEADER = re.compile(r"^##\s+(.+?)\s*$")
_METADATA_LINE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "dos",
        "e", "em", "entre", "essa", "esse", "esta", "este", "foi", "na",
        "nas", "no", "nos", "o", "os", "para", "pela", "pelas", "pelo",
        "pelos", "por", "qual", "quais", "que", "se", "sua", "suas", "seu",
        "seus", "um", "uma", "umas", "uns",
    }
)


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


def _normalizar(texto: str) -> str:
    decomposed = unicodedata.normalize("NFKD", texto)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


def _raw_tokens(texto: str) -> list[str]:
    return _TOKEN.findall(_normalizar(texto))


def _tokens(texto: str) -> list[str]:
    return [
        token
        for token in _raw_tokens(texto)
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _features(texto: str) -> Counter[str]:
    raw_tokens = _raw_tokens(texto)
    tokens = [token for token in raw_tokens if len(token) > 1 and token not in _STOPWORDS]
    content_tokens = set(tokens)
    features: Counter[str] = Counter(f"w:{token}" for token in tokens)
    features.update(
        f"b:{left}_{right}"
        for left, right in zip(raw_tokens, raw_tokens[1:])
        if left in content_tokens and right in content_tokens
    )
    return features


def _weighted_features(faq: FAQ) -> Counter[str]:
    features: Counter[str] = Counter()
    for text, weight in (
        (faq.question, 2.0),
        (faq.answer, 1.0),
        (faq.topic, 1.5),
        (faq.document, 0.75),
    ):
        for feature, count in _features(text).items():
            features[feature] += count * weight
    return features


def _normalize_vector(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if not norm:
        return {}
    return {feature: value / norm for feature, value in vector.items()}


@lru_cache(maxsize=4)
def _load_index(path: Path, modified_ns: int, size: int) -> tuple[
    tuple[FAQ, ...], tuple[dict[str, float], ...], dict[str, float]
]:
    del modified_ns, size
    faqs = parse_faqs(path.read_text(encoding="utf-8"))
    raw_vectors = tuple(_weighted_features(faq) for faq in faqs)
    document_frequency: Counter[str] = Counter(
        feature for vector in raw_vectors for feature in vector
    )
    total_documents = len(faqs)
    inverse_document_frequency = {
        feature: math.log((total_documents + 1) / (frequency + 1)) + 1
        for feature, frequency in document_frequency.items()
    }
    vectors = tuple(
        _normalize_vector(
            {
                feature: (1 + math.log(count)) * inverse_document_frequency[feature]
                for feature, count in vector.items()
            }
        )
        for vector in raw_vectors
    )
    return faqs, vectors, inverse_document_frequency


def search_faqs(
    query: str,
    *,
    limit: int = 3,
    path: Path | None = None,
) -> FAQSearch:
    """Ranqueia FAQs por cosseno TF-IDF; não cria vetores densos nem divide pares."""
    if limit < 1:
        raise ValueError("O limite de resultados precisa ser positivo.")
    corpus_path = (path or faq_path()).resolve()
    stat = corpus_path.stat()
    faqs, vectors, idf = _load_index(corpus_path, stat.st_mtime_ns, stat.st_size)
    query_features = _features(query)
    query_vector = _normalize_vector(
        {
            feature: (1 + math.log(count)) * idf[feature]
            for feature, count in query_features.items()
            if feature in idf
        }
    )
    if not query_vector:
        return FAQSearch(matches=(), ambiguous=False)

    query_words = {feature for feature in query_features if feature.startswith("w:")}

    def has_enough_query_terms(faq: FAQ, vector: dict[str, float]) -> bool:
        shared_words = query_words.intersection(vector)
        if len(shared_words) >= 2:
            return True
        if len(query_words) == 1:
            term = next(iter(query_words))[2:]
            return term in _tokens(faq.question) or term in _tokens(faq.topic)
        return False

    ranked = sorted(
        (
            FAQMatch(
                faq=faq,
                score=sum(
                    weight * vector.get(feature, 0.0)
                    for feature, weight in query_vector.items()
                ),
            )
            for faq, vector in zip(faqs, vectors)
            if has_enough_query_terms(faq, vector)
        ),
        key=lambda result: result.score,
        reverse=True,
    )
    matches = tuple(result for result in ranked if result.score > 0)[:limit]
    one_term_spans_topics = (
        len(query_words) == 1 and len({match.faq.topic for match in matches}) > 1
    )
    close_scores = (
        len(matches) > 1
        and matches[0].score - matches[1].score
        <= matches[0].score * AMBIGUITY_MARGIN_RATIO
    )
    ambiguous = one_term_spans_topics or close_scores
    return FAQSearch(matches=matches, ambiguous=ambiguous)
