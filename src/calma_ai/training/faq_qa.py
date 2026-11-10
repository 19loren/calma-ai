"""Loading, validating, and scoring the contextual FAQ QA dataset."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FAQQAExample:
    sample_id: str
    label: str
    source: str
    answer: str


def load_faq_qa_dataset(path: Path) -> list[FAQQAExample]:
    """Load JSONL conversations as system+user input and assistant target."""
    examples: list[FAQQAExample] = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                messages = record["messages"]
                metadata = record["meta"]
                by_role = {message["role"]: message["content"] for message in messages}
                system = by_role["system"].strip()
                user = by_role["user"].strip()
                answer = by_role["assistant"].strip()
                sample_id = metadata["id"]
                label = metadata["tipo"]
            except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Registro inválido em {path}, linha {line_number}.") from error

            if not all(isinstance(value, str) and value.strip() for value in (
                system, user, answer, sample_id, label
            )):
                raise ValueError(
                    f"Campos system/user/assistant/id/tipo vazios ou inválidos "
                    f"em {path}, linha {line_number}."
                )
            if label not in {"positivo", "negativo"}:
                raise ValueError(
                    f"Rótulo meta.tipo desconhecido em {path}, linha {line_number}: {label!r}."
                )

            examples.append(
                FAQQAExample(
                    sample_id=sample_id,
                    label=label,
                    source=f"{system}\n\n{user}",
                    answer=answer,
                )
            )

    if not examples:
        raise ValueError(f"O conjunto de dados está vazio: {path}")
    return examples


def validate_disjoint_ids(
    train: Iterable[FAQQAExample],
    validation: Iterable[FAQQAExample],
    test: Iterable[FAQQAExample],
) -> None:
    """Prevent leakage of a source FAQ ID between dataset partitions."""
    partitions = {
        "treino": {example.sample_id for example in train},
        "validação": {example.sample_id for example in validation},
        "teste": {example.sample_id for example in test},
    }
    names = tuple(partitions)
    for index, name in enumerate(names):
        for other_name in names[index + 1:]:
            overlap = partitions[name].intersection(partitions[other_name])
            if overlap:
                sample_ids = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"IDs de FAQ aparecem em mais de uma partição "
                    f"({name}/{other_name}): {sample_ids}."
                )


def _normalize_answer(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def calculate_metrics(
    references: list[str],
    predictions: list[str],
    labels: list[str],
) -> dict[str, float]:
    """Calculate normalized exact match and token F1, overall and by label."""
    if not references or len(references) != len(predictions) or len(labels) != len(references):
        raise ValueError("Referências, previsões e rótulos devem ter o mesmo tamanho não vazio.")

    metrics: dict[str, float] = {}
    groups = {
        "geral": list(range(len(references))),
        **{
            label: [index for index, value in enumerate(labels) if value == label]
            for label in sorted(set(labels))
        },
    }
    for group, indices in groups.items():
        if not indices:
            continue
        exact_matches = 0
        f1_scores: list[float] = []
        for index in indices:
            expected = _normalize_answer(references[index])
            predicted = _normalize_answer(predictions[index])
            exact_matches += expected == predicted

            expected_tokens = Counter(expected.split())
            predicted_tokens = Counter(predicted.split())
            common = sum((expected_tokens & predicted_tokens).values())
            if not expected_tokens and not predicted_tokens:
                f1_scores.append(1.0)
            elif not common:
                f1_scores.append(0.0)
            else:
                precision = common / sum(predicted_tokens.values())
                recall = common / sum(expected_tokens.values())
                f1_scores.append(2 * precision * recall / (precision + recall))

        metrics[f"{group}_exact_match"] = exact_matches / len(indices)
        metrics[f"{group}_token_f1"] = sum(f1_scores) / len(f1_scores)
    return metrics
