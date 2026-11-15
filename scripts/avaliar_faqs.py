"""Avalia recuperação neural das FAQs e calibra limiares usando somente validação."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from calma_ai.knowledge.faqs import FAQSearch, parse_faqs, search_many_faqs

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT_DIR / "data" / "datasets" / "faq_qa"
DEFAULT_FAQ_PATH = ROOT_DIR / "data" / "faq" / "FAQ_CONSOLIDADO.md"
QUESTION_PATTERN = re.compile(r"(?im)^\s*Pergunta:\s*(.+?)\s*$")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Example:
    faq_id: str
    context_supported: bool
    query: str


def load_split(path: Path) -> list[Example]:
    examples: list[Example] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            metadata = row["meta"]
            if not isinstance(metadata, dict) or not isinstance(row["messages"], list):
                raise TypeError("metadados ou mensagens têm formato inválido")
            user_message = next(
                message["content"]
                for message in row["messages"]
                if message["role"] == "user"
            )
        except (KeyError, StopIteration, TypeError, json.JSONDecodeError) as erro:
            raise ValueError(f"Registro inválido em {path}:{line_number}.") from erro

        match = QUESTION_PATTERN.search(user_message)
        if not match:
            raise ValueError(f"Pergunta ausente em {path}:{line_number}.")
        kind = metadata.get("tipo")
        if kind not in {"positivo", "negativo"}:
            raise ValueError(f"Rótulo inválido em {path}:{line_number}: {kind!r}.")
        faq_id = metadata.get("id")
        if not isinstance(faq_id, str) or not faq_id:
            raise ValueError(f"ID de FAQ ausente em {path}:{line_number}.")
        examples.append(
            Example(
                faq_id=faq_id,
                context_supported=kind == "positivo",
                query=match.group(1),
            )
        )
    if not examples:
        raise ValueError(f"Conjunto vazio: {path}")
    return examples


def _is_ambiguous(result: FAQSearch, margin_ratio: float) -> bool:
    matches = result.matches
    return (
        len(matches) > 1
        and matches[0].score - matches[1].score
        <= matches[0].score * margin_ratio
    )


def score_results(
    examples: list[Example],
    results: tuple[FAQSearch, ...],
    *,
    corpus_ids: set[str],
    min_similarity: float,
    ambiguity_margin_ratio: float,
) -> dict[str, int | float]:
    if len(examples) != len(results):
        raise ValueError("A quantidade de consultas e resultados precisa ser igual.")
    true_positives = false_positives = false_negatives = 0
    retrievable_targets = out_of_corpus = 0
    supported_targets = supported_hits = unsupported_targets = unsupported_hits = 0

    for example, result in zip(examples, results):
        expected_available = example.faq_id in corpus_ids
        if not expected_available:
            out_of_corpus += 1
        else:
            retrievable_targets += 1
            if example.context_supported:
                supported_targets += 1
            else:
                unsupported_targets += 1

        candidates = tuple(
            match for match in result.matches if match.score >= min_similarity
        )
        accepted = bool(candidates) and not _is_ambiguous(
            FAQSearch(matches=candidates, ambiguous=False), ambiguity_margin_ratio
        )
        if not accepted:
            if expected_available:
                false_negatives += 1
            continue

        if expected_available and candidates[0].faq.id == example.faq_id:
            true_positives += 1
            if example.context_supported:
                supported_hits += 1
            else:
                unsupported_hits += 1
        else:
            false_positives += 1
            if expected_available:
                false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (
        true_positives + false_positives
    ) else 0.0
    recall = true_positives / retrievable_targets if retrievable_targets else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "retrievable_targets": retrievable_targets,
        "out_of_corpus_targets": out_of_corpus,
        "context_supported_targets": supported_targets,
        "context_supported_hits": supported_hits,
        "context_unsupported_targets": unsupported_targets,
        "context_unsupported_hits": unsupported_hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calibrate(
    examples: list[Example],
    results: tuple[FAQSearch, ...],
    *,
    corpus_ids: set[str],
) -> tuple[float, float, dict[str, int | float]]:
    best: (
        tuple[tuple[float, float, float, float], float, float, dict[str, int | float]]
        | None
    ) = None
    thresholds = [step / 40 for step in range(1, 41)]
    ambiguity_margins = [step / 40 for step in range(0, 41)]
    for threshold in thresholds:
        for margin in ambiguity_margins:
            metrics = score_results(
                examples,
                results,
                corpus_ids=corpus_ids,
                min_similarity=threshold,
                ambiguity_margin_ratio=margin,
            )
            key = (
                float(metrics["f1"]),
                float(metrics["precision"]),
                threshold,
                margin,
            )
            if best is None or key > best[0]:
                best = (key, threshold, margin, metrics)
    if best is None:
        raise RuntimeError("Não foi possível calibrar os limiares.")
    return best[1], best[2], best[3]


def avaliar(dataset_dir: Path, faq_path: Path) -> tuple[float, float]:
    faqs = parse_faqs(faq_path.read_text(encoding="utf-8"))
    corpus_ids = {faq.id for faq in faqs}
    loaded = {split: load_split(dataset_dir / f"{split}.jsonl") for split in SPLITS}
    searches = {
        split: search_many_faqs(
            [example.query for example in examples],
            limit=3,
            path=faq_path,
            min_similarity=0.0,
            ambiguity_margin_ratio=0.0,
        )
        for split, examples in loaded.items()
    }

    threshold, margin, _ = calibrate(
        loaded["val"], searches["val"], corpus_ids=corpus_ids
    )
    print(
        "Limiares calibrados em val: "
        f"similaridade mínima={threshold:.3f}, margem de ambiguidade={margin:.3f}"
    )
    for split in SPLITS:
        metrics = score_results(
            loaded[split],
            searches[split],
            corpus_ids=corpus_ids,
            min_similarity=threshold,
            ambiguity_margin_ratio=margin,
        )
        print(
            f"{split}: P@1={metrics['precision']:.3f}, "
            f"R@1={metrics['recall']:.3f}, F1@1={metrics['f1']:.3f}, "
            f"TP={metrics['true_positives']}, FP={metrics['false_positives']}, "
            f"FN={metrics['false_negatives']}, "
            f"IDs fora do corpus={metrics['out_of_corpus_targets']}, "
            f"hit contexto+={metrics['context_supported_hits']}/"
            f"{metrics['context_supported_targets']}, "
            f"hit contexto-={metrics['context_unsupported_hits']}/"
            f"{metrics['context_unsupported_targets']}"
        )
    print(
        "Nota: `meta.tipo` rotula suporte no contexto fornecido, não relevância "
        "do retriever; IDs presentes na base são alvos de recuperação nos dois "
        "rótulos. Limiares usam somente val; test é a avaliação final. IDs ausentes "
        "da base servem como casos de abstenção."
    )
    return threshold, margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--faq-path", type=Path, default=DEFAULT_FAQ_PATH)
    args = parser.parse_args()
    avaliar(args.dataset_dir, args.faq_path)


if __name__ == "__main__":
    main()
