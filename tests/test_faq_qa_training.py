import json
from pathlib import Path

import pytest

from calma_ai.training.faq_qa import (
    FAQQAExample,
    calculate_metrics,
    load_faq_qa_dataset,
    validate_disjoint_ids,
)


def _record(sample_id="faq-1", label="positivo", answer="Resposta correta."):
    return {
        "messages": [
            {"role": "system", "content": "Use apenas o contexto."},
            {"role": "user", "content": "Contexto: texto.\n\nPergunta: qual é a resposta?"},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"id": sample_id, "tipo": label},
    }


def test_loads_context_and_question_as_input_and_assistant_as_target(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8")

    examples = load_faq_qa_dataset(dataset)

    assert examples == [
        FAQQAExample(
            sample_id="faq-1",
            label="positivo",
            source="Use apenas o contexto.\n\nContexto: texto.\n\nPergunta: qual é a resposta?",
            answer="Resposta correta.",
        )
    ]


def test_rejects_invalid_record_and_unknown_label(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text('{"messages": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="linha 1"):
        load_faq_qa_dataset(dataset)

    dataset.write_text(json.dumps(_record(label="indefinido")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Rótulo"):
        load_faq_qa_dataset(dataset)


def test_prevents_faq_id_overlap_between_splits():
    shared = FAQQAExample("faq-1", "positivo", "pergunta", "resposta")
    with pytest.raises(ValueError, match="mais de uma partição"):
        validate_disjoint_ids([shared], [], [shared])


def test_calculates_exact_match_and_token_f1():
    metrics = calculate_metrics(
        ["São Paulo tem 12 milhões.", "Não encontrei essa informação."],
        ["Sao Paulo tem 12 milhoes", "Não encontrei."],
        ["positivo", "negativo"],
    )

    assert metrics["geral_exact_match"] == 0.5
    assert metrics["positivo_exact_match"] == 1.0
    assert metrics["negativo_token_f1"] == pytest.approx(2 / 3)


def test_repository_jsonl_splits_load_and_have_no_faq_id_leakage():
    dataset_dir = Path(__file__).resolve().parents[1] / "data" / "datasets" / "faq_qa"
    train = load_faq_qa_dataset(dataset_dir / "train.jsonl")
    validation = load_faq_qa_dataset(dataset_dir / "val.jsonl")
    test = load_faq_qa_dataset(dataset_dir / "test.jsonl")

    assert (len(train), len(validation), len(test)) == (640, 136, 132)
    validate_disjoint_ids(train, validation, test)
