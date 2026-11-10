"""Fine-tune a Portuguese text-to-text model using the FAQ QA JSONL splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from calma_ai.training.faq_qa import (
    FAQQAExample,
    calculate_metrics,
    load_faq_qa_dataset,
    validate_disjoint_ids,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "datasets" / "faq_qa"
OUTPUT_DIR = ROOT_DIR / "models" / "faq_qa_ptt5"
DEFAULT_MODEL = "unicamp-dl/ptt5-small-portuguese-vocab"
LOGGER = logging.getLogger(__name__)


def _tokenize_examples(
    examples: list[FAQQAExample],
    tokenizer: Any,
    *,
    source_length: int,
    target_length: int,
) -> list[dict[str, list[int]]]:
    tokenized: list[dict[str, list[int]]] = []
    for example in examples:
        source = tokenizer(
            example.source,
            truncation=True,
            max_length=source_length,
        )
        target = tokenizer(
            text_target=example.answer,
            truncation=True,
            max_length=target_length,
        )
        tokenized.append(
            {
                "input_ids": source["input_ids"],
                "attention_mask": source["attention_mask"],
                "labels": target["input_ids"],
            }
        )
    return tokenized


def _evaluate_test_set(
    model: Any,
    tokenizer: Any,
    examples: list[FAQQAExample],
    *,
    source_length: int,
    target_length: int,
    batch_size: int,
) -> dict[str, float]:
    import torch

    predictions: list[str] = []
    model.eval()
    device = next(model.parameters()).device
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        encoded = tokenizer(
            [example.source for example in batch],
            padding=True,
            truncation=True,
            max_length=source_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=target_length,
                num_beams=4,
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    metrics = calculate_metrics(
        [example.answer for example in examples],
        predictions,
        [example.label for example in examples],
    )
    metrics["examples_test"] = float(len(examples))
    return metrics


def train(
    *,
    model_name: str = DEFAULT_MODEL,
    output_dir: Path = OUTPUT_DIR,
    epochs: int = 5,
    batch_size: int = 4,
    source_length: int = 512,
    target_length: int = 128,
) -> dict[str, float]:
    train_examples = load_faq_qa_dataset(DATA_DIR / "train.jsonl")
    validation_examples = load_faq_qa_dataset(DATA_DIR / "val.jsonl")
    test_examples = load_faq_qa_dataset(DATA_DIR / "test.jsonl")
    validate_disjoint_ids(train_examples, validation_examples, test_examples)

    try:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Treinamento requer as dependências extras: "
            "pip install -e \".[faq-qa]\""
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    train_data = _tokenize_examples(
        train_examples, tokenizer, source_length=source_length, target_length=target_length
    )
    validation_data = _tokenize_examples(
        validation_examples, tokenizer, source_length=source_length, target_length=target_length
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = _evaluate_test_set(
        trainer.model,
        tokenizer,
        test_examples,
        source_length=source_length,
        target_length=target_length,
        batch_size=batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Modelo salvo em %s", output_dir)
    LOGGER.info("Métricas do conjunto de teste: %s", json.dumps(metrics, ensure_ascii=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Seq2Seq pré-treinado.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--source-length", type=int, default=512)
    parser.add_argument("--target-length", type=int, default=128)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.source_length, args.target_length) < 1:
        parser.error("épocas, batch size e comprimentos devem ser positivos.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        source_length=args.source_length,
        target_length=args.target_length,
    )


if __name__ == "__main__":
    main()
