"""Generate an answer with the locally fine-tuned FAQ QA model."""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "faq_qa_ptt5"
SYSTEM_PROMPT = (
    "Você é um assistente de estudos para estudantes da EJA (Educação de Jovens e Adultos). "
    "Responda apenas com base no contexto fornecido, em linguagem simples e clara. "
    "Se a resposta não estiver no contexto, diga que não encontrou a informação no material."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexto", required=True, help="Trecho de material para consultar.")
    parser.add_argument("--pergunta", required=True, help="Pergunta a responder com base no contexto.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens deve ser positivo.")
    if not (args.model_dir / "config.json").is_file():
        parser.error(
            f"Modelo não encontrado em {args.model_dir}. "
            "Treine primeiro com scripts\\ml\\treino_faq_qa.py."
        )

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    source = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Contexto:\n{args.contexto.strip()}\n\n"
        f"Pergunta: {args.pergunta.strip()}"
    )
    inputs = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=4,
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
