
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load a multilingual emotion‑BERT once (cached)
_TOKENIZER = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
_MODEL = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]   # model’s output order


def _detect_emotion(text: str) -> str:
    """Return the label with highest soft‑max score."""
    inputs = _TOKENIZER(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = _MODEL(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    idx = int(probs.argmax())
    return _LABELS[idx]

class EmotionInput(BaseModel):
    estudante_id: str = Field(description="ID do estudante (para eventual logging).")
    mensagem: str = Field(description="Texto que o estudante acabou de enviar.")


class DetectEmocaoTool(BaseTool):
    name: str = "detectar_emocao"
    description: str = (
        "Analisa o texto da mensagem do estudante e devolve a emoção predominante "
        "entre: sadness, joy, love, anger, fear, surprise."
    )
    args_schema: type[BaseModel] = EmotionInput

    def _run(self, estudante_id: str, mensagem: str) -> str:
        # Opcional: persist last emotion in the student profile
        try:
            emocao = _detect_emotion(mensagem)
        except Exception as exc:
            return json.dumps({"error": f"Falha ao detectar emoção: {exc}"})

        # Simple persistence (append to profile)
        from calma_ai.tools.progresso import carregar_perfil, _arquivo_perfil

        perfil = carregar_perfil(estudante_id)
        perfil["emocao_recente"] = emocao
        # Serialize back (use the same safe‑write pattern as in progresso.py)
        arquivo = _arquivo_perfil(estudante_id)
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        arquivo.write_text(json.dumps(perfil, ensure_ascii=False, indent=2), encoding="utf-8")

        return json.dumps({"emocao": emocao})