"""Acesso à base de conhecimento pedagógica, sem depender do CrewAI."""

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

ARQUIVO_CONTEUDOS = Path(__file__).resolve().parent / "knowledge" / "conteudos.json"


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower().replace("_", " ").strip()


@lru_cache(maxsize=1)
def carregar_conteudos() -> dict[str, dict]:
    return json.loads(ARQUIVO_CONTEUDOS.read_text(encoding="utf-8"))


def buscar_topico(consulta: str) -> tuple[str, dict] | None:
    conteudos = carregar_conteudos()
    alvo = _normalizar(consulta)
    melhor: tuple[int, str] | None = None
    for chave, dados in conteudos.items():
        if _normalizar(chave) == alvo:
            return chave, dados
        pontos = sum(
            1 for palavra in dados.get("palavras_chave", []) if _normalizar(palavra) in alvo
        )
        if pontos and (melhor is None or pontos > melhor[0]):
            melhor = (pontos, chave)
    if melhor is None:
        return None
    return melhor[1], conteudos[melhor[1]]
