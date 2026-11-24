"""Recuperação de exemplos pedagógicos do split de treino GSM8K."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
ARQUIVO_TREINO = (
    ROOT_DIR
    / "notebooks"
    / "eja_diagnosticador_dataset"
    / "data"
    / "final"
    / "train.jsonl"
)
SEM_EXEMPLO = (
    "Nenhum exemplo do treino teve semelhança lexical suficiente. "
    "Faça o diagnóstico a partir da dúvida atual e das ferramentas disponíveis."
)
_STOPWORDS = frozenset(
    {
        "com", "como", "das", "dos", "ela", "ele", "era", "essa", "esse", "foi",
        "isso", "mais", "mas", "meu", "minha", "para", "pela", "pelo", "qual",
        "quando", "quanto", "que", "quem", "sua", "seu", "sobre", "tambem", "tem",
        "uma", "umas", "uns", "voce",
    }
)


@dataclass(frozen=True)
class ExemploDiagnosticador:
    erro_tipo: str
    dificuldade: str
    tentativa: str
    orientacao: str


def _tokens(texto: str) -> set[str]:
    sem_acento = (
        unicodedata.normalize("NFKD", texto)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", sem_acento)
        if token not in _STOPWORDS
    }


@lru_cache(maxsize=4)
def _carregar_treino(
    caminho: str,
    modified_ns: int,
    tamanho: int,
) -> tuple[ExemploDiagnosticador, ...]:
    del modified_ns, tamanho
    arquivo = Path(caminho)
    if not arquivo.is_file():
        raise FileNotFoundError(
            f"Dataset de exemplos do diagnosticador não encontrado: {arquivo}. "
            "Execute o notebook 05 para gerar o split train."
        )

    exemplos: list[ExemploDiagnosticador] = []
    for numero_linha, linha in enumerate(
        arquivo.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not linha.strip():
            continue
        try:
            registro = json.loads(linha)
        except json.JSONDecodeError as erro:
            raise ValueError(
                f"JSON inválido em {arquivo}:{numero_linha}."
            ) from erro
        if not isinstance(registro, dict):
            raise ValueError(
                f"Registro JSON precisa ser um objeto em {arquivo}:{numero_linha}."
            )

        entrada = registro.get("entrada")
        saida = registro.get("saida")
        dpo = registro.get("dpo")
        if not all(isinstance(valor, dict) for valor in (entrada, saida, dpo)):
            raise ValueError(
                f"Registro sem entrada/saída/DPO em {arquivo}:{numero_linha}."
            )

        campos = (
            registro.get("erro_tipo"),
            saida.get("dificuldade"),
            entrada.get("duvida"),
            dpo.get("chosen"),
        )
        if not all(isinstance(valor, str) and valor.strip() for valor in campos):
            raise ValueError(
                f"Campos obrigatórios inválidos em {arquivo}:{numero_linha}."
            )

        exemplos.append(
            ExemploDiagnosticador(
                erro_tipo=campos[0],
                dificuldade=campos[1],
                tentativa=campos[2],
                orientacao=campos[3],
            )
        )

    if not exemplos:
        raise ValueError(f"O split de treino não contém exemplos: {arquivo}.")
    return tuple(exemplos)


def formatar_exemplos_diagnosticador(
    duvida: str,
    *,
    limite: int = 2,
    arquivo: Path = ARQUIVO_TREINO,
) -> str:
    """Seleciona exemplos lexicais do treino e inclui somente orientações preferidas."""
    if not isinstance(duvida, str):
        raise TypeError("A dúvida do estudante precisa ser texto.")
    if limite < 1:
        raise ValueError("O limite de exemplos precisa ser positivo.")

    caminho = arquivo.resolve()
    stat = caminho.stat() if caminho.exists() else None
    exemplos = _carregar_treino(
        str(caminho),
        stat.st_mtime_ns if stat else 0,
        stat.st_size if stat else 0,
    )
    tokens_duvida = _tokens(duvida)
    if not tokens_duvida:
        return SEM_EXEMPLO

    ranqueados: list[tuple[float, int, ExemploDiagnosticador]] = []
    for indice, exemplo in enumerate(exemplos):
        tokens_exemplo = _tokens(exemplo.tentativa)
        uniao = tokens_duvida | tokens_exemplo
        if not uniao:
            continue
        semelhanca = len(tokens_duvida & tokens_exemplo) / len(uniao)
        if semelhanca > 0:
            ranqueados.append((semelhanca, indice, exemplo))

    if not ranqueados:
        return SEM_EXEMPLO

    selecionados = sorted(ranqueados, key=lambda item: (-item[0], item[1]))[:limite]
    linhas = []
    for numero, (_, _, exemplo) in enumerate(selecionados, start=1):
        tentativa = " ".join(exemplo.tentativa.split())
        if len(tentativa) > 240:
            tentativa = tentativa[:237].rsplit(" ", 1)[0] + "..."
        linhas.append(
            f"Exemplo sintético {numero} ({exemplo.erro_tipo}; "
            f"{exemplo.dificuldade})\n"
            f"Tentativa: {tentativa}\n"
            f"Orientação preferida: {exemplo.orientacao}"
        )
    return "\n\n".join(linhas)
