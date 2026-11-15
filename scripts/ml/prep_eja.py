# Instalação: pip install python-pptx pandas openpyxl sentence-transformers scikit-learn numpy

"""Extrai slides EJA de apresentações PowerPoint e gera embeddings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.slide import Slide
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedGroupKFold


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
RANDOM_STATE = 42
LOGGER = logging.getLogger(__name__)


def _texto_limpo(texto: str) -> str:
    return "\n".join(linha.strip() for linha in texto.splitlines() if linha.strip())


def extrair_slide(slide: Slide) -> tuple[str, str]:
    """Retorna o título do slide e o texto das demais caixas de texto."""
    titulo_shape = slide.shapes.title
    titulo = _texto_limpo(titulo_shape.text) if titulo_shape is not None else ""
    id_titulo = titulo_shape.shape_id if titulo_shape is not None else None
    blocos: list[str] = []

    for shape in slide.shapes:
        if shape.shape_id == id_titulo or not shape.has_text_frame:
            continue
        texto = _texto_limpo(shape.text)
        if texto:
            blocos.append(texto)

    return titulo, "\n".join(blocos)


def extrair_apresentacoes(pasta: Path) -> pd.DataFrame:
    """Extrai todos os slides, ignorando apresentações que não puderem ser lidas."""
    if not pasta.exists() or not pasta.is_dir():
        raise FileNotFoundError(f"A pasta de apresentações não existe: {pasta}")

    arquivos = sorted(
        (caminho for caminho in pasta.rglob("*") if caminho.is_file() and caminho.suffix.lower() == ".pptx"),
        key=lambda caminho: str(caminho).casefold(),
    )
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo .pptx encontrado em: {pasta}")

    registros: list[dict[str, object]] = []
    for arquivo in arquivos:
        nome_relativo = str(arquivo.relative_to(pasta))
        try:
            apresentacao = Presentation(str(arquivo))
        except Exception:
            LOGGER.exception("Não foi possível abrir a apresentação %s; arquivo ignorado.", arquivo)
            continue

        for numero, slide in enumerate(apresentacao.slides, start=1):
            try:
                titulo, conteudo = extrair_slide(slide)
            except Exception:
                LOGGER.exception(
                    "Não foi possível extrair o slide %s de %s; será mantido sem texto.",
                    numero,
                    arquivo,
                )
                titulo, conteudo = "", ""

            registros.append(
                {
                    "arquivo": nome_relativo,
                    "slide_num": numero,
                    "titulo": titulo,
                    "conteudo": conteudo,
                    "texto_completo": "\n".join(parte for parte in (titulo, conteudo) if parte),
                    "categoria": 0,
                }
            )

    if not registros:
        raise RuntimeError(f"Nenhum slide pôde ser extraído dos arquivos .pptx em: {pasta}")

    return pd.DataFrame(
        registros,
        columns=["arquivo", "slide_num", "titulo", "conteudo", "texto_completo", "categoria"],
    )


def dividir_e_salvar(
    embeddings: np.ndarray,
    categorias: np.ndarray,
    grupos: np.ndarray,
    pasta_saida: Path = OUTPUT_DIR,
) -> None:
    """Cria splits estratificados sem separar slides da mesma apresentação."""
    embeddings = np.asarray(embeddings)
    categorias = np.asarray(categorias)
    if embeddings.ndim != 2 or categorias.ndim != 1:
        raise ValueError("Embeddings devem ser 2D e categorias devem ser 1D.")
    if len(embeddings) < 3:
        raise ValueError("São necessários pelo menos 3 slides para criar os três conjuntos.")
    if len(embeddings) != len(categorias):
        raise ValueError("A quantidade de embeddings e categorias precisa ser igual.")
    grupos = np.asarray(grupos)
    if grupos.ndim != 1 or len(grupos) != len(categorias):
        raise ValueError("Cada slide precisa ter exatamente um grupo de apresentação.")
    if any(not str(grupo).strip() for grupo in grupos):
        raise ValueError("O nome da apresentação de cada slide precisa estar preenchido.")
    if not np.issubdtype(categorias.dtype, np.number) or not np.isfinite(categorias).all():
        raise ValueError("As categorias precisam ser rótulos numéricos finitos.")
    if (
        not np.issubdtype(embeddings.dtype, np.number)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("As matrizes de embeddings contêm valores não finitos.")

    rotulos = np.unique(categorias)
    grupos_por_rotulo = [
        len(np.unique(grupos[categorias == rotulo])) for rotulo in rotulos
    ]
    menor_quantidade_grupos = min(grupos_por_rotulo)
    if menor_quantidade_grupos < 3:
        raise ValueError(
            "São necessárias pelo menos três apresentações distintas por categoria "
            "para criar treino, validação e teste sem vazamento."
        )

    outer_folds = min(10, menor_quantidade_grupos)
    outer = StratifiedGroupKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = None
    for treino_validacao_idx, teste_idx in outer.split(embeddings, categorias, grupos):
        y_treino_validacao = categorias[treino_validacao_idx]
        grupos_treino_validacao = grupos[treino_validacao_idx]
        minimo_interno = min(
            len(np.unique(grupos_treino_validacao[y_treino_validacao == rotulo]))
            for rotulo in rotulos
        )
        if minimo_interno < 2:
            continue

        inner = StratifiedGroupKFold(
            n_splits=min(9, minimo_interno),
            shuffle=True,
            random_state=RANDOM_STATE + 1,
        )
        for treino_relativo, validacao_relativa in inner.split(
            embeddings[treino_validacao_idx],
            y_treino_validacao,
            grupos_treino_validacao,
        ):
            treino_idx = treino_validacao_idx[treino_relativo]
            validacao_idx = treino_validacao_idx[validacao_relativa]
            indices = (treino_idx, validacao_idx, teste_idx)
            if any(set(np.unique(categorias[index])) != set(rotulos) for index in indices):
                continue
            grupos_split = [set(grupos[index]) for index in indices]
            if (
                grupos_split[0] & grupos_split[1]
                or grupos_split[0] & grupos_split[2]
                or grupos_split[1] & grupos_split[2]
            ):
                raise RuntimeError("Uma apresentação apareceu em mais de um conjunto.")
            splits = indices
            break
        if splits is not None:
            break

    if splits is None:
        raise ValueError(
            "Não foi possível criar splits estratificados com todas as categorias "
            "representadas. Adicione apresentações rotuladas por categoria."
        )

    treino_idx, validacao_idx, teste_idx = splits
    conjuntos = {
        "X_train_eja_emb.npy": embeddings[treino_idx],
        "X_val_eja_emb.npy": embeddings[validacao_idx],
        "X_test_eja_emb.npy": embeddings[teste_idx],
        "y_train_eja.npy": categorias[treino_idx],
        "y_val_eja.npy": categorias[validacao_idx],
        "y_test_eja.npy": categorias[teste_idx],
        "groups_train_eja.npy": grupos[treino_idx].astype(str),
        "groups_val_eja.npy": grupos[validacao_idx].astype(str),
        "groups_test_eja.npy": grupos[teste_idx].astype(str),
    }

    pasta_saida.mkdir(parents=True, exist_ok=True)
    for nome, valores in conjuntos.items():
        np.save(pasta_saida / nome, valores)
    LOGGER.info(
        "Splits agrupados salvos: %s slides de treino, %s de validação e %s de teste.",
        len(treino_idx),
        len(validacao_idx),
        len(teste_idx),
    )


def preservar_categorias(dataframe: pd.DataFrame, arquivo_excel: Path) -> pd.DataFrame:
    """Reaproveita rótulos que foram anotados na planilha de uma execução anterior."""
    if not arquivo_excel.exists():
        return dataframe

    colunas = ["arquivo", "slide_num", "categoria"]
    anotacoes = pd.read_excel(arquivo_excel, usecols=colunas).dropna(subset=colunas)
    if anotacoes.duplicated(subset=["arquivo", "slide_num"]).any():
        raise ValueError("A planilha contém mais de uma categoria para o mesmo slide.")

    valores = pd.to_numeric(anotacoes["categoria"], errors="raise")
    if not np.equal(valores, np.floor(valores)).all():
        raise ValueError("As categorias na planilha devem ser números inteiros.")

    mapa_categorias = {
        (str(arquivo), int(numero)): int(categoria)
        for arquivo, numero, categoria in zip(
            anotacoes["arquivo"],
            anotacoes["slide_num"],
            valores,
        )
    }
    chaves = zip(dataframe["arquivo"], dataframe["slide_num"])
    dataframe["categoria"] = [mapa_categorias.get(chave, 0) for chave in chaves]
    return dataframe


def preparar_dados(pasta_entrada: Path) -> Path:
    """Extrai os slides, salva a planilha e gera os embeddings divididos."""
    dataframe = extrair_apresentacoes(pasta_entrada)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arquivo_excel = OUTPUT_DIR / "slides_eja_estruturados.xlsx"
    dataframe = preservar_categorias(dataframe, arquivo_excel)
    dataframe.to_excel(arquivo_excel, index=False)

    modelo = SentenceTransformer(MODEL_NAME)
    textos = dataframe["texto_completo"].fillna("").astype(str).tolist()
    embeddings = modelo.encode(
        textos,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    categorias = dataframe["categoria"].to_numpy(dtype=np.int64)
    dividir_e_salvar(
        embeddings,
        categorias,
        dataframe["arquivo"].to_numpy(dtype=str),
    )
    LOGGER.info("Planilha e embeddings salvos em %s", OUTPUT_DIR)
    return arquivo_excel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT_DIR / "data" / "raw",
        help="Pasta com apresentações .pptx (padrão: data/raw na raiz do projeto).",
    )
    args = parser.parse_args()
    preparar_dados(args.input_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
