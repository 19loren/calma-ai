# Instalação: pip install numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

"""Treina e avalia um MLP para classificação de materiais didáticos da EJA."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, learning_curve
from sklearn.neural_network import MLPClassifier


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
GRAPH_DIR = ROOT_DIR / "resultados" / "sup" / "graficos"
MODEL_PATH = DATA_DIR / "modelo_mlp_eja.pkl"
RANDOM_STATE = 42
LOGGER = logging.getLogger(__name__)


def carregar_conjuntos(pasta: Path = DATA_DIR) -> tuple[np.ndarray, ...]:
    """Carrega os três conjuntos e verifica isolamento por apresentação."""
    nomes = (
        "X_train_eja_emb.npy",
        "y_train_eja.npy",
        "groups_train_eja.npy",
        "X_val_eja_emb.npy",
        "y_val_eja.npy",
        "groups_val_eja.npy",
        "X_test_eja_emb.npy",
        "y_test_eja.npy",
        "groups_test_eja.npy",
    )
    arquivos = {nome: pasta / nome for nome in nomes}
    ausentes = [str(caminho) for caminho in arquivos.values() if not caminho.is_file()]
    if ausentes:
        raise FileNotFoundError("Arquivos de dados ausentes: " + ", ".join(ausentes))

    (
        X_train,
        y_train,
        groups_train,
        X_val,
        y_val,
        groups_val,
        X_test,
        y_test,
        groups_test,
    ) = (
        np.load(arquivos[nome], allow_pickle=False) for nome in nomes
    )
    if X_train.ndim != 2 or X_val.ndim != 2 or X_test.ndim != 2:
        raise ValueError("As matrizes de embeddings devem ser bidimensionais.")
    if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Treino, validação e teste devem ter a mesma dimensão.")
    if y_train.ndim != 1 or y_val.ndim != 1 or y_test.ndim != 1:
        raise ValueError("Os vetores de categorias devem ser unidimensionais.")
    if (
        len(X_train) != len(y_train)
        or len(X_val) != len(y_val)
        or len(X_test) != len(y_test)
    ):
        raise ValueError("Cada embedding precisa ter exatamente uma categoria.")
    if len(X_train) < 2 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Treino precisa ter ao menos duas amostras; val/test não podem ser vazios.")
    for groups, labels, name in (
        (groups_train, y_train, "treino"),
        (groups_val, y_val, "validação"),
        (groups_test, y_test, "teste"),
    ):
        if groups.ndim != 1 or len(groups) != len(labels):
            raise ValueError(f"Cada amostra de {name} precisa ter um grupo.")
    sets = [set(groups_train), set(groups_val), set(groups_test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Há apresentações compartilhadas entre treino, validação e teste.")
    if (
        not np.isfinite(X_train).all()
        or not np.isfinite(X_val).all()
        or not np.isfinite(X_test).all()
    ):
        raise ValueError("As matrizes de embeddings contêm valores não finitos.")

    return (
        X_train,
        y_train,
        groups_train,
        X_val,
        y_val,
        groups_val,
        X_test,
        y_test,
        groups_test,
    )


def balancear_treino(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica SMOTE quando há classes suficientes para sintetizar amostras."""
    classes, contagens = np.unique(y_train, return_counts=True)
    if len(classes) < 2:
        LOGGER.warning(
            "SMOTE não foi aplicado: o treino tem somente uma categoria. "
            "Atribua rótulos variados para treinar uma classificação real."
        )
        return X_train, y_train

    menor_classe = int(contagens.min())
    if menor_classe < 2:
        LOGGER.warning(
            "SMOTE não foi aplicado: cada categoria precisa ter pelo menos "
            "duas amostras no treino."
        )
        return X_train, y_train

    vizinhos = min(5, menor_classe - 1)
    X_balanceado, y_balanceado = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=vizinhos,
    ).fit_resample(X_train, y_train)
    LOGGER.info("SMOTE aplicado: %s amostras após o balanceamento.", len(y_balanceado))
    return X_balanceado, y_balanceado


def criar_mlp(*, early_stopping: bool = False) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=0.01,
        early_stopping=early_stopping,
        max_iter=500,
        random_state=RANDOM_STATE,
    )


def salvar_matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    rotulos = np.unique(np.concatenate((y_true, y_pred)))
    matriz = confusion_matrix(y_true, y_pred, labels=rotulos)

    figura, eixo = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=rotulos,
        yticklabels=rotulos,
        ax=eixo,
    )
    eixo.set(
        xlabel="Categoria prevista",
        ylabel="Categoria verdadeira",
        title="Matriz de confusão - MLP EJA",
    )
    figura.tight_layout()
    destino = GRAPH_DIR / "matriz_confusao_mlp_eja.png"
    figura.savefig(destino, dpi=300)
    plt.close(figura)
    return destino


def salvar_curva_aprendizado(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
) -> Path | None:
    """Gera curva F1-Macro com folds agrupados por apresentação."""
    if len(y_train) < 2 or len(groups_train) != len(y_train):
        LOGGER.warning("Curva de aprendizado não gerada: treino insuficiente para CV.")
        return None

    rotulos = np.unique(y_train)
    grupos_por_rotulo = [
        len(np.unique(groups_train[y_train == rotulo])) for rotulo in rotulos
    ]
    divisores = min(5, min(grupos_por_rotulo))
    if divisores < 2:
        LOGGER.warning(
            "Curva de aprendizado não gerada: cada categoria precisa estar em "
            "pelo menos duas apresentações no treino."
        )
        return None
    cv = StratifiedGroupKFold(
        n_splits=divisores,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    tamanhos, pontuacoes_treino, pontuacoes_validacao = learning_curve(
        estimator=criar_mlp(early_stopping=False),
        X=X_train,
        y=y_train,
        groups=groups_train,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )
    media_treino = pontuacoes_treino.mean(axis=1)
    media_validacao = pontuacoes_validacao.mean(axis=1)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    figura, eixo = plt.subplots(figsize=(8, 6))
    eixo.plot(tamanhos, media_treino, marker="o", label="Treino")
    eixo.plot(tamanhos, media_validacao, marker="o", label="Validação cruzada")
    eixo.set(
        xlabel="Número de amostras de treino",
        ylabel="F1-Macro",
        title="Curva de aprendizado - MLP EJA",
    )
    eixo.set_ylim(0, 1.05)
    eixo.grid(True, alpha=0.3)
    eixo.legend()
    figura.tight_layout()
    destino = GRAPH_DIR / "curva_aprendizado_f1_macro_mlp_eja.png"
    figura.savefig(destino, dpi=300)
    plt.close(figura)
    return destino


def treinar_e_avaliar() -> MLPClassifier:
    (
        X_train,
        y_train,
        groups_train,
        X_val,
        y_val,
        _groups_val,
        X_test,
        y_test,
        _groups_test,
    ) = carregar_conjuntos()
    X_balanceado, y_balanceado = balancear_treino(X_train, y_train)

    modelo = criar_mlp()
    modelo.fit(X_balanceado, y_balanceado)
    previsoes_validacao = modelo.predict(X_val)
    previsoes_teste = modelo.predict(X_test)

    print("Relatório de classificação (validação):")
    print(classification_report(y_val, previsoes_validacao, zero_division=0))
    print("Relatório de classificação (teste final):")
    print(classification_report(y_test, previsoes_teste, zero_division=0))
    f1_validacao = f1_score(
        y_val, previsoes_validacao, average="macro", zero_division=0
    )
    f1_teste = f1_score(y_test, previsoes_teste, average="macro", zero_division=0)

    matriz_path = salvar_matriz_confusao(y_test, previsoes_teste)
    curva_path = salvar_curva_aprendizado(X_train, y_train, groups_train)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)

    LOGGER.info("Matriz de confusão salva em %s", matriz_path)
    if curva_path is not None:
        LOGGER.info("Curva de aprendizado salva em %s", curva_path)
    LOGGER.info("Modelo salvo em %s", MODEL_PATH)
    LOGGER.info("F1-Macro na validação: %.4f", f1_validacao)
    LOGGER.info("F1-Macro no teste final: %.4f", f1_teste)
    return modelo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    treinar_e_avaliar()
