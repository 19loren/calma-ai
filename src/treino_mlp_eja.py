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
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve
from sklearn.neural_network import MLPClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
GRAPH_DIR = ROOT_DIR / "resultados" / "sup" / "graficos"
MODEL_PATH = DATA_DIR / "modelo_mlp_eja.pkl"
RANDOM_STATE = 42
LOGGER = logging.getLogger(__name__)


def carregar_conjuntos(pasta: Path = DATA_DIR) -> tuple[np.ndarray, ...]:
    """Carrega embeddings e rótulos de treino e validação."""
    nomes = (
        "X_train_eja_emb.npy",
        "y_train_eja.npy",
        "X_val_eja_emb.npy",
        "y_val_eja.npy",
    )
    arquivos = {nome: pasta / nome for nome in nomes}
    ausentes = [str(caminho) for caminho in arquivos.values() if not caminho.is_file()]
    if ausentes:
        raise FileNotFoundError("Arquivos de dados ausentes: " + ", ".join(ausentes))

    X_train, y_train, X_val, y_val = (
        np.load(arquivos[nome], allow_pickle=False) for nome in nomes
    )
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("As matrizes de embeddings devem ser bidimensionais.")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Treino e validação devem ter a mesma dimensão de embedding.")
    if y_train.ndim != 1 or y_val.ndim != 1:
        raise ValueError("Os vetores de categorias devem ser unidimensionais.")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Cada embedding precisa ter exatamente uma categoria.")
    if len(X_train) < 10:
        raise ValueError(
            "O conjunto de treino precisa ter pelo menos 10 amostras para usar "
            "early_stopping=True."
        )
    if len(X_val) == 0:
        raise ValueError("O conjunto de validação está vazio.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("As matrizes de embeddings contêm valores não finitos.")

    return X_train, y_train, X_val, y_val


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


def criar_mlp(*, early_stopping: bool = True) -> MLPClassifier:
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


def salvar_curva_aprendizado(X_train: np.ndarray, y_train: np.ndarray) -> Path | None:
    """Gera a curva F1-Macro usando validação cruzada sem SMOTE entre folds."""
    if len(y_train) < 2:
        LOGGER.warning("Curva de aprendizado não gerada: treino insuficiente para CV.")
        return None

    _, contagens = np.unique(y_train, return_counts=True)
    menor_classe = int(contagens.min())
    if menor_classe >= 2:
        divisores = min(5, menor_classe)
        cv = StratifiedKFold(n_splits=divisores, shuffle=True, random_state=RANDOM_STATE)
    else:
        divisores = min(5, len(y_train))
        cv = KFold(n_splits=divisores, shuffle=True, random_state=RANDOM_STATE)

    tamanhos, pontuacoes_treino, pontuacoes_validacao = learning_curve(
        estimator=criar_mlp(early_stopping=False),
        X=X_train,
        y=y_train,
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
    X_train, y_train, X_val, y_val = carregar_conjuntos()
    X_balanceado, y_balanceado = balancear_treino(X_train, y_train)

    modelo = criar_mlp()
    modelo.fit(X_balanceado, y_balanceado)
    previsoes = modelo.predict(X_val)

    print("Relatório de classificação (validação):")
    print(classification_report(y_val, previsoes, zero_division=0))

    matriz_path = salvar_matriz_confusao(y_val, previsoes)
    curva_path = salvar_curva_aprendizado(X_train, y_train)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)

    LOGGER.info("Matriz de confusão salva em %s", matriz_path)
    if curva_path is not None:
        LOGGER.info("Curva de aprendizado salva em %s", curva_path)
    LOGGER.info("Modelo salvo em %s", MODEL_PATH)
    LOGGER.info("F1-Macro na validação: %.4f", f1_score(y_val, previsoes, average="macro", zero_division=0))
    return modelo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    treinar_e_avaliar()
