import numpy as np
import pytest

from scripts.ml.prep_eja import dividir_e_salvar
from scripts.ml.treino_mlp_eja import carregar_conjuntos


def _dados_rotulados():
    embeddings = []
    categorias = []
    grupos = []
    for categoria in range(2):
        for apresentacao in range(12):
            grupo = f"categoria-{categoria}-aula-{apresentacao}"
            for slide in range(4):
                embeddings.append([categoria, apresentacao, slide, 1.0])
                categorias.append(categoria)
                grupos.append(grupo)
    return (
        np.asarray(embeddings, dtype=np.float32),
        np.asarray(categorias, dtype=np.int64),
        np.asarray(grupos),
    )


def test_splits_agrupados_nao_compartilham_apresentacoes(tmp_path):
    embeddings, categorias, grupos = _dados_rotulados()

    dividir_e_salvar(embeddings, categorias, grupos, tmp_path)
    conjuntos = carregar_conjuntos(tmp_path)
    groups_train, groups_val, groups_test = conjuntos[2], conjuntos[5], conjuntos[8]

    assert set(groups_train).isdisjoint(groups_val)
    assert set(groups_train).isdisjoint(groups_test)
    assert set(groups_val).isdisjoint(groups_test)
    assert len(conjuntos[0]) + len(conjuntos[3]) + len(conjuntos[6]) == len(embeddings)


def test_loader_rejeita_apresentacao_compartilhada_entre_splits(tmp_path):
    embeddings, categorias, grupos = _dados_rotulados()
    dividir_e_salvar(embeddings, categorias, grupos, tmp_path)
    groups_train = np.load(tmp_path / "groups_train_eja.npy", allow_pickle=False)
    groups_val = np.load(tmp_path / "groups_val_eja.npy", allow_pickle=False)
    groups_val[0] = groups_train[0]
    np.save(tmp_path / "groups_val_eja.npy", groups_val)

    with pytest.raises(ValueError, match="apresentações compartilhadas"):
        carregar_conjuntos(tmp_path)


def test_split_agrupado_exige_representacao_de_categorias_em_tres_grupos():
    embeddings = np.ones((6, 3), dtype=np.float32)
    categorias = np.array([0, 0, 0, 0, 1, 1])
    grupos = np.array(["a", "a", "b", "b", "c", "c"])

    with pytest.raises(ValueError, match="três apresentações distintas"):
        dividir_e_salvar(embeddings, categorias, grupos)
