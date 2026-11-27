from __future__ import annotations

import json
import pytest

from calma_ai.knowledge.faqs import parse_faqs
from calma_ai.tools.progresso import (
    carregar_perfil,
    listar_trilha_materia,
    submeter_exercicios_aula,
)


def test_parse_faqs_aceita_metadados_de_trilha():
    md = """# Documento 1: Teste

## O que é porcentagem?
É uma razão de denominador 100.

**ID:** TEST-01
**Materia:** matematica
**Modulo:** 1 - Porcentagem
**Aula:** 1 - Conceito inicial
**Aula ID:** MAT-01
**Tema:** Porcentagem
**Fonte:** Apostila EJA
**PDF:** mat.pdf
**Páginas do PDF / slides:** p. 1
**Origem:** [Portal](https://exemplo.com)
"""
    faqs = parse_faqs(md)
    assert len(faqs) == 1
    assert faqs[0].materia == "matematica"
    assert faqs[0].aula_id == "MAT-01"


def test_desbloqueio_sequencial_e_bloqueio_de_aula_futura(tmp_path, monkeypatch):
    lotes_dir = tmp_path / "lotes" / "01_mat"
    lotes_dir.mkdir(parents=True)
    (lotes_dir / "exercicios.json").write_text(
        json.dumps(
            {
                "materia": "matematica",
                "aulas": [
                    {
                        "aula_id": "AULA-1",
                        "materia": "matematica",
                        "modulo_ordem": 1,
                        "aula_ordem": 1,
                        "titulo": "Aula 1",
                        "nota_minima_aprovacao": 0.7,
                        "exercicios": [
                            {"exercicio_id": "EX-1", "enunciado": "1+1?", "alternativas": ["1", "2"], "correta_index": 1}
                        ],
                    },
                    {
                        "aula_id": "AULA-2",
                        "materia": "matematica",
                        "modulo_ordem": 1,
                        "aula_ordem": 2,
                        "titulo": "Aula 2",
                        "nota_minima_aprovacao": 0.7,
                        "exercicios": [
                            {"exercicio_id": "EX-2", "enunciado": "2+2?", "alternativas": ["3", "4"], "correta_index": 1}
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("CALMA_LOTES_DIR", str(tmp_path / "lotes"))
    monkeypatch.setenv("CALMA_DATA_DIR", str(tmp_path / "estudantes"))

    trilha_inicial = listar_trilha_materia("matematica", "aluno_teste")
    assert [a["status"] for a in trilha_inicial] == ["disponivel", "bloqueada"]

    with pytest.raises(PermissionError):
        submeter_exercicios_aula("aluno_teste", "AULA-2", [1])

    reprovacao = submeter_exercicios_aula("aluno_teste", "AULA-1", [0])
    assert reprovacao["aprovado"] is False
    assert [a["status"] for a in listar_trilha_materia("matematica", "aluno_teste")] == ["disponivel", "bloqueada"]

    aprovacao = submeter_exercicios_aula("aluno_teste", "AULA-1", [1])
    assert aprovacao["aprovado"] is True
    assert aprovacao["proxima_aula_id"] == "AULA-2"
    assert [a["status"] for a in listar_trilha_materia("matematica", "aluno_teste")] == ["concluida", "disponivel"]

    perfil = carregar_perfil("aluno_teste")
    assert "AULA-1" in perfil["aulas_concluidas"]