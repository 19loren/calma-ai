import json

from calma_ai.knowledge.exemplos_diagnosticador import (
    SEM_EXEMPLO,
    formatar_exemplos_diagnosticador,
)


def _registro(tentativa, orientacao, rejeitada):
    return {
        "erro_tipo": "ERRO_CALCULO",
        "entrada": {"duvida": tentativa},
        "saida": {"dificuldade": "Conferir o cálculo."},
        "dpo": {"chosen": orientacao, "rejected": rejeitada},
    }


def test_recupera_exemplo_sem_incluir_resposta_rejeitada(tmp_path):
    caminho = tmp_path / "train.jsonl"
    registros = [
        _registro(
            "Problema de conta de luz com porcentagem.",
            "Como você pode conferir a porcentagem?",
            "O resultado completo é 18.",
        ),
        _registro(
            "Problema sobre páginas lidas em um livro.",
            "Qual informação do livro deve entrar na conta?",
            "A resposta completa é 48.",
        ),
    ]
    caminho.write_text(
        "\n".join(json.dumps(registro, ensure_ascii=False) for registro in registros),
        encoding="utf-8",
    )

    resultado = formatar_exemplos_diagnosticador(
        "Minha conta de luz teve um aumento em porcentagem.",
        arquivo=caminho,
    )

    assert "Como você pode conferir a porcentagem?" in resultado
    assert "O resultado completo é 18." not in resultado
    assert "A resposta completa é 48." not in resultado


def test_nao_injeta_exemplo_sem_semelhanca_lexical(tmp_path):
    caminho = tmp_path / "train.jsonl"
    registro = _registro(
        "Problema sobre páginas lidas em um livro.",
        "Qual informação deve entrar na conta?",
        "A resposta completa é 48.",
    )
    caminho.write_text(json.dumps(registro), encoding="utf-8")

    resultado = formatar_exemplos_diagnosticador(
        "Explique astronomia e planetas.",
        arquivo=caminho,
    )

    assert resultado == SEM_EXEMPLO
