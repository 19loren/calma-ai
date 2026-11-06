import pytest

from calma_ai import reflexo
from calma_ai.reflexo import AcaoReflexa, decidir, perceber


@pytest.mark.parametrize(
    ("mensagem", "regra", "acao", "encaminha"),
    [
        ("às vezes penso que quero morrer", "R1", AcaoReflexa.ENCAMINHAR_APOIO, False),
        ("   ...  ", "R2", AcaoReflexa.PEDIR_DUVIDA, False),
        ("", "R2", AcaoReflexa.PEDIR_DUVIDA, False),
        ("Oi, tudo bem?", "R3", AcaoReflexa.CUMPRIMENTAR, False),
        ("Boa noite!", "R3", AcaoReflexa.CUMPRIMENTAR, False),
        ("Não consigo entender porcentagem, sou burro", "R4", AcaoReflexa.ACOLHER, True),
        ("Me dá a resposta logo: quanto é 10% de 180?", "R5", AcaoReflexa.RECUSAR_RESPOSTA_PRONTA, True),
        ("Qual é o resultado de 3/4 de 200?", "R5", AcaoReflexa.RECUSAR_RESPOSTA_PRONTA, True),
        ("Como calculo 10% de 180?", "R6", AcaoReflexa.ENCAMINHAR_EQUIPE, True),
        ("não entendi juros", "R6", AcaoReflexa.ENCAMINHAR_EQUIPE, True),
        ("eu fiz 180 + 10", "R6", AcaoReflexa.ENCAMINHAR_EQUIPE, True),
        ("Max", "R7", AcaoReflexa.PEDIR_DUVIDA, False),
        ("beleza, valeu", "R7", AcaoReflexa.PEDIR_DUVIDA, False),
    ],
)
def test_regras_condicao_acao(mensagem, regra, acao, encaminha):
    decisao = decidir(mensagem)
    assert decisao.regra == regra
    assert decisao.acao == acao
    assert decisao.encaminhar_para_equipe is encaminha


def test_crise_tem_prioridade_sobre_desanimo():
    decisao = decidir("Não consigo mais, quero morrer")
    assert decisao.regra == "R1"
    assert "188" in decisao.resposta


def test_negacao_nao_conta_como_pedido_de_resposta():
    mensagem = "Minha conta aumentou 10%. Como descubro o aumento? Não me diga o resultado final."
    percepcao = perceber(mensagem)
    assert not percepcao.pede_resposta_pronta
    assert decidir(mensagem).regra == "R6"


def test_percebe_tentativa_e_numeros():
    percepcao = perceber("Eu fiz 180 + 10 = 190, acho que está certo")
    assert percepcao.tem_numeros
    assert percepcao.mostra_tentativa


def test_saudacao_com_duvida_nao_e_apenas_saudacao():
    assert decidir("Oi, como calculo juros simples?").regra == "R6"


def test_resumo_explica_a_decisao():
    resumo = decidir("Como calculo 10% de 180?").resumo()
    assert "R6" in resumo
    assert "encaminhar_equipe" in resumo


def test_r6_responde_com_pergunta_da_base_de_conhecimento():
    decisao = decidir("Minha conta de luz era R$ 180 e aumentou 10%. Como descubro o aumento?")
    assert decisao.pergunta_guia == (
        "Vamos por partes: o que significa a porcentagem indicada no problema? "
        "Como você poderia representá-la como partes iguais de um todo?"
    )


def test_reflexo_recebe_contexto_e_usa_assunto_para_pergunta_guia():
    decisao = decidir(
        "Como descubro o aumento?",
        assunto_curricular="porcentagem",
        contexto_cotidiano="conta de luz",
    )
    assert "porcentagem indicada no problema" in decisao.pergunta_guia
    assert decisao.percepcao.assunto_curricular == "porcentagem"
    assert decisao.percepcao.contexto_cotidiano == "conta de luz"
    assert "assunto=porcentagem" in decisao.resumo()
    assert "contexto=conta de luz" in decisao.resumo()


def test_reflexo_prioriza_topico_da_mensagem_sobre_assunto_padrao():
    decisao = decidir(
        "Como calcular uma fração de pizza?",
        assunto_curricular="porcentagem",
    )
    assert "partes iguais" in decisao.pergunta_guia


def test_r6_com_tentativa_pede_para_revisar_a_conta():
    assert "A sua conta responde isso?" in decidir("eu somei 180 + 10 e deu 190").pergunta_guia


def test_r6_sem_topico_usa_pergunta_generica():
    assert "parte está mais difícil" in decidir(
        "Como calculo a média das notas?"
    ).pergunta_guia


def test_reflexo_orienta_interpretacao_de_texto():
    decisao = decidir(
        "Como interpreto bem um texto?",
        assunto_curricular="Interpretação de texto",
        contexto_cotidiano="Estou muito cansado",
    )
    assert "o que costuma ser mais difícil" in decisao.pergunta_guia
    assert decisao.pergunta_guia.startswith("Como você disse que está cansado")


def test_reflexo_ajuda_a_comecar_quando_estudante_nao_sabe():
    decisao = decidir(
        "Não sei, como eu poderia fazer?",
        assunto_curricular="Interpretação de texto",
    )
    assert "Comece pelo título e pelo primeiro parágrafo" in decisao.pergunta_guia
    assert "Sobre quem ou o que" in decisao.pergunta_guia


def test_regras_que_param_nao_tem_pergunta_guia():
    assert decidir("Max").pergunta_guia is None


def test_ignora_marcador_de_lista_colado_do_arquivo_de_exemplos():
    assert decidir("- Oi").regra == "R3"


def test_modo_reflexo_apresenta_apenas_a_conversa(monkeypatch, capsys):
    respostas = iter(
        [
            "Max",
            "Interpretação de texto",
            "Estou cansado",
            "Como interpreto bem um texto?",
            "sair",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(respostas))
    monkeypatch.delenv("CALMA_DEBUG", raising=False)

    reflexo.run()

    texto = capsys.readouterr().out
    assert "Tutor:\nComo você disse que está cansado" in texto
    assert "[reflexo]" not in texto
    assert "a equipe de agentes daria" not in texto
