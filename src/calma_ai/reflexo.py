"""Agente reflexivo simples (Russell & Norvig).

Decide apenas com base na percepção atual (a mensagem do estudante), usando
regras condição-ação em ordem de prioridade. Não usa LLM nem memória, então
cada decisão pode ser explicada pela regra que disparou.
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from calma_ai.conhecimento import buscar_topico


class AcaoReflexa(str, Enum):
    ENCAMINHAR_APOIO = "encaminhar_apoio"
    PEDIR_DUVIDA = "pedir_duvida"
    CUMPRIMENTAR = "cumprimentar"
    ACOLHER = "acolher"
    RECUSAR_RESPOSTA_PRONTA = "recusar_resposta_pronta"
    ENCAMINHAR_EQUIPE = "encaminhar_equipe"


# Os padrões são aplicados ao texto normalizado: minúsculo e sem acentos.
PADRAO_CRISE = re.compile(
    r"quero morrer|me matar|tirar (a )?minha (propria )?vida|"
    r"nao quero mais viver|acabar com (a minha vida|tudo)|suicid"
)
PADRAO_DESANIMO = re.compile(
    r"\bnao consigo\b|\bnao sou capaz\b|\bsou burr[oa]\b|\bdesist|\bnao sirvo\b|"
    r"\bnao aprendo\b|\bvergonha\b|\bansios[oa]\b|\bnervos[oa]\b|"
    r"\bnunca vou (aprender|conseguir)\b|\bmuito dificil\b|\bburrice\b"
)
PADRAO_SAUDACAO = re.compile(
    r"^(oi+|ola|bom dia|boa tarde|boa noite|e ai|opa|salve)"
    r"([\s,!.?]+(tudo bem|tudo bom|professor[a]?|tutor[a]?))?[\s,!.?]*$"
)
PADRAO_PEDE_RESPOSTA = re.compile(
    r"\bme (da|de|diga|diz|fala|passa|mostra) (logo )?(a |o )?(resposta|resultado|conta)\b|"
    r"\bqual (e |eh )?(a resposta|o resultado)\b|"
    r"\b(resolve|faz|faca|resolva) (isso |essa conta |o exercicio )?(pra|para) mim\b"
)
PADRAO_NEGA_RESPOSTA = re.compile(r"\bnao (me )?(da|de|diga|diz|fala|passa|mostra|quero)\b")
PADRAO_TENTATIVA = re.compile(
    r"=|\bdeu\b|\bacho que\b|\bfiz\b|\bcalculei\b|\bminha resposta\b|"
    r"\bmultipliquei\b|\bdividi\b|\bsomei\b|\bsubtrai\b|\btentei\b"
)
PADRAO_DUVIDA = re.compile(
    r"\?|\b(como|quanto|quantos|quantas|qual|quais|por ?que|o que e|explica|entend|"
    r"calcul|conta|resolv|exercicio|questao|problema|ajuda|duvida|aprend|estud|"
    r"porcentagem|fracao|juros|regra de tres|area|perimetro|divis|multiplic|soma|subtra)"
)
PADRAO_NUMERO = re.compile(r"\d")
PADRAO_CONTEUDO = re.compile(r"[a-z0-9]")

RESPOSTAS = {
    AcaoReflexa.ENCAMINHAR_APOIO: (
        "Sinto muito que você esteja passando por isso. Você não está sozinho(a). "
        "Por favor, converse agora com alguém de confiança ou ligue para o CVV no 188 "
        "(gratuito, 24 horas) ou acesse cvv.org.br. Em caso de risco imediato, ligue 192 (SAMU). "
        "Você aceita procurar uma dessas ajudas agora?"
    ),
    AcaoReflexa.PEDIR_DUVIDA: (
        "Não identifiquei uma dúvida na mensagem. Pode me contar qual conteúdo ou exercício "
        "você quer trabalhar? Por exemplo: 'Como calculo 10% de 180?'"
    ),
    AcaoReflexa.CUMPRIMENTAR: (
        "Olá! Que bom ter você aqui. Qual conteúdo ou exercício você quer trabalhar hoje?"
    ),
    AcaoReflexa.ACOLHER: (
        "Tudo bem sentir isso: aprender depois de adulto, junto com trabalho e casa, exige "
        "muito, e travar faz parte do processo. Vamos um passo de cada vez."
    ),
    AcaoReflexa.RECUSAR_RESPOSTA_PRONTA: (
        "Eu não vou te entregar o resultado pronto, porque você aprende mais chegando nele, "
        "mas vou te acompanhar em cada passo."
    ),
}


@dataclass(frozen=True)
class Percepcao:
    texto_normalizado: str
    vazia: bool
    sinal_de_crise: bool
    sinal_de_desanimo: bool
    apenas_saudacao: bool
    pede_resposta_pronta: bool
    tem_numeros: bool
    mostra_tentativa: bool
    parece_duvida: bool


@dataclass(frozen=True)
class DecisaoReflexa:
    regra: str
    condicao: str
    acao: AcaoReflexa
    encaminhar_para_equipe: bool
    resposta: str | None
    pergunta_guia: str | None
    percepcao: Percepcao

    def resumo(self) -> str:
        p = self.percepcao
        return (
            f"regra {self.regra} ({self.condicao}) -> {self.acao.value}; "
            f"parece_duvida={p.parece_duvida}, tem_numeros={p.tem_numeros}, "
            f"mostra_tentativa={p.mostra_tentativa}, "
            f"sinal_de_desanimo={p.sinal_de_desanimo}, pede_resposta_pronta={p.pede_resposta_pronta}"
        )


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    texto = re.sub(r"\s+", " ", sem_acento.lower()).strip()
    return re.sub(r"^[-*•]\s+", "", texto)


def escolher_pergunta_guia(p: Percepcao) -> str:
    """Pergunta orientadora fixa: regra simples sobre a percepção + base de conhecimento."""
    if p.mostra_tentativa:
        return (
            "Boa, você já tentou! Releia o problema: o que exatamente ele pergunta? "
            "A sua conta responde isso?"
        )
    encontrado = buscar_topico(p.texto_normalizado)
    if encontrado:
        perguntas = encontrado[1].get("perguntas_orientadoras", [])
        if perguntas:
            return perguntas[0]
    return "Vamos por partes: quais valores o problema te dá e o que ele quer que você descubra?"


def perceber(mensagem: str) -> Percepcao:
    texto = normalizar(mensagem)
    pede = bool(PADRAO_PEDE_RESPOSTA.search(texto)) and not PADRAO_NEGA_RESPOSTA.search(texto)
    return Percepcao(
        texto_normalizado=texto,
        vazia=not PADRAO_CONTEUDO.search(texto),
        sinal_de_crise=bool(PADRAO_CRISE.search(texto)),
        sinal_de_desanimo=bool(PADRAO_DESANIMO.search(texto)),
        apenas_saudacao=bool(PADRAO_SAUDACAO.match(texto)),
        pede_resposta_pronta=pede,
        tem_numeros=bool(PADRAO_NUMERO.search(texto)),
        mostra_tentativa=bool(PADRAO_TENTATIVA.search(texto)),
        parece_duvida=bool(PADRAO_DUVIDA.search(texto)),
    )


def decidir(mensagem: str) -> DecisaoReflexa:
    """Aplica as regras condição-ação em ordem; a primeira que casar vence."""
    p = perceber(mensagem)

    if p.sinal_de_crise:
        regra, condicao, acao, encaminhar = "R1", "sinal de crise emocional", AcaoReflexa.ENCAMINHAR_APOIO, False
    elif p.vazia:
        regra, condicao, acao, encaminhar = "R2", "mensagem vazia", AcaoReflexa.PEDIR_DUVIDA, False
    elif p.apenas_saudacao:
        regra, condicao, acao, encaminhar = "R3", "apenas saudação", AcaoReflexa.CUMPRIMENTAR, False
    elif p.sinal_de_desanimo:
        regra, condicao, acao, encaminhar = "R4", "sinal de desânimo ou ansiedade", AcaoReflexa.ACOLHER, True
    elif p.pede_resposta_pronta:
        regra, condicao, acao, encaminhar = "R5", "pede a resposta pronta", AcaoReflexa.RECUSAR_RESPOSTA_PRONTA, True
    elif p.parece_duvida or p.tem_numeros or p.mostra_tentativa:
        regra, condicao, acao, encaminhar = "R6", "dúvida de conteúdo", AcaoReflexa.ENCAMINHAR_EQUIPE, True
    else:
        regra, condicao, acao, encaminhar = "R7", "nenhuma dúvida identificada", AcaoReflexa.PEDIR_DUVIDA, False

    return DecisaoReflexa(
        regra=regra,
        condicao=condicao,
        acao=acao,
        encaminhar_para_equipe=encaminhar,
        resposta=RESPOSTAS.get(acao),
        pergunta_guia=escolher_pergunta_guia(p) if encaminhar else None,
        percepcao=p,
    )


def run() -> None:
    """Modo demonstração: só o agente reflexivo, sem LLM e sem chave de API."""
    print("Agente reflexivo Calma.AI (digite 'sair' para encerrar)\n")
    try:
        nome = input("Qual é o seu nome? ").strip() or "Estudante"
    except EOFError:
        return
    print(f"Olá, {nome}! Escreva sua dúvida ou sua tentativa de resolver o exercício.\n")
    while True:
        try:
            mensagem = input(f"{nome}: ")
        except EOFError:
            break
        if normalizar(mensagem) == "sair":
            break
        decisao = decidir(mensagem)
        print(f"[reflexo] {decisao.resumo()}")
        resposta = " ".join(t for t in (decisao.resposta, decisao.pergunta_guia) if t)
        if resposta:
            print(f"Tutor: {resposta}")
        if decisao.encaminhar_para_equipe:
            print("[reflexo] no fluxo completo (uv run crewai run) a equipe de agentes daria uma orientação personalizada.")
        print()


if __name__ == "__main__":
    run()
