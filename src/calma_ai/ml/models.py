from enum import Enum

from pydantic import BaseModel, Field


class ProximaAcao(str, Enum):
    FAZER_PERGUNTA = "fazer_pergunta"
    DAR_DICA = "dar_dica"
    VALIDAR_E_AVANCAR = "validar_e_avancar"
    PROPOR_EXERCICIO = "propor_exercicio"


class Diagnostico(BaseModel):
    topico: str = Field(
        description="Chave do tópico na base de conteúdo (ex.: 'porcentagem') ou 'outro'."
    )
    conceitos_dominados: list[str] = Field(
        default_factory=list,
        description="Conceitos que o estudante já demonstrou dominar nesta interação.",
    )
    dificuldade_principal: str = Field(
        description="Principal obstáculo identificado no raciocínio, ou 'nenhuma'."
    )
    raciocinio_do_estudante_correto: bool | None = Field(
        default=None,
        description="True/False se o estudante apresentou um raciocínio; null se ainda não tentou.",
    )
    resposta_esperada: str = Field(
        description="Resultado correto verificado com a calculadora. Uso interno: nunca mostrar ao estudante."
    )
    proxima_acao: ProximaAcao = Field(
        description="Ação pedagógica escolhida para este turno."
    )
    justificativa: str = Field(description="Por que essa ação foi escolhida.")


class RespostaRevisada(BaseModel):
    mensagem: str = Field(description="Mensagem final, pronta para o estudante.")
    revelava_resposta: bool = Field(
        description="True se a versão do tutor entregava o resultado final e precisou ser corrigida."
    )
    ajustes: list[str] = Field(
        default_factory=list, description="Ajustes feitos pelo revisor, se houver."
    )
