import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from calma_ai.conhecimento import buscar_topico, carregar_conteudos


class ConsultarConteudoInput(BaseModel):
    consulta: str = Field(
        description="Nome do tópico (ex.: 'porcentagem', 'juros_simples') ou a própria dúvida do estudante."
    )


class ConsultarConteudoTool(BaseTool):
    name: str = "consultar_conteudo_eja"
    description: str = (
        "Consulta a base pedagógica da EJA. Retorna conceitos-chave, erros comuns, "
        "perguntas orientadoras e exercícios contextualizados do tópico. "
        "Use para diagnosticar dificuldades e para escolher perguntas ou exercícios."
    )
    args_schema: type[BaseModel] = ConsultarConteudoInput

    def _run(self, consulta: str) -> str:
        encontrado = buscar_topico(consulta)
        if encontrado is None:
            disponiveis = ", ".join(carregar_conteudos())
            return (
                f"Nenhum tópico encontrado para '{consulta}'. "
                f"Tópicos disponíveis: {disponiveis}. Se nenhum servir, use 'outro'."
            )
        chave, dados = encontrado
        dados_sem_chaves = {k: v for k, v in dados.items() if k != "palavras_chave"}
        return json.dumps({"topico": chave, **dados_sem_chaves}, ensure_ascii=False, indent=2)
