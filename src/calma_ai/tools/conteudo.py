import json
import logging

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from calma_ai.conhecimento import buscar_topico, carregar_conteudos
from calma_ai.knowledge.faqs import FAQSearch, search_faqs

LOGGER = logging.getLogger(__name__)


class ConsultarConteudoInput(BaseModel):
    consulta: str = Field(
        description="Nome do tópico (ex.: 'porcentagem', 'juros_simples') ou a própria dúvida do estudante."
    )


class ConsultarConteudoTool(BaseTool):
    name: str = "consultar_conteudo_eja"
    description: str = (
        "Consulta a base pedagógica da EJA. Retorna conceitos-chave, erros comuns, "
        "perguntas orientadoras, exercícios e FAQs recuperadas com tema e fontes. "
        "Use para diagnosticar dificuldades e identificar materiais. Se a busca FAQ "
        "for ambígua ou sem correspondência, não deduza o material."
    )
    args_schema: type[BaseModel] = ConsultarConteudoInput

    @staticmethod
    def _faq_payload(result: FAQSearch) -> dict[str, object]:
        return {
            "status_faq": "ambigua" if result.ambiguous else "correspondencia",
            "referencias_faq": [
                {
                    "id": match.faq.id,
                    "pergunta": match.faq.question,
                    "resposta": match.faq.answer,
                    "tema": match.faq.topic,
                    "documento": match.faq.document,
                    "fonte": match.faq.source,
                    "pdf": match.faq.pdf,
                    "paginas": match.faq.pages,
                    "origem": match.faq.origin,
                    "similaridade": round(match.score, 4),
                }
                for match in result.matches
            ],
        }

    def _run(self, consulta: str) -> str:
        encontrado = buscar_topico(consulta)
        try:
            faqs = search_faqs(consulta)
        except FileNotFoundError as erro:
            LOGGER.warning("Base FAQ indisponível: %s", erro)
            faqs = None

        if encontrado is None:
            if faqs and faqs.matches:
                resultado: dict[str, object] = {"topico": "outro"}
                resultado.update(self._faq_payload(faqs))
            else:
                disponiveis = ", ".join(carregar_conteudos())
                aviso = (
                    "Nenhum tópico encontrado. Use 'outro' se nenhum servir. "
                    f"Tópicos disponíveis: {disponiveis}."
                )
                if faqs is None:
                    aviso += (
                        " status_faq=indisponivel. Não infira matéria a partir da base FAQ."
                    )
                elif not faqs.matches:
                    aviso += (
                        " status_faq=sem_correspondencia. "
                        "A FAQ não contém uma correspondência para esta consulta."
                    )
                return aviso
        else:
            chave, dados = encontrado
            dados_sem_chaves = {
                key: value for key, value in dados.items() if key != "palavras_chave"
            }
            resultado = {"topico": chave, **dados_sem_chaves}
            if faqs and faqs.matches:
                resultado.update(self._faq_payload(faqs))
            elif faqs is None:
                resultado["status_faq"] = "indisponivel"
            else:
                resultado["status_faq"] = "sem_correspondencia"
        return json.dumps(resultado, ensure_ascii=False, indent=2)
