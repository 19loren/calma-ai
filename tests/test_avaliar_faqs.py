from types import SimpleNamespace

from calma_ai.knowledge.faqs import FAQMatch, FAQSearch
from scripts.avaliar_faqs import Example, score_results


def test_metricas_tratam_tipo_negativo_com_faq_no_corpus_como_alvo_de_busca():
    exemplos = [
        Example(faq_id="faq-1", context_supported=False, query="pergunta"),
        Example(faq_id="guia-1", context_supported=True, query="fora da base"),
    ]
    resultados = (
        FAQSearch(
            matches=(FAQMatch(faq=SimpleNamespace(id="faq-1"), score=0.8),),
            ambiguous=False,
        ),
        FAQSearch(
            matches=(FAQMatch(faq=SimpleNamespace(id="faq-1"), score=0.7),),
            ambiguous=False,
        ),
    )

    metricas = score_results(
        exemplos,
        resultados,
        corpus_ids={"faq-1"},
        min_similarity=0.6,
        ambiguity_margin_ratio=0.0,
    )

    assert metricas["true_positives"] == 1
    assert metricas["false_positives"] == 1
    assert metricas["false_negatives"] == 0
    assert metricas["context_unsupported_hits"] == 1
    assert metricas["out_of_corpus_targets"] == 1


def test_metricas_contabilizam_abstencao_para_alvo_ausente():
    exemplo = Example(faq_id="guia-1", context_supported=True, query="fora da base")

    metricas = score_results(
        [exemplo],
        (FAQSearch(matches=(), ambiguous=False),),
        corpus_ids={"faq-1"},
        min_similarity=0.6,
        ambiguity_margin_ratio=0.0,
    )

    assert metricas["false_positives"] == 0
    assert metricas["out_of_corpus_targets"] == 1
