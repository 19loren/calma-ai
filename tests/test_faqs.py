import json

import pytest

from calma_ai.knowledge.faqs import parse_faqs, search_faqs
from calma_ai.tools.conteudo import ConsultarConteudoTool

FAQ_FIXTURE = """\
# FAQs CMSP - Base para RAG

# Documento 02: Como pensar diferentes realidades

## O que é socialização?

Socialização é o processo de aprendizagem de normas, regras e costumes que permite a convivência em sociedade. Começa nas relações familiares e continua na escola e em outros grupos.

**ID:** cmsp-02-002
**Tema:** Como pensar diferentes realidades
**Fonte:** 17.08_20H00_AUD1_ID1960_1T_EJA_EM_Como_pensar_diferentes_realidades.pptx
**PDF:** 02_como_pensar_diferentes_realidades.pdf
**Páginas do PDF / slides:** 8
**Origem:** [Google Drive](https://docs.google.com/presentation/d/1BSEZGTpQfoaSSF7UiWJUGaYuZoFcWK2x/edit?usp=drivesdk&ouid=107159505966451091098&rtpof=true&sd=true)

## O que significa desnaturalizar uma cultura?

Significa reconhecer que práticas culturais são construções humanas, em vez de tratá-las como naturais e inevitáveis. Essa compreensão favorece o respeito a hábitos de outras culturas.

**ID:** cmsp-02-003
**Tema:** Como pensar diferentes realidades
**Fonte:** 17.08_20H00_AUD1_ID1960_1T_EJA_EM_Como_pensar_diferentes_realidades.pptx
**PDF:** 02_como_pensar_diferentes_realidades.pdf
**Páginas do PDF / slides:** 8
**Origem:** [Google Drive](https://docs.google.com/presentation/d/1BSEZGTpQfoaSSF7UiWJUGaYuZoFcWK2x/edit?usp=drivesdk&ouid=107159505966451091098&rtpof=true&sd=true)

# Documento 26: Desnaturalização

## Como a socialização influencia nossos costumes?

Por meio da família, da escola e de outros grupos, aprendemos e internalizamos normas e concepções. A aula destaca que experiências e trajetórias diferentes produzem pessoas e perspectivas singulares.

**ID:** cmsp-26-005
**Tema:** Desnaturalização
**Fonte:** 01.07_20H00_AUD1_ID1058_CI_EJA_1T_Desnaturalização.pptx
**PDF:** 26_desnaturalizacao.pdf
**Páginas do PDF / slides:** 10
**Origem:** [Google Drive](https://docs.google.com/presentation/d/1nhn0qv12joYBKVHD-fOOHdRIS7tE8WAs/edit?usp=drivesdk&ouid=107159505966451091098&rtpof=true&sd=true)
"""


def test_parser_preserves_question_answer_and_source_metadata():
    faqs = parse_faqs(FAQ_FIXTURE)

    assert len(faqs) == 3
    assert faqs[0].id == "cmsp-02-002"
    assert faqs[0].question == "O que é socialização?"
    assert faqs[0].answer.startswith("Socialização é o processo")
    assert faqs[0].topic == "Como pensar diferentes realidades"
    assert faqs[0].pages == "8"
    assert faqs[0].origin.startswith("Google Drive (https://docs.google.com/")


def test_parser_rejects_missing_metadata_and_duplicate_ids():
    with pytest.raises(ValueError, match="metadados ausentes"):
        parse_faqs("# Documento 01: Teste\n\n## Pergunta\n\nResposta.")

    duplicated = FAQ_FIXTURE.replace("cmsp-02-003", "cmsp-02-002")
    with pytest.raises(ValueError, match="identificadores duplicados"):
        parse_faqs(duplicated)


def test_retrieval_ranks_paraphrase_and_does_not_split_answer(tmp_path):
    corpus = tmp_path / "faqs.md"
    corpus.write_text(FAQ_FIXTURE, encoding="utf-8")

    result = search_faqs(
        "Como aprendemos regras e costumes para viver em sociedade?",
        path=corpus,
    )

    assert result.matches[0].faq.id == "cmsp-02-002"
    assert "relações familiares" in result.matches[0].faq.answer
    assert result.matches[0].faq.topic == "Como pensar diferentes realidades"


def test_retrieval_abstains_without_shared_informative_terms(tmp_path):
    corpus = tmp_path / "faqs.md"
    corpus.write_text(FAQ_FIXTURE, encoding="utf-8")

    assert search_faqs("o que é?", path=corpus).matches == ()
    assert search_faqs("receita de bolo", path=corpus).matches == ()


def test_retrieval_marks_nearby_subjects_as_ambiguous(tmp_path):
    corpus = tmp_path / "faqs.md"
    corpus.write_text(FAQ_FIXTURE, encoding="utf-8")

    result = search_faqs("socialização e costumes", path=corpus)

    assert {match.faq.id for match in result.matches} == {"cmsp-02-002", "cmsp-26-005"}
    assert result.ambiguous


def test_single_term_matching_multiple_subjects_is_ambiguous(tmp_path):
    corpus = tmp_path / "faqs.md"
    corpus.write_text(FAQ_FIXTURE, encoding="utf-8")

    result = search_faqs("socialização", path=corpus)

    assert {match.faq.topic for match in result.matches} == {
        "Como pensar diferentes realidades",
        "Desnaturalização",
    }
    assert result.ambiguous


def test_conteudo_tool_returns_grounded_faq_and_ambiguity_status(tmp_path, monkeypatch):
    corpus = tmp_path / "faqs.md"
    corpus.write_text(FAQ_FIXTURE, encoding="utf-8")
    monkeypatch.setenv("CALMA_FAQ_PATH", str(corpus))

    response = json.loads(ConsultarConteudoTool()._run("aprendizagem de normas sociais"))

    assert response["status_faq"] in {"correspondencia", "ambigua"}
    assert response["referencias_faq"][0]["id"] == "cmsp-02-002"
    assert response["referencias_faq"][0]["tema"] == "Como pensar diferentes realidades"


def test_conteudo_tool_reports_missing_corpus_without_breaking_curriculum(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CALMA_FAQ_PATH", str(tmp_path / "inexistente.md"))

    response = json.loads(ConsultarConteudoTool()._run("porcentagem"))

    assert response["topico"] == "porcentagem"
    assert response["status_faq"] == "indisponivel"
