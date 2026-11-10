# Dataset de perguntas e respostas com FAQs

Este conjunto contém conversas para desenvolver e avaliar respostas de QA com
recuperação de conteúdo da EJA. Cada registro é uma conversa completa com três
mensagens (`system`, `user` e `assistant`); o contexto e a pergunta ficam na
mensagem do usuário, e a resposta esperada fica na mensagem do assistente.

## Arquivos

| Arquivo | Exemplos | Uso pretendido |
|---|---:|---|
| `train.jsonl` | 640 | Conjunto de treino |
| `val.jsonl` | 136 | Conjunto de validação |
| `test.jsonl` | 132 | Conjunto de teste |

Cada linha é um objeto JSON com:

- `messages`: mensagens da conversa, identificadas por `role` e `content`.
- `meta`: metadados `id`, `tema`, `fonte` e `tipo`.
- `meta.tipo`: `positivo` indica resposta apoiada pelo contexto fornecido;
  `negativo` indica que o contexto não contém a informação e a resposta deve
  declarar que não a encontrou.

Os três conjuntos usam IDs distintos. O script
[`../../../scripts/ml/treino_faq_qa.py`](../../../scripts/ml/treino_faq_qa.py)
usa essas partições para ajustar e avaliar um modelo seq2seq: `train.jsonl` é
usado para o ajuste, `val.jsonl` para selecionar o checkpoint e `test.jsonl`
somente para a avaliação final. A entrada do modelo é a mensagem `system` junto
da mensagem `user` (contexto e pergunta); o alvo é a mensagem `assistant`.
`meta.tipo` é mantido para métricas separadas entre exemplos positivos e
negativos. O treinamento valida que os IDs de FAQ não cruzem partições.

A base de FAQs consultada pelo tutor fica separadamente em
[`../../faq/FAQ_CONSOLIDADO.md`](../../faq/FAQ_CONSOLIDADO.md).

Mantenha os conjuntos separados: use treino para ajuste, validação para decisões
durante o desenvolvimento e teste para a avaliação final. Não inclua dados
pessoais de estudantes.
