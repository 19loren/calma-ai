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

Os três conjuntos usam IDs distintos. O script `scripts/avaliar_faqs.py` usa os
três arquivos para avaliar o retriever neural da base de FAQs; ele calibra os
limiares somente em `val` e reserva `test` para a avaliação final. Para exemplos
com IDs presentes no corpus, `meta.id` identifica a FAQ esperada para recuperação,
independentemente do rótulo `meta.tipo`. Esse rótulo indica se a resposta está
apoiada pelo contexto fornecido no exemplo; não é um rótulo de relevância para a
busca na base completa. IDs ausentes do corpus são contabilizados como casos sem
alvo recuperável e usados para medir abstenção. A base de FAQs consultada pelo
tutor fica separadamente em
[`../../faq/FAQ_CONSOLIDADO.md`](../../faq/FAQ_CONSOLIDADO.md).

Mantenha os conjuntos separados ao criar um pipeline futuro: use treino para
ajuste, validação para decisões durante o desenvolvimento e teste para uma
avaliação final. Não inclua dados pessoais de estudantes.
