# Experimento isolado: suporte da resposta ao contexto

Este diretório contém um experimento exploratório separado do tutor. Ele usa os
splits `train.jsonl`, `val.jsonl` e `test.jsonl` de
`data/datasets/faq_qa/` para classificar se a resposta esperada está apoiada no
contexto (`positivo`/`negativo`).

O notebook compara uma MLP com TF-IDF a uma regressão logística de referência.
Ele não classifica emoções, não altera o Agente Diagnosticador e não salva nem
carrega modelos da aplicação.

Abra `classificador_suporte_contexto.ipynb` na raiz do repositório e execute as
células em ordem. O treino e a seleção de limiar usam `train` e `val`; o conjunto
`test` é usado somente na avaliação final.
