# Engenharia reversa: rede neural e RAG

## Escopo e conclusão principal

O repositório contém três mecanismos diferentes que podem ser confundidos:

1. **A rede neural de QA contextual** é um pipeline offline em
   `scripts/ml/treino_faq_qa.py`. Ela ajusta um modelo seq2seq em português usando
   os exemplos JSONL para gerar respostas a partir do system prompt, contexto e
   pergunta.
2. **A recuperação de FAQs usada pelo tutor** é lexical: vetores esparsos de
   palavras e bigramas, ponderados por TF-IDF e comparados por similaridade do
   cosseno. Não usa embeddings neurais nem um banco vetorial.
3. **A geração de respostas do tutor** é feita pelos agentes CrewAI com
   `ollama/llama3.2`. A recuperação fornece referências ao diagnosticador; não
   treina nem ajusta esse LLM.

O ajuste fino de QA é separado do LLM Ollama usado no runtime e da recuperação
lexical. Os JSONL não treinam a busca TF-IDF, nem substituem automaticamente os
agentes do tutor.

## Mapa dos fluxos

### Fluxo usado na conversa

```text
pergunta do estudante
  -> regras reflexas (regex e palavras-chave; sem LLM)
  -> se encaminhada:
       diagnosticador CrewAI
         -> perfil local do estudante
         -> conteúdo curricular + busca de FAQs
         -> calculadora
       -> tutor CrewAI
       -> revisor CrewAI + validação determinística
  -> resposta apresentada ao estudante
```

O agente reflexivo decide se encerra o turno com uma resposta fixa ou o envia à
equipe. Crise, mensagem vazia, saudação e mensagem sem dúvida identificada
podem terminar sem chamada ao LLM. Dúvidas, tentativas, desânimo ou pedidos de
resposta pronta são encaminhados, com a resposta reflexa disponível como
contexto para a equipe. Consulte `src/calma_ai/main.py` e
`src/calma_ai/reflexo.py`.

Na equipe, o diagnosticador recebe as ferramentas de perfil, conteúdo/FAQ e
calculadora; o tutor recebe a ferramenta de conteúdo/FAQ; o revisor não recebe
ferramentas. As tarefas executam em sequência. O resultado final é estruturado
e validado: exige mensagem não vazia, pergunta orientadora e limite de 1.000
caracteres, além de bloquear um padrão de conta resolvida. Consulte
`src/calma_ai/crew.py`, `src/calma_ai/models.py` e `src/calma_ai/config/`.

## Recuperação de FAQs (RAG)

### Base e preparação dos registros

- Fonte consultada em runtime: `data/faq/FAQ_CONSOLIDADO.md`.
- `src/calma_ai/knowledge/faqs.py` interpreta cada cabeçalho `##` como uma
  pergunta e o texto seguinte como a resposta. O registro também preserva tema,
  documento, fonte, PDF, páginas, origem e ID.
- Resposta vazia, metadados obrigatórios ausentes ou IDs duplicados tornam a
  base inválida.
- A base é carregada e indexada sob demanda; o índice em memória é armazenado
  em cache (até quatro caminhos/versões, identificadas por caminho, data de
  modificação e tamanho do arquivo).

### Algoritmo de busca

1. Normaliza texto para minúsculas e remove acentos.
2. Extrai tokens alfanuméricos; remove palavras curtas e uma lista fixa de
   stopwords em português.
3. Cria features de palavras (`w:...`) e bigramas adjacentes (`b:...`).
4. Constrói features por FAQ a partir de pergunta, resposta, tema e documento,
   com pesos respectivos `2.0`, `1.0`, `1.5` e `0.75`.
5. Calcula IDF por feature no conjunto de FAQs; aplica TF logarítmico e
   normalização L2. A consulta é vetorizada com o mesmo IDF.
6. Calcula produto escalar entre vetores normalizados (equivalente a cosseno),
   ordena por pontuação e devolve no máximo três resultados.
7. Filtra correspondências fracas: exige pelo menos duas palavras informativas
   compartilhadas; para consulta com uma única palavra, ela precisa aparecer na
   pergunta ou no tema da FAQ.
8. Marca busca como ambígua se a consulta de uma palavra alcançar mais de um
   tema, ou se a diferença entre as duas maiores pontuações não superar 25% da
   maior pontuação.

Portanto, trata-se de **recuperação lexical esparsa com TF-IDF**, não busca
semântica densa por embeddings, nem BM25. Parâmetros relevantes e a lógica estão
em `src/calma_ai/knowledge/faqs.py`.

### Como o resultado chega ao LLM

`ConsultarConteudoTool` combina duas buscas independentes:

- busca curricular por tópico exato ou contagem de palavras-chave em
  `src/calma_ai/knowledge/conteudos.json`;
- busca de FAQs por `search_faqs`.

Quando há referências, a ferramenta retorna pergunta, resposta, tema, fonte,
documento, páginas, origem e pontuação, além de `status_faq` (`correspondencia`
ou `ambigua`). Sem correspondência, informa isso explicitamente; se o arquivo
FAQ não existir, registra aviso e retorna `indisponivel`. A ausência da FAQ não
impede o retorno do tópico curricular. A ferramenta não gera a resposta final:
o diagnosticador usa os trechos para identificar o material e fornecer fatos
fundamentados ao tutor, conforme as instruções em `config/agents.yaml` e
`config/tasks.yaml`.

## Rede neural: geração de respostas com JSONL

`scripts/ml/treino_faq_qa.py` ajusta `unicamp-dl/ptt5-small-portuguese-vocab`,
um modelo encoder-decoder T5 pré-treinado em português com aproximadamente 60
milhões de parâmetros. BERT não é usado nesse pipeline. O pipeline:

1. Lê cada partição JSONL e monta a entrada com as mensagens `system` e `user`;
   o alvo de geração é a resposta `assistant`.
2. Usa `train.jsonl` para otimizar o modelo, com learning rate `5e-5`, até cinco
   épocas por padrão, batch 4 e weight decay `0.01`.
3. Avalia a perda em `val.jsonl` ao final de cada época e preserva o melhor
   checkpoint.
4. Após concluir o treinamento e selecionar o checkpoint, gera respostas para
   `test.jsonl` uma única vez. Calcula exact match normalizado e token F1,
   tanto globalmente quanto separados pelo rótulo `positivo`/`negativo`.
5. Salva modelo, tokenizer e `test_metrics.json` em `models/faq_qa_ptt5/`, pasta
   ignorada pelo Git.

Antes do treino, verifica que nenhum `meta.id` de FAQ apareça em mais de uma
partição, prevenindo vazamento entre treino, validação e teste. IDs podem ter
várias paráfrases dentro de uma mesma partição. Tamanho máximo de entrada e
resposta são 512 e 128 tokens por padrão; GPU CUDA é usada se disponível, senão
o treino ocorre em CPU e pode ser mais lento.

### O que este ajuste fino não faz

- Não treina o modelo do zero: adapta um T5 pré-treinado para a tarefa/contexto
  do dataset.
- Não substitui o `ollama/llama3.2` do runtime; não é integrado ao tutor até que
  uma etapa de inferência seja implementada separadamente.
- Não altera nem treina o recuperador lexical de FAQs.
- O conjunto JSONL tem contexto já fornecido a cada exemplo. As métricas avaliam
  geração condicionada a esse contexto; não medem se o RAG encontra o trecho
  correto.
- Exact match e token F1 são métricas lexicais e não substituem a revisão
  pedagógica/factual de uma amostra de respostas.

## Dataset JSONL de QA com FAQs

O dataset fica em `data/datasets/faq_qa/`; cada exemplo contém mensagens de
system/user/assistant e metadados `id`, `tema`, `fonte` e `tipo`. O contexto e a
pergunta ficam juntos na mensagem do usuário. `positivo` indica resposta
sustentada pelo contexto e `negativo` indica que o contexto não contém a
resposta, devendo o assistente se abster.

Contagem verificada diretamente nos arquivos:

| Partição | Exemplos | Positivos | Negativos | IDs distintos |
|---|---:|---:|---:|---:|
| Treino | 640 | 480 | 160 | 160 |
| Validação | 136 | 102 | 34 | 34 |
| Teste | 132 | 99 | 33 | 33 |

Cada ID aparece em quatro exemplos dentro da sua partição; os 160 IDs de treino,
34 de validação e 33 de teste não se sobrepõem. Isso não substitui uma avaliação
de recuperação: esses exemplos são diálogos de QA com contexto já fornecido,
não pares consulta/FAQ consumidos por `search_faqs`.

Os resultados concretos são produzidos localmente após o treinamento; não se
presume um desempenho até a avaliação no conjunto de teste.

## Caminhos para reproduzir

Instale as dependências opcionais e rode o ajuste fino:

```powershell
pip install -e ".[faq-qa]"
python scripts/ml/treino_faq_qa.py
```

O extra `faq-qa` instala PyTorch, Transformers, SentencePiece e Accelerate
(backend necessário pelo `Trainer`).

No runtime, o fluxo de FAQ ocorre quando um agente usa a ferramenta
`consultar_conteudo_eja`; não há um comando dedicado para pré-computar embeddings
ou índice vetorial. O provedor LLM configurado para a equipe é Ollama local com
`llama3.2`.

## Arquivos-fonte principais

- `scripts/ml/treino_faq_qa.py` — fine-tuning, seleção por validação e avaliação
  final no teste.
- `src/calma_ai/training/faq_qa.py` — leitura/validação do JSONL e métricas.
- `src/calma_ai/knowledge/faqs.py` — parser, indexação TF-IDF e ranking de FAQs.
- `src/calma_ai/tools/conteudo.py` — integração entre tópico curricular e FAQ.
- `src/calma_ai/knowledge/conteudos.py` — busca determinística de tópicos.
- `src/calma_ai/crew.py` e `src/calma_ai/config/` — fluxo de agentes e instruções.
- `data/datasets/faq_qa/README.md` — formato e uso pretendido do dataset JSONL.
