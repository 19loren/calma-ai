# Dataset do Agente Diagnosticador da EJA

Pipeline experimental para transformar problemas do GSM8K em exemplos de
diagnóstico pedagógico em português brasileiro. O agente diagnosticador trabalha
nos bastidores: os textos `chosen` são orientações para um agente tutor, não
respostas diretas ao estudante. Nenhum dado pessoal real deve ser incluído.

## Ordem de execução

Abra os notebooks em Jupyter/VS Code e execute todas as células, em ordem:

1. `01_carregar_e_extrair_gsm8k.ipynb` baixa `openai/gsm8k` (`main`) e grava os
   problemas, as etapas, as contas `<<expressao=resultado>>` e as respostas
   `####` em `data/raw/gsm8k_extraido.jsonl`.
2. `02_localizar_problemas_eja.ipynb` adapta uma amostra para situações
   brasileiras e verifica as contas; grava `data/interim/problemas_localizados.jsonl`.
3. `03_injetar_erros_e_validar.ipynb` constrói tentativas corretas e incorretas
   controladas e grava `data/interim/erros_validados.jsonl`.
4. `04_gerar_diagnosticos_e_dpo.ipynb` cria diagnósticos tipados, rubricas e
   pares DPO em `data/interim/diagnosticos_dpo.jsonl`.
5. `05_split_e_exportar_revisao.ipynb` separa por problema-base e exporta
   `train.jsonl`, `val.jsonl`, `test.jsonl` e `amostra_revisao.csv` em
   `data/final/`.

Os notebooks usam caminhos relativos ao próprio arquivo e leem os resultados
anteriores do disco; reiniciar o kernel entre notebooks é seguro. O arquivo
`data/config.json` é a configuração compartilhada. Ajuste `max_examples`,
`seed`, `review_sample_size` e `error_proportions` nele. A proporção de erros é
normalizada para somar 1; usar limite baixo facilita validar o fluxo antes de
processar todo o corpus.

## Dependências e LLM

Requer Python 3.10+ e `datasets`, `pandas`, `sympy`, `pydantic>=2.7`,
`openai` e `python-dotenv`.
Cada notebook importa as dependências no começo; a célula inicial inclui um
comando `%pip install` opcional e comentado. O GSM8K é baixado pela biblioteca
Hugging Face `datasets`; a primeira execução precisa de internet e pode exigir
aceitar os termos do dataset na Hugging Face.

Se as dependências ainda não estiverem no ambiente do kernel:

```python
%pip install datasets pandas sympy "pydantic>=2.7" openai python-dotenv
```

O modo padrão é `MOCK`: não usa rede nem chave e permite testar localização e
geração de registros com exemplos sintéticos. Para adaptar com Groq, configure
`GROQ_API_KEY` no `.env` da raiz do projeto ou no ambiente, altere
`data/config.json` para `"llm_mode": "REAL"` e mantenha
`llm_base_url` como `https://api.groq.com/openai/v1`. O modelo padrão
`qwen/qwen3.8-27b` foi escolhido entre os modelos disponíveis na conta usada no
piloto; troque-o se sua conta expuser outros modelos adequados. A chamada usa o cliente OpenAI contra a API compatível da Groq e está
isolada em `chamar_llm()`, no notebook 02. A chave nunca deve ser gravada no
notebook nem exibida. Consulte a [compatibilidade OpenAI da Groq](https://console.groq.com/docs/openai)
e a [lista de modelos disponíveis](https://console.groq.com/docs/models).
O modo REAL não calcula: o LLM deve preservar os números; mudanças numéricas
fazem o exemplo ser descartado, e a resposta de referência é validada localmente.

## Formato e exemplos ilustrativos

Os registros são validados com Pydantic antes de salvar. Este trio ilustra o
formato e as decisões esperadas (não é saída obtida de uma chamada real ao LLM):

```json
{"id":"exemplo-operacao","base_id":"gsm8k-exemplo-1","erro_tipo":"ERRO_OPERACAO","entrada":{"duvida":"Somei o valor da compra ao troco.","contexto_cotidiano":"Conferir o troco de uma compra na feira.","historico_resumido":"Já identifica preços, mas confunde a operação do troco."},"saida":{"dificuldade":"Escolha da operação para calcular o troco.","verificacao_raciocinio":"O troco é encontrado subtraindo o valor pago pelo preço; a tentativa usou adição.","acao":"DAR_DICA","resposta_esperada":"O troco é a diferença entre o valor pago e o preço. Pergunte qual subtração representa a situação, sem calcular o resultado.","justificativa_acao":"Houve tentativa com operação inadequada; uma dica gradual apoia a revisão."},"rubrica":{"usou_contexto_cotidiano":true,"evitou_infantilizar":true,"linguagem_simples":true,"nao_vazou_resposta":true},"dpo":{"chosen":"Na feira, pense na diferença entre o que você pagou e o preço da compra. Qual operação mostra essa diferença?","rejected":"A compra custa R$ 18 e você pagou R$ 20. O troco é 20 - 18 = R$ 2."}}
{"id":"exemplo-calculo","base_id":"gsm8k-exemplo-2","erro_tipo":"ERRO_CALCULO","entrada":{"duvida":"Fiz 3 vezes 4 e encontrei 10.","contexto_cotidiano":"Separar itens em embalagens para o trabalho.","historico_resumido":"Escolhe a multiplicação corretamente, com erros ocasionais no produto."},"saida":{"dificuldade":"Erro de cálculo em uma multiplicação.","verificacao_raciocinio":"A operação escolhida corresponde a grupos iguais; o produto informado não confere.","acao":"DAR_DICA","resposta_esperada":"A operação faz sentido. Sugira conferir o produto com grupos ou uma conta auxiliar, sem informar o resultado.","justificativa_acao":"A estrutura está correta; feedback específico pode ajudar a revisar o cálculo."},"rubrica":{"usou_contexto_cotidiano":true,"evitou_infantilizar":true,"linguagem_simples":true,"nao_vazou_resposta":true},"dpo":{"chosen":"Você escolheu a multiplicação para representar grupos iguais. Como poderia conferir esse produto somando os grupos?","rejected":"3 vezes 4 é 12, então o total é 12 itens."}}
{"id":"exemplo-omissao","base_id":"gsm8k-exemplo-3","erro_tipo":"PASSO_OMITIDO","entrada":{"duvida":"Calculei o total das horas, mas não expliquei como achei o pagamento.","contexto_cotidiano":"Conferir o pagamento de horas extras.","historico_resumido":"Consegue somar períodos, mas às vezes não relaciona o total à taxa por hora."},"saida":{"dificuldade":"Falta explicitar a relação entre as horas e o valor por hora.","verificacao_raciocinio":"O total de horas foi identificado; falta o passo que combina esse total com a taxa.","acao":"VALIDAR_PARCIALMENTE","resposta_esperada":"Reconheça a soma correta das horas e pergunte como a taxa por hora entra no cálculo, sem calcular o pagamento.","justificativa_acao":"Há uma parte correta e um passo ausente; validar o avanço e solicitar continuação preserva a autonomia."},"rubrica":{"usou_contexto_cotidiano":true,"evitou_infantilizar":true,"linguagem_simples":true,"nao_vazou_resposta":true},"dpo":{"chosen":"Você já juntou as horas trabalhadas. Como o valor pago por cada hora poderia entrar no próximo passo?","rejected":"Com 5 horas a R$ 12 por hora, o pagamento é 5 x 12 = R$ 60."}}
```

## Revisão pedagógica e limitações

O CSV é uma amostra editável para professores, com `aprovado/reprovado` e
`comentário` em branco. Antes de qualquer uso, revisar: naturalidade da
localização; adequação à EJA regional; clareza e nível de leitura; consistência
entre erro, diagnóstico e ação; segurança matemática e unidades; se `chosen`
evita revelar a solução e se `rejected` de fato a revela. Os dados sintetizados
não substituem respostas reais de estudantes nem avaliação de professores.

Riscos/limitações: localização `MOCK` usa templates e não é tradução semântica
de cada enunciado; adaptação por LLM pode alterar detalhes, números ou contexto;
GSM8K foi criado em inglês e não representa o currículo nem a diversidade da
EJA brasileira; erros sintéticos não cobrem todas as estratégias de raciocínio;
regras de ação são heurísticas; a validação de contas cobre apenas expressões
aritméticas simples e seguras. Questões com unidades, condições implícitas,
frações complexas ou solução não reconhecida são descartadas ou exigem revisão.
O agrupamento por `base_id` evita vazamento direto, mas paráfrases semelhantes
entre problemas diferentes ainda podem atravessar splits. Verifique licença,
termos de uso e atribuição do GSM8K antes de redistribuir os dados derivados.
Não foram treinados modelos nem gerado conteúdo com informações pessoais.
