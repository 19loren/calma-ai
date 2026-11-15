# calma-ai

Tutor socrático para estudantes adultos da EJA. Ele não entrega respostas prontas:
ajuda o estudante a chegar nelas, um passo de cada vez.

## Arquitetura

```
mensagem do estudante
   │
   ├── contexto: conteúdo curricular e situação do dia a dia
   ▼
Agente reflexivo (regras if/else + regex, sem chamada ao LLM) src/calma_ai/reflexo.py
   │  R1 crise → CVV 188 (para aqui)
   │  R2 vazia / R3 saudação → resposta fixa (para aqui)
   │  R4 desânimo → acolhe e encaminha
   │  R5 pede resposta pronta → recusa e encaminha
   │  R6 dúvida de conteúdo (pergunta, número ou tentativa) → encaminha
   │  R7 nenhuma dúvida identificada (ex.: só um nome) → pede a dúvida (para aqui)
   ▼
Equipe CrewAI (Llama 3.2 via Ollama local)                  src/calma_ai/crew.py
   1. Diagnosticador: consulta histórico, base de conteúdo e calculadora;
      decide a próxima ação (fazer_pergunta, dar_dica, validar_e_avancar,
      propor_exercicio) e devolve um diagnóstico estruturado.
   2. Tutor: escreve a mensagem executando a ação decidida.
   3. Revisor: garante que a mensagem não revela a resposta final.
      Um guardrail em código exige que a mensagem tenha uma pergunta.
   ▼
Progresso do estudante salvo em data/estudantes/<id>.json
```

## Estrutura do projeto

```text
.
├── .crewai/               # Metadados de execução do CLI CrewAI
├── .env                   # Configuração local (não versionar)
├── .gitignore             # Arquivos locais e saídas que não entram no Git
├── src/calma_ai/          # Aplicação do tutor
│   ├── config/            # Instruções dos agentes e tarefas
│   ├── knowledge/         # Dados curriculares, consulta e recuperação de FAQs
│   └── tools/             # Calculadora, consulta de conteúdo e progresso
├── data/
│   ├── faq/               # Base de FAQs usada pelo RAG
│   └── datasets/
│       └── faq_qa/        # Exemplos JSONL para QA com recuperação de FAQs
├── scripts/ml/            # Pré-processamento e treinamento opcionais de slides
├── tests/                 # Testes automatizados
├── docs/                  # Exemplos e material de apoio
├── pyproject.toml         # Dependências, comandos e configuração do projeto
├── uv.lock                # Versões resolvidas das dependências
└── README.md              # Este guia
```

### O que é usado por cada parte

- **Tutor:** `src/calma_ai/`; o comando `uv run reflexo` usa regras determinísticas,
  enquanto `uv run crewai run` também usa os agentes configurados em `config/`.
- **Base de conhecimento:** `src/calma_ai/knowledge/conteudos.json` contém o conteúdo
  curricular, carregado por `src/calma_ai/knowledge/conteudos.py`;
  `src/calma_ai/knowledge/faqs.py` recupera FAQs de
  `data/faq/FAQ_CONSOLIDADO.md`.
- **Progresso:** os históricos dos estudantes são gravados em `data/estudantes/`.
- **Classificação de slides:** `scripts/ml/` é um fluxo opcional e separado do tutor.
  Ele lê apresentações `.pptx`, gera dados em `data/processed/` e salva gráficos em
  `resultados/`.
- **Dataset de QA:** `data/datasets/faq_qa/` contém conjuntos `train`, `val` e
  `test` de conversas com respostas apoiadas ou não pelas FAQs. O README dessa pasta
  explica o formato, os rótulos e as divisões. Nenhum pipeline atual do repositório
  carrega esses arquivos para treinar ou avaliar um modelo.
- **Exemplos para testar o tutor:** [`docs/exemplos_perguntas.md`](docs/exemplos_perguntas.md).

As bases em `data/faq/` e `data/datasets/` são versionadas. Já `data/raw/`,
`data/processed/`, `data/estudantes/`, `resultados/`, `state/` e o ambiente virtual
são pastas locais/de saída e ficam fora do Git. O diretório `.crewai/` contém
metadados da execução do CLI CrewAI; não faz parte do código do tutor.

No fluxo completo, acolhimentos e limites do agente reflexivo são incorporados à
orientação do tutor em uma única resposta. Perguntas-guia são sugestões: o tutor deve
adaptá-las à dúvida, à tentativa e ao contexto reais do estudante.

- Agentes e tarefas: `src/calma_ai/config/agents.yaml` e `tasks.yaml`.
- Ferramentas: `src/calma_ai/tools/` (calculadora segura, conteúdo e perfil do estudante).
- Base curricular: `src/calma_ai/knowledge/conteudos.json`.
- Recuperação de FAQs e metadados dos materiais: `src/calma_ai/knowledge/faqs.py`.

## Setup

Requer Python 3.10 ou 3.11 e [uv](https://docs.astral.sh/uv/). O fluxo completo também requer
o [Ollama](https://ollama.com/) instalado e em execução localmente.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install uv
uv sync
```

Baixe o modelo local usado pelos agentes:

```powershell
ollama pull llama3.2
```

O Ollama normalmente inicia o serviço local automaticamente. Se necessário, inicie-o
com `ollama serve`. O padrão é `http://localhost:11434`; para outro endereço, configure
`OLLAMA_HOST` conforme a convenção do Ollama. O fluxo reflexivo continua usando regras
determinísticas para priorizar crise e encaminhamento; o contexto, a ação reflexa e a
pergunta orientadora são enviados à equipe Llama junto com a pergunta do estudante.

## Uso

Somente o agente reflexivo (não usa LLM nem chave de API):

```powershell
uv run reflexo
```
Este modo usa regras fixas e perguntas-guia da base; ele não mantém histórico e não
gera respostas personalizadas com um modelo. Para uma conversa mais adaptativa, use
o fluxo completo abaixo. Para exibir informações técnicas de diagnóstico no terminal,
defina `CALMA_DEBUG=1` antes de iniciar.

Fluxo completo (agente reflexivo + equipe de agentes):

```powershell
uv run crewai run
```

O CLI pede o identificador do estudante, a dúvida (ou tentativa) e, opcionalmente,
o conteúdo curricular e o contexto do dia a dia.
Depois de cada resposta, a conversa continua; digite `sair` no campo da dúvida para
encerrar. O histórico é consultado para contextualizar os próximos turnos.
Na primeira consulta a uma FAQ, os pesos do modelo de embeddings são baixados para
`.venv/hf-cache` (ou para o caminho definido por `HF_HOME`).

## Testes

```powershell
uv run pytest
```

## Classificação de materiais da EJA

Coloque os arquivos `.pptx` em `data/raw` ou informe outra pasta para extrair
títulos e textos dos slides e gerar os embeddings:

```powershell
python scripts/ml/prep_eja.py
python scripts/ml/prep_eja.py --input-dir "C:\caminho\para\apresentacoes"
```

O pré-processamento grava a planilha estruturada e os conjuntos de embeddings
em `data/processed`. Para supervisionar o modelo, preencha a coluna `categoria`
na planilha gerada e rode o pré-processamento novamente; os rótulos anotados
para os slides correspondentes serão preservados. Os conjuntos são separados por
apresentação com `StratifiedGroupKFold`: slides de um mesmo arquivo não podem
aparecer em treino, validação e teste. São necessárias pelo menos três
apresentações distintas por categoria. O script salva também os grupos de cada
conjunto para auditoria. Depois, treine e avalie o MLP:

```powershell
python scripts/ml/treino_mlp_eja.py
```

Os gráficos são salvos em `resultados/sup/graficos` e o modelo em
`data/processed/modelo_mlp_eja.pkl`. A categoria inicial de cada slide é `0`.
O modelo é avaliado na validação e no teste final com F1-macro; o teste não é
usado para ajuste. A curva de aprendizado também usa folds agrupados por
apresentação. O SMOTE é aplicado somente ao treino e ignorado com aviso quando
não há classes suficientes para balancear.
As dependências específicas desses utilitários estão listadas no início de cada
script; elas não são necessárias para executar o tutor.

## Recuperação de FAQs (RAG)

O tool `consultar_conteudo_eja` mantém a busca curricular existente e também pode
recuperar até três FAQs completas da base consolidada. Cada pergunta, resposta e
metadados (ID, tema, fonte, PDF, páginas e origem) formam um único registro: as
203 FAQs atuais têm perguntas de até 98 caracteres e respostas de até 327, então
dividi-las ou sobrepor fragmentos separaria informações que já são unidades
semânticas completas. Não há chunking adicional nem overlap.

A recuperação usa embeddings do modelo
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, similaridade de
cosseno e cache em memória do índice. O modelo multilíngue atende consultas em
português; os pesos são carregados sob demanda. O padrão é salvar os pesos em
`.venv/hf-cache` quando a aplicação roda dentro de um ambiente virtual. Defina
`HF_HOME` para escolher outro cache e `CALMA_EMBEDDING_MODEL` para substituir o
modelo. O BERT de `prep_eja.py` permanece separado e é usado na classificação de
slides.

O limiar mínimo padrão (`CALMA_FAQ_MIN_SIMILARITY`) e a margem de ambiguidade
(`CALMA_FAQ_AMBIGUITY_MARGIN_RATIO`) foram calibrados com o conjunto `val` do
dataset. Para repetir a avaliação:

```powershell
uv run python scripts/avaliar_faqs.py
```

O script mede P@1, R@1 e F1@1 em `train`, `val` e `test`; calibra os dois
limiares somente em `val` e deixa `test` para avaliação final. Alvos cujo ID não
existe na base FAQ são contabilizados à parte, pois não podem ser recuperados por
este corpus. `meta.tipo` indica suporte no contexto original, não
relevância para a busca: IDs presentes no corpus são alvos de recuperação nos dois
rótulos. O dataset não contém rótulos específicos de ambiguidade; a margem é
escolhida pelo F1 da decisão de recuperação/abstenção e precisa ser reavaliada
quando a base ou o modelo mudar. O agente deve apoiar fatos do material apenas
nas FAQs retornadas. Para detalhes do fluxo e dos componentes, consulte
[`docs/arquitetura_rag.md`](docs/arquitetura_rag.md).

O arquivo `data/faq/FAQ_CONSOLIDADO.md` é a base versionada usada por padrão. Para
usar outra localização, defina `CALMA_FAQ_PATH`; se o arquivo não estiver disponível,
o tool informa que não pode recuperar FAQs e mantém a busca curricular existente.

## Privacidade

Nunca faça commit do `.env` nem compartilhe sua chave de API. O histórico dos
estudantes fica em `data/estudantes/`, uma pasta local ignorada pelo Git. Não coloque
informações pessoais de estudantes nas bases versionadas de `data/faq/` ou
`data/datasets/`.