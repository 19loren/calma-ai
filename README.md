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
para os slides correspondentes serão preservados nos arquivos `.npy`. Depois,
treine e avalie o MLP:

```powershell
python scripts/ml/treino_mlp_eja.py
```

Os gráficos são salvos em `resultados/sup/graficos` e o modelo em
`data/processed/modelo_mlp_eja.pkl`. A categoria inicial de cada slide é `0`.
O SMOTE é ignorado com aviso quando não há classes suficientes para balancear.
As dependências específicas desses utilitários estão listadas no início de cada
script; elas não são necessárias para executar o tutor.

## Recuperação de FAQs (RAG)

O tool `consultar_conteudo_eja` mantém a busca curricular existente e também pode
recuperar até três FAQs completas da base consolidada. Cada pergunta, resposta e
metadados (ID, tema, fonte, PDF, páginas e origem) formam um único registro: as
203 FAQs atuais têm perguntas de até 98 caracteres e respostas de até 327, então
dividi-las ou sobrepor fragmentos separaria informações que já são unidades
semânticas completas. Não há chunking adicional nem overlap.

A recuperação usa vetores TF-IDF esparsos, similaridade de cosseno e normalização
de acentos; o índice em memória é adequado a esta base pequena e não exige modelo
denso nem serviço externo. O BERT português em `prep_eja.py` continua reservado à
classificação MLP dos slides e não é usado para recuperar FAQs. Para evitar falsos
positivos, consultas com vários termos precisam compartilhar ao menos dois termos
de conteúdo; resultados com pontuação até 25% da primeira são sinalizados como
ambíguos. Termos únicos que aparecem em temas diferentes também são ambíguos. O
agente deve apoiar fatos do material apenas nas FAQs retornadas.

O arquivo `data/faq/FAQ_CONSOLIDADO.md` é a base versionada usada por padrão. Para
usar outra localização, defina `CALMA_FAQ_PATH`; se o arquivo não estiver disponível,
o tool informa que não pode recuperar FAQs e mantém a busca curricular existente.

## Privacidade

Nunca faça commit do `.env` nem compartilhe sua chave de API. O histórico dos
estudantes fica em `data/estudantes/`, uma pasta local ignorada pelo Git. Não coloque
informações pessoais de estudantes nas bases versionadas de `data/faq/` ou
`data/datasets/`.