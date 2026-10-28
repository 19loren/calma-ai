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

No fluxo completo, acolhimentos e limites do agente reflexivo são incorporados à
orientação do tutor em uma única resposta. Perguntas-guia são sugestões: o tutor deve
adaptá-las à dúvida, à tentativa e ao contexto reais do estudante.

- Agentes e tarefas: `src/calma_ai/config/agents.yaml` e `tasks.yaml`.
- Ferramentas: `src/calma_ai/tools/` (calculadora segura, base de conteúdo, perfil do estudante).
- Base de conhecimento: `src/calma_ai/knowledge/conteudos.json`.

## Setup

Requer Python 3.12+ e [uv](https://docs.astral.sh/uv/). O fluxo completo também requer
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

## Privacidade

Nunca faça commit do `.env` nem compartilhe sua chave de API. A pasta `data/`
guarda o histórico dos estudantes e também fica fora do Git.