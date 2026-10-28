# calma-ai

Tutor socrático para estudantes adultos da EJA. Ele não entrega respostas prontas:
ajuda o estudante a chegar nelas, um passo de cada vez.

## Arquitetura

```
mensagem do estudante
   │
   ▼
Agente reflexivo (regras if/else + regex, sem LLM)          src/calma_ai/reflexo.py
   │  R1 crise → CVV 188 (para aqui)
   │  R2 vazia / R3 saudação → resposta fixa (para aqui)
   │  R4 desânimo → acolhe e encaminha
   │  R5 pede resposta pronta → recusa e encaminha
   │  R6 dúvida de conteúdo (pergunta, número ou tentativa) → encaminha
   │  R7 nenhuma dúvida identificada (ex.: só um nome) → pede a dúvida (para aqui)
   ▼
Equipe CrewAI (Gemini)                                      src/calma_ai/crew.py
   1. Diagnosticador: consulta histórico, base de conteúdo e calculadora;
      decide a próxima ação (fazer_pergunta, dar_dica, validar_e_avancar,
      propor_exercicio) e devolve um diagnóstico estruturado.
   2. Tutor: escreve a mensagem executando a ação decidida.
   3. Revisor: garante que a mensagem não revela a resposta final.
      Um guardrail em código exige que a mensagem tenha uma pergunta.
   ▼
Progresso do estudante salvo em data/estudantes/<id>.json
```

- Agentes e tarefas: `src/calma_ai/config/agents.yaml` e `tasks.yaml`.
- Ferramentas: `src/calma_ai/tools/` (calculadora segura, base de conteúdo, perfil do estudante).
- Base de conhecimento: `src/calma_ai/knowledge/conteudos.json`.

## Setup

Requer Python 3.12+, [uv](https://docs.astral.sh/uv/) e uma chave da API Gemini.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install uv
uv sync
```

Crie um arquivo `.env` com `GEMINI_API_KEY=<sua chave>`.

## Uso

Somente o agente reflexivo (não usa LLM nem chave de API):

```powershell
uv run reflexo
```

Fluxo completo (agente reflexivo + equipe de agentes):

```powershell
uv run crewai run
```

O CLI pede o identificador do estudante, a dúvida (ou tentativa) e, opcionalmente,
o conteúdo curricular e o contexto do dia a dia.

## Testes

```powershell
uv run pytest
```

## Privacidade

Nunca faça commit do `.env` nem compartilhe sua chave de API. A pasta `data/`
guarda o histórico dos estudantes e também fica fora do Git.