# calma-ai

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and a Gemini API key.
CrewAI AMP login is not required for this local declarative crew.

```powershell
uv sync
Copy-Item .env.example .env
```

Add your `GEMINI_API_KEY` to `.env`, then run the tutor:

```powershell
uv run crewai run
```

The CLI asks for the student question. The subject and everyday-life context
have defaults in `crew.jsonc`; edit them there or override them when prompted
for any missing values. The tutor is defined in `agents/tutor.jsonc`.

Never commit `.env` or share your API key. `.env.example` is a safe template;
the real `.env` is ignored by Git.