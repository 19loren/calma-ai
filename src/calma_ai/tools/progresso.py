import json
import os
import re
from datetime import datetime
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from calma_ai.models import Diagnostico

LIMITE_HISTORICO = 50


def pasta_dados() -> Path:
    return Path(os.getenv("CALMA_DATA_DIR", Path.cwd() / "data" / "estudantes"))


def _arquivo_perfil(estudante_id: str) -> Path:
    # Normaliza o identificador para impedir caminhos arbitrários (ex.: "../").
    seguro = re.sub(r"[^a-z0-9_-]+", "-", estudante_id.strip().lower()).strip("-") or "estudante"
    return pasta_dados() / f"{seguro}.json"


def carregar_perfil(estudante_id: str) -> dict:
    arquivo = _arquivo_perfil(estudante_id)
    if not arquivo.exists():
        return {
            "estudante_id": estudante_id,
            "conceitos_dominados": [],
            "dificuldades": {},
            "historico": [],
        }
    return json.loads(arquivo.read_text(encoding="utf-8"))


def registrar_interacao(estudante_id: str, pergunta: str, diagnostico: Diagnostico) -> Path:
    perfil = carregar_perfil(estudante_id)

    dominados = set(perfil.get("conceitos_dominados", []))
    dominados.update(diagnostico.conceitos_dominados)
    perfil["conceitos_dominados"] = sorted(dominados)

    dificuldade = diagnostico.dificuldade_principal.strip()
    if dificuldade and dificuldade.lower() != "nenhuma":
        dificuldades = perfil.setdefault("dificuldades", {})
        dificuldades[dificuldade] = dificuldades.get(dificuldade, 0) + 1

    historico = perfil.setdefault("historico", [])
    historico.append(
        {
            "data": datetime.now().isoformat(timespec="seconds"),
            "topico": diagnostico.topico,
            "pergunta": pergunta,
            "raciocinio_correto": diagnostico.raciocinio_do_estudante_correto,
            "acao": diagnostico.proxima_acao.value,
        }
    )
    perfil["historico"] = historico[-LIMITE_HISTORICO:]

    arquivo = _arquivo_perfil(estudante_id)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps(perfil, ensure_ascii=False, indent=2), encoding="utf-8")
    return arquivo


class PerfilEstudanteInput(BaseModel):
    estudante_id: str = Field(description="Identificador do estudante.")


class PerfilEstudanteTool(BaseTool):
    name: str = "consultar_perfil_estudante"
    description: str = (
        "Retorna o histórico do estudante: conceitos que já domina, dificuldades "
        "recorrentes (com contagem) e as últimas interações. Use para adaptar o "
        "diagnóstico e evitar repetir abordagens que não funcionaram."
    )
    args_schema: type[BaseModel] = PerfilEstudanteInput

    def _run(self, estudante_id: str) -> str:
        perfil = carregar_perfil(estudante_id)
        if not perfil["historico"]:
            return f"Estudante '{estudante_id}' ainda não tem histórico. Esta é a primeira interação."
        resumo = {**perfil, "historico": perfil["historico"][-5:]}
        return json.dumps(resumo, ensure_ascii=False, indent=2)
