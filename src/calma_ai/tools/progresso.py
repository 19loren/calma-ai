from __future__ import annotations

import json
import logging
import os
import re
import socket
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from calma_ai.models import Diagnostico

LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=ROOT_DIR / ".env")

LIMITE_HISTORICO = 50
DEFAULT_DB_NAME = "calma_ai_db"


def _force_ipv4() -> None:
    old_getaddrinfo = socket.getaddrinfo
    if getattr(old_getaddrinfo, "_ipv4_forced", False):
        return

    def new_getaddrinfo(*args: Any, **kwargs: Any) -> list[Any]:
        responses = old_getaddrinfo(*args, **kwargs)
        return [r for r in responses if r[0] == socket.AF_INET]

    new_getaddrinfo._ipv4_forced = True  # type: ignore[attr-defined]
    socket.getaddrinfo = new_getaddrinfo


def _usar_mongo() -> bool:
    return bool(os.getenv("MONGODB_URI", "").strip()) and os.getenv("CALMA_DATA_DIR") is None


@lru_cache(maxsize=2)
def _get_mongo_db(uri: str, db_name: str) -> Any:
    from pymongo import MongoClient

    _force_ipv4()
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    return client[db_name]


def _db() -> Any | None:
    if not _usar_mongo():
        return None
    uri = os.getenv("MONGODB_URI", "").strip()
    db_name = os.getenv("MONGODB_DB_NAME", DEFAULT_DB_NAME).strip()
    try:
        return _get_mongo_db(uri, db_name)
    except Exception as erro:
        LOGGER.warning("Falha ao conectar ao MongoDB Atlas, usando fallback local: %s", erro)
        return None


def pasta_dados() -> Path:
    return Path(os.getenv("CALMA_DATA_DIR", ROOT_DIR / "data" / "estudantes"))


def _arquivo_perfil(estudante_id: str) -> Path:
    seguro = re.sub(r"[^a-z0-9_-]+", "-", estudante_id.strip().lower()).strip("-") or "estudante"
    return pasta_dados() / f"{seguro}.json"


def _perfil_padrao(estudante_id: str) -> dict[str, Any]:
    return {
        "estudante_id": estudante_id,
        "conceitos_dominados": [],
        "dificuldades": {},
        "historico": [],
        "aulas_concluidas": [],
        "aula_atual_por_materia": {},
        "notas_exercicios": {},
    }


def _garantir_campos_perfil(perfil: dict[str, Any], estudante_id: str) -> dict[str, Any]:
    base = _perfil_padrao(estudante_id)
    base.update(perfil)
    return base


def _salvar_local_atomico(estudante_id: str, perfil: dict[str, Any]) -> Path:
    arquivo = _arquivo_perfil(estudante_id)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    arquivo_temporario: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=arquivo.parent,
            prefix=f".{arquivo.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporario:
            arquivo_temporario = Path(temporario.name)
            json.dump(perfil, temporario, ensure_ascii=False, indent=2)
            temporario.flush()
            os.fsync(temporario.fileno())
        os.replace(arquivo_temporario, arquivo)
    finally:
        if arquivo_temporario is not None and arquivo_temporario.exists():
            arquivo_temporario.unlink()
    return arquivo


def carregar_perfil(estudante_id: str) -> dict[str, Any]:
    db = _db()
    if db is not None:
        try:
            doc = db["progresso"].find_one({"estudante_id": estudante_id}, {"_id": 0})
            perfil = _garantir_campos_perfil(doc or {}, estudante_id)
            aluno = db["alunos"].find_one({"estudante_id": estudante_id}, {"_id": 0})
            if aluno:
                perfil["dados_cadastrais"] = {
                    k: v
                    for k, v in aluno.items()
                    if k in ("nome_preferido", "profissao", "observacao")
                }
            return perfil
        except Exception as erro:
            LOGGER.warning("Falha ao ler perfil no MongoDB Atlas, usando local: %s", erro)

    arquivo = _arquivo_perfil(estudante_id)
    if not arquivo.exists():
        return _perfil_padrao(estudante_id)
    return _garantir_campos_perfil(
        json.loads(arquivo.read_text(encoding="utf-8")), estudante_id
    )


def registrar_interacao(estudante_id: str, pergunta: str, diagnostico: Diagnostico) -> Path:
    perfil = carregar_perfil(estudante_id)

    dominados = set(perfil.get("conceitos_dominados", []))
    dominados.update(diagnostico.conceitos_dominados)
    perfil["conceitos_dominados"] = sorted(dominados)

    dificuldade = diagnostico.dificuldade_principal.strip()
    if dificuldade and dificuldade.lower() != "nenhuma":
        dificuldades = perfil.setdefault("dificuldades", {})
        dificuldades[dificuldade] = dificuldades.get(dificuldade, 0) + 1

    registro_turno = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "topico": diagnostico.topico,
        "pergunta": pergunta,
        "raciocinio_correto": diagnostico.raciocinio_do_estudante_correto,
        "acao": diagnostico.proxima_acao.value,
    }
    historico = perfil.setdefault("historico", [])
    historico.append(registro_turno)
    perfil["historico"] = historico[-LIMITE_HISTORICO:]

    db = _db()
    if db is not None:
        try:
            para_salvar = {k: v for k, v in perfil.items() if k != "dados_cadastrais"}
            db["progresso"].update_one(
                {"estudante_id": estudante_id},
                {"$set": para_salvar},
                upsert=True,
            )
            db["sessoes_tutoria"].insert_one(
                {"estudante_id": estudante_id, **registro_turno}
            )
        except Exception as erro:
            LOGGER.warning("Falha ao gravar interação no MongoDB Atlas: %s", erro)

    return _salvar_local_atomico(estudante_id, perfil)


def _carregar_aulas_locais(materia: str | None = None) -> list[dict[str, Any]]:
    pasta_lotes = Path(os.getenv("CALMA_LOTES_DIR", ROOT_DIR / "data" / "faq" / "lotes"))
    if not pasta_lotes.exists():
        return []
    aulas: list[dict[str, Any]] = []
    for arquivo in sorted(pasta_lotes.rglob("exercicios.json")):
        conteudo = json.loads(arquivo.read_text(encoding="utf-8"))
        for aula in conteudo.get("aulas", []):
            if materia is None or aula.get("materia") == materia:
                aulas.append(aula)
    aulas.sort(key=lambda a: (int(a.get("modulo_ordem", 0)), int(a.get("aula_ordem", 0))))
    return aulas


def _buscar_aulas_materia(materia: str) -> list[dict[str, Any]]:
    db = _db()
    if db is not None:
        try:
            docs = list(
                db["trilhas"]
                .find({"materia": materia}, {"_id": 0})
                .sort([("modulo_ordem", 1), ("aula_ordem", 1)])
            )
            if docs:
                return docs
        except Exception as erro:
            LOGGER.warning("Falha ao ler trilhas no MongoDB Atlas, usando lotes locais: %s", erro)
    return _carregar_aulas_locais(materia)


def listar_trilha_materia(materia: str, estudante_id: str) -> list[dict[str, Any]]:
    """Retorna as aulas da matéria em ordem sequencial com status de progressão."""
    aulas = _buscar_aulas_materia(materia)
    perfil = carregar_perfil(estudante_id)
    concluidas = set(perfil.get("aulas_concluidas", []))

    resultado: list[dict[str, Any]] = []
    anterior_concluida = True
    for aula in aulas:
        aula_id = aula["aula_id"]
        if aula_id in concluidas:
            status = "concluida"
            anterior_concluida = True
        elif anterior_concluida:
            status = "disponivel"
            anterior_concluida = False
        else:
            status = "bloqueada"
            anterior_concluida = False

        resultado.append({**aula, "status": status})
    return resultado


def submeter_exercicios_aula(
    estudante_id: str, aula_id: str, respostas_indices: list[int]
) -> dict[str, Any]:
    """Corrige a lista de exercícios da aula, registra nota e desbloqueia a próxima aula."""
    todas_aulas = _carregar_aulas_locais()
    db = _db()
    aula_encontrada: dict[str, Any] | None = None

    if db is not None:
        try:
            aula_encontrada = db["trilhas"].find_one({"aula_id": aula_id}, {"_id": 0})
        except Exception as erro:
            LOGGER.warning("Falha ao buscar aula no MongoDB Atlas: %s", erro)

    if aula_encontrada is None:
        aula_encontrada = next((a for a in todas_aulas if a["aula_id"] == aula_id), None)
    if aula_encontrada is None:
        raise ValueError(f"Aula '{aula_id}' não encontrada na trilha.")

    materia = aula_encontrada["materia"]
    trilha = listar_trilha_materia(materia, estudante_id)
    item_trilha = next((a for a in trilha if a["aula_id"] == aula_id), None)
    if item_trilha and item_trilha["status"] == "bloqueada":
        raise PermissionError(f"A aula '{aula_id}' ainda está bloqueada para '{estudante_id}'.")

    exercicios = aula_encontrada.get("exercicios", [])
    if len(respostas_indices) != len(exercicios):
        raise ValueError(
            f"Esperadas {len(exercicios)} respostas, mas foram enviadas {len(respostas_indices)}."
        )

    acertos = 0
    detalhes: list[dict[str, Any]] = []
    for exercicio, resp in zip(exercicios, respostas_indices):
        correta = int(exercicio["correta_index"])
        acertou = int(resp) == correta
        if acertou:
            acertos += 1
        detalhes.append(
            {
                "exercicio_id": exercicio["exercicio_id"],
                "acertou": acertou,
                "resposta_enviada": int(resp),
                "correta_index": correta,
                "dica": "" if acertou else exercicio.get("dica", ""),
            }
        )

    total = len(exercicios)
    aproveitamento = (acertos / total) if total > 0 else 1.0
    nota_minima = float(aula_encontrada.get("nota_minima_aprovacao", 0.7))
    aprovado = aproveitamento >= nota_minima

    perfil = carregar_perfil(estudante_id)
    notas = perfil.setdefault("notas_exercicios", {})
    notas[aula_id] = {
        "acertos": acertos,
        "total": total,
        "aproveitamento": round(aproveitamento, 2),
        "aprovado": aprovado,
        "data": datetime.now().isoformat(timespec="seconds"),
    }

    proxima_aula_id: str | None = None
    if aprovado:
        concluidas = perfil.setdefault("aulas_concluidas", [])
        if aula_id not in concluidas:
            concluidas.append(aula_id)

        ids_ordenados = [a["aula_id"] for a in trilha]
        idx = ids_ordenados.index(aula_id)
        if idx + 1 < len(ids_ordenados):
            proxima_aula_id = ids_ordenados[idx + 1]
            perfil.setdefault("aula_atual_por_materia", {})[materia] = proxima_aula_id

    if db is not None:
        try:
            para_salvar = {k: v for k, v in perfil.items() if k != "dados_cadastrais"}
            db["progresso"].update_one(
                {"estudante_id": estudante_id},
                {"$set": para_salvar},
                upsert=True,
            )
        except Exception as erro:
            LOGGER.warning("Falha ao atualizar progresso de aula no MongoDB Atlas: %s", erro)

    _salvar_local_atomico(estudante_id, perfil)

    return {
        "aula_id": aula_id,
        "aprovado": aprovado,
        "acertos": acertos,
        "total": total,
        "aproveitamento": round(aproveitamento, 2),
        "proxima_aula_id": proxima_aula_id,
        "detalhes": detalhes,
    }


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