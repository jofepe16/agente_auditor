from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
import urllib.request
import warnings
from contextlib import contextmanager, redirect_stderr
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from rich.console import Console
from rich.table import Table

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _showwarning = warnings.showwarning
    warnings.showwarning = lambda *args, **kwargs: None
    from langgraph.graph import END, START, StateGraph

    warnings.showwarning = _showwarning


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"
MONEY_PATTERN = re.compile(r"\$\s?([0-9][0-9,\.]*)\s*(?:USD)?", re.IGNORECASE)


class Action(str, Enum):
    APPROVE = "aprobar"
    ESCALATE = "rechazar_o_escalar"
    BLOCK = "bloquear"
    UNKNOWN = "indeterminado"


class Severity(str, Enum):
    CRITICAL = "critica"
    HIGH = "alta"
    MEDIUM = "media"
    LOW = "baja"


class AuditCase(BaseModel):
    id_caso: int = Field(gt=0)
    contexto_rag: str = Field(min_length=10)
    respuesta_agent_b: str = Field(min_length=10)


class ControlRule(BaseModel):
    id: str
    descripcion: str
    tipo: str
    indicadores_contexto: List[str] = Field(min_length=1)
    severidad: Severity
    accion_esperada: Optional[Action] = None
    accion_esperada_si_cumple: Optional[Action] = None
    accion_esperada_si_incumple: Optional[Action] = None

    @model_validator(mode="after")
    def has_expected_action(self) -> "ControlRule":
        if not any(
            [
                self.accion_esperada,
                self.accion_esperada_si_cumple,
                self.accion_esperada_si_incumple,
            ]
        ):
            raise ValueError(f"Control sin accion esperada: {self.id}")
        return self


class FidelityWeights(BaseModel):
    alineacion_accion: float = Field(ge=0, le=1)
    cumplimiento_numerico: float = Field(ge=0, le=1)
    controles_criticos: float = Field(ge=0, le=1)
    consistencia_semantica: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def totals_one(self) -> "FidelityWeights":
        total = (
            self.alineacion_accion
            + self.cumplimiento_numerico
            + self.controles_criticos
            + self.consistencia_semantica
        )
        if abs(total - 1) > 0.001:
            raise ValueError(f"Los pesos deben sumar 1.0; suma actual: {total}")
        return self


class RulesConfig(BaseModel):
    version: str
    umbral_aprobacion: float = Field(ge=0, le=100)
    umbral_similitud_semantica: float = Field(ge=0, le=100)
    pesos_indice_fidelidad: FidelityWeights
    acciones_respuesta: Dict[Action, List[str]]
    controles: List[ControlRule]

    @field_validator("acciones_respuesta")
    @classmethod
    def requires_actions(cls, value: Dict[Action, List[str]]) -> Dict[Action, List[str]]:
        required = {Action.APPROVE, Action.ESCALATE, Action.BLOCK}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Faltan acciones: {', '.join(action.value for action in missing)}")
        return value


@dataclass
class NumericFinding:
    compliant: bool
    context_limit: Optional[float] = None
    approved_amount: Optional[float] = None


class LlmFinding(BaseModel):
    accion_detectada: Action
    contradiccion_normativa: bool
    nivel_riesgo: str = Field(pattern="^(bajo|medio|alto|critico)$")
    razon: str = Field(min_length=10)

    @field_validator("accion_detectada", mode="before")
    @classmethod
    def normalize_action(cls, value: Any) -> str:
        if isinstance(value, Action):
            return value.value
        aliases = {
            "escalar": Action.ESCALATE.value,
            "rechazar": Action.ESCALATE.value,
            "revision": Action.ESCALATE.value,
            "bloqueo": Action.BLOCK.value,
        }
        normalized = normalize_text(str(value)).replace(" ", "_")
        return aliases.get(normalized, normalized)


@dataclass
class RagDocument:
    source: str
    text: str
    score: float


@dataclass
class AuditResult:
    id_caso: int
    estado_transaccion: str
    indice_fidelidad_analitica: float
    consistencia_semantica: float
    diagnostico: str
    decision_recomendada: str
    controles_activados: List[str]
    accion_detectada: str
    accion_esperada: str
    cumplimiento_numerico: bool
    fuentes_rag: List[str]
    analisis_llm: str


class AuditState(TypedDict, total=False):
    case: AuditCase
    rag_documents: List[RagDocument]
    rag_score: float
    controls: List[ControlRule]
    numeric: NumericFinding
    llm_finding: LlmFinding
    detected_action: Action
    expected_action: Action
    result: AuditResult


class VectorPolicyIndex:
    def __init__(self, policies_path: Path, model_name: str = MODEL_NAME) -> None:
        self.model = self._load_model(model_name)
        self.sources, self.documents = self._load_documents(policies_path)
        self.embeddings = self._normalize(self.model.encode(self.documents, convert_to_numpy=True))

    def search(self, query: str, k: int = 2) -> List[RagDocument]:
        query_embedding = self._normalize(self.model.encode([query], convert_to_numpy=True))[0]
        scores = self.embeddings @ query_embedding
        ranking = np.argsort(scores)[::-1][:k]
        return [
            RagDocument(
                source=self.sources[index],
                text=self.documents[index],
                score=round(float(scores[index] * 100), 2),
            )
            for index in ranking
        ]

    def similarity(self, left: str, right: str) -> float:
        embeddings = self._normalize(self.model.encode([left, right], convert_to_numpy=True))
        return round(float(np.dot(embeddings[0], embeddings[1]) * 100), 2)

    def _load_model(self, model_name: str):
        with suppress_native_stderr():
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(model_name, local_files_only=True)

    def _load_documents(self, policies_path: Path) -> tuple[List[str], List[str]]:
        paths = sorted(policies_path.glob("*.md"))
        if not paths:
            raise SystemExit(f"No hay politicas .md en {policies_path}")
        return [str(path) for path in paths], [path.read_text(encoding="utf-8") for path in paths]

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL, url: str = "http://127.0.0.1:11434/api/generate") -> None:
        self.model = model
        self.url = url

    def analyze(self, case: AuditCase, policies: Sequence[RagDocument]) -> LlmFinding:
        return self.analyze_many([(case, policies)])[case.id_caso]

    def analyze_many(self, items: Sequence[tuple[AuditCase, Sequence[RagDocument]]]) -> Dict[int, LlmFinding]:
        cases_payload = []
        for case, _ in items:
            cases_payload.append(
                {
                    "id_caso": case.id_caso,
                    "respuesta_agent_b": case.respuesta_agent_b,
                }
            )
        prompt = f"""
Clasifica respuestas de un agente de seguros. Devuelve solo JSON.
accion_detectada debe ser: aprobar, rechazar_o_escalar, bloquear o indeterminado.
nivel_riesgo debe ser: bajo, medio, alto o critico.
{json.dumps(cases_payload, ensure_ascii=False)}
Formato: {{"casos":[{{"id_caso":1,"accion_detectada":"aprobar","contradiccion_normativa":false,"nivel_riesgo":"bajo","razon":"breve"}}]}}
""".strip()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 320},
            "keep_alive": "10m",
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                raw_response = json.loads(response.read().decode("utf-8"))["response"]
            payload = normalize_llm_payload(json.loads(raw_response))
            findings = {}
            for item in payload.get("casos", []):
                normalized_item = normalize_llm_payload(item)
                case_id = int(normalized_item["id_caso"])
                findings[case_id] = LlmFinding.model_validate(normalized_item)
            missing = {case.id_caso for case, _ in items}.difference(findings)
            if missing:
                raise ValueError(f"Ollama no devolvio analisis para casos: {sorted(missing)}")
            return findings
        except Exception as exc:
            raise SystemExit(f"No fue posible obtener analisis batch desde Ollama: {exc}") from exc

    def _analyze_single_legacy(self, case: AuditCase, policies: Sequence[RagDocument]) -> LlmFinding:
        policy_context = "\n\n".join(
            f"Fuente: {document.source}\n{document.text}" for document in policies
        )
        prompt = f"""
Actua como auditor de seguros. Responde solo JSON.
Acciones validas: aprobar, rechazar_o_escalar, bloquear, indeterminado.
Riesgos validos: bajo, medio, alto, critico.

Politicas:
{policy_context[:1800]}

Caso:
{case.contexto_rag}

Decision Agent B:
{case.respuesta_agent_b}

JSON:
{{"accion_detectada":"","contradiccion_normativa":false,"nivel_riesgo":"","razon":""}}
""".strip()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 160},
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                raw_response = json.loads(response.read().decode("utf-8"))["response"]
            return LlmFinding.model_validate(normalize_llm_payload(json.loads(raw_response)))
        except Exception as exc:
            raise SystemExit(f"No fue posible obtener analisis estructurado desde Ollama: {exc}") from exc


class DeterministicLlmClient:
    def analyze_many(self, items: Sequence[tuple[AuditCase, Sequence[RagDocument]]]) -> Dict[int, LlmFinding]:
        return {case.id_caso: self.analyze(case, policies) for case, policies in items}

    def analyze(self, case: AuditCase, policies: Sequence[RagDocument]) -> LlmFinding:
        response = normalize_text(case.respuesta_agent_b)
        if "no puedo procesar" in response or "analista" in response:
            action = Action.ESCALATE
            risk = "medio"
        elif "bloquear" in response or "bloqueado" in response:
            action = Action.BLOCK
            risk = "critico"
        elif any(term in response for term in ["aprob", "procedemos", "emitida", "exitoso"]):
            action = Action.APPROVE
            risk = "bajo"
        else:
            action = Action.UNKNOWN
            risk = "alto"
        return LlmFinding(
            accion_detectada=action,
            contradiccion_normativa=False,
            nivel_riesgo=risk,
            razon="Clasificacion estructurada de la respuesta del agente.",
        )


class AuditAgent:
    def __init__(self, rules: RulesConfig, index: VectorPolicyIndex, llm: OllamaClient) -> None:
        self.rules = rules
        self.index = index
        self.llm = llm
        self.graph = self._build_graph()

    def audit(self, cases: Sequence[AuditCase]) -> List[AuditResult]:
        prepared = []
        for case in cases:
            documents = self.index.search(f"{case.contexto_rag}\n{case.respuesta_agent_b}", k=2)
            prepared.append((case, documents))
        llm_findings = self.llm.analyze_many(prepared)
        results = []
        for case, documents in prepared:
            state = {
                "case": case,
                "rag_documents": documents,
                "rag_score": round(sum(document.score for document in documents) / len(documents), 2),
                "llm_finding": llm_findings[case.id_caso],
            }
            results.append(self.graph.invoke(state)["result"])
        return results

    def _build_graph(self):
        graph = StateGraph(AuditState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("llm_review", self._llm_review)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("decide", self._decide)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "llm_review")
        graph.add_edge("llm_review", "evaluate")
        graph.add_edge("evaluate", "decide")
        graph.add_edge("decide", END)
        return graph.compile()

    def _retrieve(self, state: AuditState) -> AuditState:
        if "rag_documents" in state and "rag_score" in state:
            return {}
        case = state["case"]
        query = f"{case.contexto_rag}\n{case.respuesta_agent_b}"
        documents = self.index.search(query, k=2)
        return {
            "rag_documents": documents,
            "rag_score": round(sum(document.score for document in documents) / len(documents), 2),
        }

    def _llm_review(self, state: AuditState) -> AuditState:
        if "llm_finding" in state:
            return {}
        return {"llm_finding": self.llm.analyze(state["case"], state["rag_documents"])}

    def _evaluate(self, state: AuditState) -> AuditState:
        case = state["case"]
        controls = self._active_controls(case.contexto_rag)
        numeric = self._numeric_finding(case)
        detected_action = state["llm_finding"].accion_detectada
        expected_action = self._expected_action(controls, numeric, detected_action)
        return {
            "controls": controls,
            "numeric": numeric,
            "detected_action": detected_action,
            "expected_action": expected_action,
        }

    def _decide(self, state: AuditState) -> AuditState:
        case = state["case"]
        controls = state["controls"]
        numeric = state["numeric"]
        llm_finding = state["llm_finding"]
        detected = state["detected_action"]
        expected = state["expected_action"]
        consistency = self._semantic_consistency(case.respuesta_agent_b, expected, state["rag_score"])
        index = self._fidelity_index(detected, expected, numeric, controls, consistency)
        result = AuditResult(
            id_caso=case.id_caso,
            estado_transaccion=self._state(index, expected, detected),
            indice_fidelidad_analitica=index,
            consistencia_semantica=consistency,
            diagnostico=self._diagnostic(controls, numeric, detected, expected, consistency),
            decision_recomendada=self._recommendation(expected),
            controles_activados=[control.id for control in controls],
            accion_detectada=detected.value,
            accion_esperada=expected.value,
            cumplimiento_numerico=numeric.compliant,
            fuentes_rag=[document.source for document in state["rag_documents"]],
            analisis_llm=llm_finding.razon,
        )
        return {"result": result}

    def _active_controls(self, context: str) -> List[ControlRule]:
        context = normalize_text(context)
        return [
            control
            for control in self.rules.controles
            if any(normalize_text(indicator) in context for indicator in control.indicadores_contexto)
        ]

    def _detect_action(self, response: str) -> Action:
        scores = {}
        for action, examples in self.rules.acciones_respuesta.items():
            scores[action] = max(self.index.similarity(response, example) for example in examples)
        action, score = max(scores.items(), key=lambda item: item[1])
        return action if score >= self.rules.umbral_similitud_semantica else Action.UNKNOWN

    def _numeric_finding(self, case: AuditCase) -> NumericFinding:
        limits = parse_money_values(case.contexto_rag)
        amounts = parse_money_values(case.respuesta_agent_b)
        if not limits or not amounts:
            return NumericFinding(compliant=True, context_limit=limits[0] if limits else None)
        limit = max(limits)
        amount = max(amounts)
        return NumericFinding(compliant=amount <= limit, context_limit=limit, approved_amount=amount)

    def _expected_action(
        self,
        controls: Sequence[ControlRule],
        numeric: NumericFinding,
        detected: Action,
    ) -> Action:
        if any(control.tipo == "bloqueo_critico" for control in controls):
            return Action.BLOCK
        if not numeric.compliant:
            return Action.ESCALATE
        if any(control.accion_esperada == Action.ESCALATE for control in controls):
            return Action.ESCALATE
        return Action.APPROVE if detected == Action.APPROVE else Action.ESCALATE

    def _semantic_consistency(self, response: str, expected: Action, rag_score: float) -> float:
        action_score = max(
            self.index.similarity(response, example)
            for example in self.rules.acciones_respuesta[expected]
        )
        return round(action_score * 0.70 + rag_score * 0.30, 2)

    def _fidelity_index(
        self,
        detected: Action,
        expected: Action,
        numeric: NumericFinding,
        controls: Sequence[ControlRule],
        consistency: float,
    ) -> float:
        weights = self.rules.pesos_indice_fidelidad
        score = (
            self._action_score(detected, expected) * weights.alineacion_accion
            + (100 if numeric.compliant else 0) * weights.cumplimiento_numerico
            + self._critical_score(controls, detected, expected) * weights.controles_criticos
            + consistency * weights.consistencia_semantica
        )
        return round(score, 2)

    def _action_score(self, detected: Action, expected: Action) -> float:
        if detected == expected:
            return 100
        if expected == Action.BLOCK and detected == Action.ESCALATE:
            return 65
        if expected == Action.ESCALATE and detected == Action.BLOCK:
            return 85
        return 40 if detected == Action.UNKNOWN else 0

    def _critical_score(self, controls: Sequence[ControlRule], detected: Action, expected: Action) -> float:
        has_critical = any(control.severidad == Severity.CRITICAL for control in controls)
        if has_critical and detected != Action.BLOCK:
            return 0
        return 100 if detected == expected else 60

    def _state(self, index: float, expected: Action, detected: Action) -> str:
        if expected == Action.BLOCK and detected != Action.BLOCK:
            return "NO CONFORME - BLOQUEO CRITICO"
        if index < self.rules.umbral_aprobacion:
            return "NO CONFORME - REQUIERE REVISION"
        if detected == Action.ESCALATE:
            return "CONFORME - ESCALADA A ANALISTA"
        if detected == Action.BLOCK:
            return "CONFORME - BLOQUEADA"
        return "CONFORME - APROBADA"

    def _diagnostic(
        self,
        controls: Sequence[ControlRule],
        numeric: NumericFinding,
        detected: Action,
        expected: Action,
        consistency: float,
    ) -> str:
        parts = []
        if controls:
            parts.append("Controles activados: " + ", ".join(control.id for control in controls) + ".")
        if numeric.context_limit and numeric.approved_amount:
            relation = "esta dentro del" if numeric.compliant else "supera el"
            parts.append(f"El monto aprobado USD {numeric.approved_amount:,.0f} {relation} limite USD {numeric.context_limit:,.0f}.")
        if detected == expected:
            parts.append(f"La accion detectada ({detected.value}) coincide con la accion esperada.")
        else:
            parts.append(f"La accion detectada ({detected.value}) no coincide con la accion esperada ({expected.value}).")
        parts.append(f"Consistencia RAG vectorial/accion: {consistency:.2f}/100.")
        return " ".join(parts)

    def _recommendation(self, expected: Action) -> str:
        return {
            Action.APPROVE: "Continuar la transaccion con trazabilidad.",
            Action.ESCALATE: "Pausar la automatizacion y escalar a analista.",
            Action.BLOCK: "Bloquear la transaccion y escalar a Cumplimiento/SARLAFT.",
            Action.UNKNOWN: "Solicitar revision manual.",
        }[expected]


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().strip()


def normalize_llm_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in payload.items():
        clean_key = normalize_text(key).replace(" ", "_")
        normalized[clean_key] = value
    return normalized


def parse_money_values(text: str) -> List[float]:
    values = []
    for match in MONEY_PATTERN.finditer(text):
        values.append(float(match.group(1).replace(",", "")))
    return values


@contextmanager
def suppress_native_stderr():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 2)
            with redirect_stderr(devnull):
                yield
        finally:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)


def load_cases(path: Path) -> List[AuditCase]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [AuditCase.model_validate(item) for item in payload]
    except (json.JSONDecodeError, ValidationError) as exc:
        raise SystemExit(f"Casos invalidos en {path}: {exc}") from exc


def load_rules(path: Path) -> RulesConfig:
    try:
        return RulesConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise SystemExit(f"Reglas invalidas en {path}: {exc}") from exc


def save_results(path: Path, results: Sequence[AuditResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def render_plain(result: AuditResult) -> str:
    return (
        f"Caso {result.id_caso}: {result.estado_transaccion}\n"
        f"- Índice de Fidelidad Analítica: {result.indice_fidelidad_analitica:.2f}\n"
        f"- Diagnóstico/Razón: {result.diagnostico} Analisis LLM: {result.analisis_llm}"
    )


def render_table(results: Sequence[AuditResult], output: Path) -> None:
    table = Table(title="Agent A | Auditoria con LangGraph y RAG vectorial")
    table.add_column("Caso", justify="right")
    table.add_column("Estado")
    table.add_column("Indice", justify="right")
    table.add_column("RAG/Accion", justify="right")
    table.add_column("LLM")
    table.add_column("Fuentes")
    table.add_column("Decision recomendada")
    for result in results:
        table.add_row(
            str(result.id_caso),
            result.estado_transaccion,
            f"{result.indice_fidelidad_analitica:.2f}",
            f"{result.consistencia_semantica:.2f}",
            result.accion_detectada,
            str(len(result.fuentes_rag)),
            result.decision_recomendada,
        )
    console = Console()
    console.print(table)
    console.print(f"[green]Reporte generado:[/green] {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent A: auditoria con LangGraph y RAG vectorial.")
    subparsers = parser.add_subparsers(dest="command")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--input", default="data/casos_agent_b.json")
    audit.add_argument("--rules", default="reglas.json")
    audit.add_argument("--policies", default="data/politicas")
    audit.add_argument("--output", default="outputs/reporte_auditoria.json")
    audit.add_argument("--ollama-model", default=OLLAMA_MODEL)
    audit.add_argument("--llm-provider", choices=["ollama", "deterministic"], default="ollama")
    audit.add_argument("--plain", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "audit":
        args = build_parser().parse_args(["audit"])

    rules = load_rules(Path(args.rules))
    cases = load_cases(Path(args.input))
    index = VectorPolicyIndex(Path(args.policies))
    llm = OllamaClient(args.ollama_model) if args.llm_provider == "ollama" else DeterministicLlmClient()
    results = AuditAgent(rules, index, llm).audit(cases)
    save_results(Path(args.output), results)

    if args.plain:
        for result in results:
            print(render_plain(result))
            print()
    else:
        render_table(results, Path(args.output))
        print()
        for result in results:
            print(render_plain(result))
            print()


if __name__ == "__main__":
    main()
