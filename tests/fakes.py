"""Fake LLM implementation for testing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from aegis.models.llm_provider import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)


class FakeLLM(BaseLLMProvider):
    """
    Fake LLM provider for unit and integration testing.
    Records all requests and returns queued responses if available,
    falling back to deterministic schema generation when the queue is empty.
    """

    def __init__(
        self,
        responses: Optional[List[Any]] = None,
        model: Optional[str] = "fake-model",
    ):
        super().__init__(model=model)
        self.responses: List[Any] = list(responses or [])
        self.requests: List[Dict[str, Any]] = []

    def queue_response(self, response: Any) -> None:
        """Queue a response to be returned by next call."""
        self.responses.append(response)

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        self.requests.append({
            "method": "generate_text",
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return str(resp)
        return "Deterministic test response from FakeLLM"

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        self.requests.append({
            "method": "generate_structured",
            "prompt": prompt,
            "schema": schema,
            "system_prompt": system_prompt,
            "temperature": temperature,
        })
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if isinstance(resp, schema):
                return resp
            if isinstance(resp, dict):
                return schema.model_validate(resp)
            return resp

        # Deterministic schema generator fallback (moved from OfflineIntelligentProvider)
        schema_name = schema.__name__
        prompt_lower = prompt.lower()

        if schema_name == "ExecutionPlan":
            has_data = any(w in prompt_lower for w in ["sensor", "data", "upload", "dataset", "csv", "readings", "telemetry"])
            subtasks = [
                {
                    "id": "subtask-1",
                    "title": "Literature & Technical Standards Research",
                    "description": "Retrieve industry predictive maintenance benchmarks and failure mode studies.",
                    "assigned_agent": "research",
                    "required_tools": ["rag_retriever"],
                    "dependencies": [],
                    "status": "pending",
                }
            ]
            if has_data:
                subtasks.extend([
                    {
                        "id": "subtask-2",
                        "title": "Sensor Telemetry Profiling & Anomaly Detection",
                        "description": "Profile sensor distributions, missing values, and calculate statistical anomalies across vibration and temperature channels.",
                        "assigned_agent": "data_analyst",
                        "required_tools": ["python_sandbox", "sql_query"],
                        "dependencies": ["subtask-1"],
                        "status": "pending",
                    },
                ])
            return schema.model_validate({
                "task_id": "plan-fake",
                "original_query": prompt,
                "objective": "Autonomous multi-agent investigation into equipment health and degradation.",
                "rationale": "Decomposing task into rigorous evidence retrieval and empirical telemetry analysis.",
                "subtasks": subtasks,
            })

        elif schema_name == "VerificationResult":
            return schema.model_validate({
                "all_verified": True,
                "verified_claims": [
                    {
                        "claim": "Vibration velocity exceeding critical threshold indicates impending mechanical bearing failure.",
                        "is_supported": True,
                        "confidence": 0.95,
                        "evidence_sources": ["Mechanical Vibration Guidelines", "Turbomachinery Maintenance Report"],
                        "notes": "Corroborated by independent engineering references.",
                    },
                ],
                "unsupported_claims": [],
                "contradictions": [],
                "verification_notes": "All claims empirically supported.",
            })

        elif schema_name == "CriticEvaluation":
            return schema.model_validate({
                "quality_pass": True,
                "score": 9.0,
                "issues_found": [],
                "missing_evidence": [],
                "weak_reasoning": [],
                "contradictions": [],
                "critique_summary": "All specialist conclusions demonstrate consistency.",
                "retry_recommendations": None,
                "target_subtask_for_retry": None,
            })

        elif schema_name == "ExecutiveSummary":
            return schema.model_validate({
                "final_finding": "Predictive maintenance is technically feasible for this equipment.",
                "key_conclusions": [
                    "Vibration and thermal sensor data demonstrate clear degradation signatures.",
                    "Payback period is estimated at under 6 months.",
                ],
                "overall_confidence": 0.90,
                "business_impact": "Prevents catastrophic failure and production downtime.",
                "actionable_recommendations": [
                    "Deploy edge vibration sensors.",
                    "Schedule planned maintenance within the next operational window.",
                ],
            })

        return schema.model_validate({})
