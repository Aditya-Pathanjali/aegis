"""Verification Agent for empirical auditing, consistency checking, and hallucination prevention."""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

from aegis.agents.base import BaseAgent
from aegis.config import get_settings
from aegis.llm.pricing import calculate_cost
from aegis.llm.types import Message, TextBlock, Usage
from aegis.models.llm_outputs import JudgeVerdict
from aegis.models.schemas import (
    AgentExecutionResult,
    AgentRole,
    ClaimVerification,
    Finding,
    VerificationMetrics,
    VerificationResult,
)

if TYPE_CHECKING:
    from aegis.llm.base import LLMProvider
    from aegis.tools.registry import ToolRegistry


def extract_numbers(text: str) -> List[float]:
    """Extracts all int and float numbers from text, including negatives and scientific notation."""
    pattern = r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])"
    matches = re.findall(pattern, text)
    numbers: List[float] = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            pass
    return numbers


def number_in_evidence(num: float, evidence_numbers: List[float]) -> bool:
    """Verifies whether num appears in evidence_numbers within 1% relative or 0.01 absolute tolerance."""
    for ev in evidence_numbers:
        if abs(num) < 0.01 or abs(ev) < 0.01:
            if abs(num - ev) <= 0.01:
                return True
        else:
            rel_diff = abs(num - ev) / abs(ev)
            if rel_diff <= 0.01:
                return True
    return False


class VerificationAgent(BaseAgent):
    """
    Independently audits specialist findings against empirical tool output and RAG chunks.
    Performs deterministic numeric verification and invokes an LLM judge with quote verification.
    """

    SYSTEM_PROMPT = (
        "You are the Aegis Independent Verification Judge.\n"
        "Rules:\n"
        "1. Strictly judge if the technical claim is 'supported', 'contradicted', or 'not_enough_info' "
        "using ONLY the provided evidence text.\n"
        "2. For 'supported' or 'contradicted', quote_span MUST be a verbatim substring copied exactly from the evidence.\n"
        "3. If evidence is ambiguous, incomplete, or missing details, choose 'not_enough_info'."
    )

    def __init__(self, llm_provider: Optional[LLMProvider] = None, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(role=AgentRole.VERIFICATION, llm_provider=llm_provider, tool_registry=tool_registry)

    def _resolve_evidence(
        self,
        finding: Finding,
        specialist_results: List[AgentExecutionResult],
    ) -> tuple[str, List[str]]:
        """Resolves cited tool calls and chunk references into verified evidence texts."""
        resolved_texts: List[str] = []
        sources: List[str] = []

        # Map tool calls strictly by call_id (never by tool name)
        all_tool_calls = [tc for res in specialist_results for tc in res.tool_calls]
        tool_by_call_id = {tc.call_id: tc for tc in all_tool_calls if tc.call_id}

        # Check vector store if available
        vector_store = None
        if self.tool_registry:
            rag_tool = self.tool_registry.get_tool("rag_search")
            if rag_tool and hasattr(rag_tool, "retriever") and hasattr(rag_tool.retriever, "vector_store"):
                vector_store = rag_tool.retriever.vector_store

        for ref in finding.evidence:
            ref_id = ref.ref_id
            if ref.kind in ("tool_call", "tool_output"):
                tc = tool_by_call_id.get(ref_id)
                if tc and tc.output is not None:
                    out_str = json.dumps(tc.output) if isinstance(tc.output, (dict, list)) else str(tc.output)
                    resolved_texts.append(out_str)
                    sources.append(ref_id)
            elif ref.kind == "chunk":
                chunk_found = False
                # Try vector store
                if vector_store:
                    chunk = vector_store.get_chunk(ref_id)
                    if chunk:
                        resolved_texts.append(chunk.text)
                        sources.append(ref_id)
                        chunk_found = True

                # Try RAG tool outputs in tool calls
                if not chunk_found:
                    for tc in all_tool_calls:
                        if isinstance(tc.output, dict):
                            ev_list = tc.output.get("evidence", [])
                            if isinstance(ev_list, list):
                                for item in ev_list:
                                    if isinstance(item, dict) and str(item.get("chunk_id")) == ref_id:
                                        content = item.get("content") or item.get("text", "")
                                        if content:
                                            resolved_texts.append(str(content))
                                            sources.append(ref_id)
                                            chunk_found = True
                                            break
                            if chunk_found:
                                break

        return "\n\n".join(resolved_texts), sources

    def _get_judge_llm(self) -> tuple[LLMProvider, str]:
        """Returns (LLMProvider, model_name) for judge model if configured, else self.llm."""
        cfg = getattr(self.llm, "settings", None) or get_settings()
        if cfg.judge_model:
            from aegis.llm import get_llm_provider

            judge_llm = get_llm_provider(settings=cfg.model_copy(update={"model": cfg.judge_model}))
            return judge_llm, judge_llm.model

        return self.llm, self.llm.model

    def execute(
        self,
        specialist_results: Union[List[AgentExecutionResult], Dict[str, AgentExecutionResult]],
        original_query: str = "",
    ) -> VerificationResult:
        t0 = time.perf_counter()

        # Normalize results to list
        results_list: List[AgentExecutionResult] = (
            list(specialist_results.values())
            if isinstance(specialist_results, dict)
            else list(specialist_results)
        )

        claims_verified: List[ClaimVerification] = []
        unsupported_claims: List[str] = []
        judge_provider, judge_model_name = self._get_judge_llm()

        total_judge_in_tokens = 0
        total_judge_out_tokens = 0
        total_judge_calls = 0

        for s_res in results_list:
            for finding in s_res.findings:
                statement = finding.statement.strip()
                if not statement:
                    continue

                evidence_text, sources = self._resolve_evidence(finding, results_list)

                # If no resolvable evidence: status is no_evidence
                if not evidence_text:
                    claim_obj = ClaimVerification(
                        claim=statement,
                        status="no_evidence",
                        is_supported=False,
                        confidence=0.0,
                        evidence_sources=[],
                        notes="No resolvable evidence found for claim.",
                    )
                    claims_verified.append(claim_obj)
                    unsupported_claims.append(statement)
                    continue

                # Deterministic numeric check
                claim_nums = extract_numbers(statement)
                evidence_nums = extract_numbers(evidence_text)
                has_numeric_mismatch = False

                for c_num in claim_nums:
                    if not number_in_evidence(c_num, evidence_nums):
                        has_numeric_mismatch = True
                        break

                # LLM judge call
                prompt = (
                    f"User Query:\n\"{original_query}\"\n\n"
                    f"Claim to Verify:\n\"{statement}\"\n\n"
                    f"Evidence Text:\n\"\"\"{evidence_text}\"\"\"\n\n"
                    "Evaluate whether the claim is 'supported', 'contradicted', or 'not_enough_info'."
                )
                judge_resp = judge_provider.structured(
                    messages=[Message(role="user", content=[TextBlock(text=prompt)])],
                    schema=JudgeVerdict,
                    system=self.SYSTEM_PROMPT,
                )
                total_judge_calls += 1
                total_judge_in_tokens += judge_resp.usage.input_tokens
                total_judge_out_tokens += judge_resp.usage.output_tokens
                judge_model_name = judge_resp.model
                verdict: Literal["supported", "contradicted", "not_enough_info"] = judge_resp.value.verdict
                quote_span = (judge_resp.value.quote_span or "").strip()

                # Downgrade check: quote span must be verbatim in evidence text
                if verdict in ("supported", "contradicted"):
                    if not quote_span or quote_span not in evidence_text:
                        verdict = "not_enough_info"

                # Status determination
                final_status: Literal["supported", "contradicted", "unsupported", "no_evidence"]
                if has_numeric_mismatch:
                    final_status = "contradicted" if verdict == "contradicted" else "unsupported"
                elif verdict == "supported":
                    final_status = "supported"
                elif verdict == "contradicted":
                    final_status = "contradicted"
                else:  # not_enough_info
                    final_status = "unsupported"

                is_supported = (final_status == "supported")
                contradiction_detected = (final_status == "contradicted")

                notes_parts = []
                if has_numeric_mismatch:
                    notes_parts.append("Numeric discrepancy detected: claim numbers do not match cited evidence.")
                if quote_span and quote_span not in evidence_text:
                    notes_parts.append("Quote span was not found verbatim in evidence (downgraded to not_enough_info).")
                if judge_resp.value.rationale:
                    notes_parts.append(judge_resp.value.rationale)

                claim_obj = ClaimVerification(
                    claim=statement,
                    status=final_status,
                    is_supported=is_supported,
                    evidence_sources=sources,
                    quote_span=quote_span if quote_span in evidence_text else None,
                    numeric_mismatch=has_numeric_mismatch,
                    contradiction_detected=contradiction_detected,
                    notes="; ".join(notes_parts),
                )
                claims_verified.append(claim_obj)
                if not is_supported:
                    unsupported_claims.append(statement)

        # Aggregate Metrics
        total_claims = len(claims_verified)
        duration = round(time.perf_counter() - t0, 3)

        if total_claims == 0:
            audit_notes = "no findings to verify"
            metrics = VerificationMetrics(
                supported=0,
                contradicted=0,
                unsupported=0,
                no_evidence=0,
                supported_ratio=None,
                citation_coverage=None,
                numeric_mismatches=0,
                quality_pass=False,
                evidence_coverage=0.0,
                overall_reliability=0.0,
            )
            return VerificationResult(
                status="needs_revision",
                claims=[],
                metrics=metrics,
                unsupported_claims=[],
                audit_notes=audit_notes,
                verification_time_seconds=duration,
                metadata={"judge_model": judge_model_name},
            )

        supported_count = sum(1 for c in claims_verified if c.status == "supported")
        contradicted_count = sum(1 for c in claims_verified if c.status == "contradicted")
        unsupported_count = sum(1 for c in claims_verified if c.status == "unsupported")
        no_evidence_count = sum(1 for c in claims_verified if c.status == "no_evidence")
        numeric_mismatches = sum(1 for c in claims_verified if c.numeric_mismatch)

        supported_ratio = round(supported_count / total_claims, 4)
        citation_coverage = round(
            sum(1 for c in claims_verified if c.status != "no_evidence") / total_claims, 4
        )

        # Quality pass: 0 contradictions, 0 numeric mismatches, and unsupported <= 20% (supported_ratio >= 0.8)
        quality_pass = (
            contradicted_count == 0
            and numeric_mismatches == 0
            and supported_ratio >= 0.8
        )

        metrics = VerificationMetrics(
            supported=supported_count,
            contradicted=contradicted_count,
            unsupported=unsupported_count,
            no_evidence=no_evidence_count,
            supported_ratio=supported_ratio,
            citation_coverage=citation_coverage,
            numeric_mismatches=numeric_mismatches,
            quality_pass=quality_pass,
            evidence_coverage=citation_coverage,
            overall_reliability=round(supported_ratio * 100.0, 1),
        )

        status_str = "passed" if quality_pass else "needs_revision"

        audit_notes = (
            f"Verification audit: {supported_count}/{total_claims} supported ({supported_ratio*100:.1f}%), "
            f"{contradicted_count} contradictions, {numeric_mismatches} numeric mismatches. Quality pass: {quality_pass}."
        )

        judge_cost = (
            calculate_cost(judge_model_name, total_judge_in_tokens, total_judge_out_tokens)
            if (judge_model_name and total_judge_calls > 0)
            else None
        )

        return VerificationResult(
            status=status_str,
            claims=claims_verified,
            metrics=metrics,
            unsupported_claims=unsupported_claims,
            audit_notes=audit_notes,
            verification_time_seconds=duration,
            usage=Usage(input_tokens=total_judge_in_tokens, output_tokens=total_judge_out_tokens)
            if total_judge_calls > 0
            else None,
            metadata={
                "call_count": total_judge_calls,
                "model": judge_model_name,
                "judge_model": judge_model_name,
                "cost_usd": judge_cost,
            },
        )
