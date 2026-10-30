"""Generic tool-using specialist agent loop."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, List, Optional, Set

from aegis.config import get_settings
from aegis.llm.pricing import calculate_cost
from aegis.llm.types import (
    Block,
    LLMResponse,
    Message,
    StructuredResult,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from aegis.models.llm_outputs import SpecialistOutput
from aegis.models.schemas import (
    AgentExecutionResult,
    AgentRole,
    EvidenceRef,
    ToolCallRecord,
)

if TYPE_CHECKING:
    from aegis.llm.base import LLMProvider
    from aegis.tools.registry import ToolRegistry


def run_tool_agent(
    llm_provider: LLMProvider,
    system_prompt: str,
    user_prompt: str,
    permitted_tool_names: List[str],
    tool_registry: ToolRegistry,
    role: AgentRole,
    subtask_id: str = "",
    step_limit: Optional[int] = None,
) -> AgentExecutionResult:
    """
    Executes a multi-turn tool interaction loop:
    1. Loops with chat(tools=...) up to step_limit.
    2. Executes only permitted tools, capturing real tool calls.
    3. Feeds back tool results or parse errors as is_error blocks.
    4. Concludes with structured(schema=SpecialistOutput).
    5. Validates and drops fabricated evidence references, noting them in limitations.
    """
    settings = get_settings()
    max_steps = step_limit if step_limit is not None else settings.agent_step_limit

    # Prepare tool specifications
    tool_specs: List[ToolSpec] = []
    for tool_name in permitted_tool_names:
        tool = tool_registry.get_tool(tool_name)
        if tool:
            tool_specs.append(
                ToolSpec(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.parameters,
                )
            )

    messages: List[Message] = [
        Message(role="user", content=[TextBlock(text=user_prompt)])
    ]

    total_in_tokens = 0
    total_out_tokens = 0
    total_calls = 0
    all_tool_records: List[ToolCallRecord] = []
    produced_chunk_ids: Set[str] = set()
    produced_tool_ids: Set[str] = set()

    t0 = time.perf_counter()

    step_limit_hit = False
    # Step limit loop
    for _ in range(max_steps):
        resp: LLMResponse = llm_provider.chat(
            messages=messages,
            tools=tool_specs,
            system=system_prompt,
        )
        total_calls += 1
        total_in_tokens += resp.usage.input_tokens
        total_out_tokens += resp.usage.output_tokens

        messages.append(Message(role="assistant", content=resp.content))

        tool_calls: List[ToolUseBlock] = resp.tool_calls
        if not tool_calls:
            # Model finished tool usage and provided text response
            break

        tool_result_blocks: List[Block] = []
        for tc in tool_calls:
            call_id = tc.id

            # Check for OpenAI parse errors
            if tc.parse_error or tc.name == "parse_error":
                err_msg = f"Tool call parameter parse error: {tc.parse_error or 'invalid arguments'}"
                tool_result_blocks.append(
                    ToolResultBlock(
                        tool_call_id=call_id,
                        content=err_msg,
                        is_error=True,
                    )
                )
                all_tool_records.append(
                    ToolCallRecord(
                        call_id=call_id,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        output=None,
                        execution_time_ms=0.0,
                        success=False,
                        error=err_msg,
                    )
                )
                continue

            # Check tool permission
            if tc.name not in permitted_tool_names:
                err_msg = f"Permission Denied: Agent role '{role.value}' is not authorized to use tool '{tc.name}'."
                tool_result_blocks.append(
                    ToolResultBlock(
                        tool_call_id=call_id,
                        content=err_msg,
                        is_error=True,
                    )
                )
                all_tool_records.append(
                    ToolCallRecord(
                        call_id=call_id,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        output=None,
                        execution_time_ms=0.0,
                        success=False,
                        error=err_msg,
                    )
                )
                continue

            # Execute permitted tool
            rec: ToolCallRecord = tool_registry.execute_tool(
                role=role,
                tool_name=tc.name,
                arguments=tc.arguments,
                call_id=call_id,
            )
            all_tool_records.append(rec)

            if rec.success:
                # Only successful tool calls are eligible evidence references
                produced_tool_ids.add(call_id)
                # Harvest chunk IDs from RAG output
                if isinstance(rec.output, dict):
                    evidence_items = rec.output.get("evidence", [])
                    if isinstance(evidence_items, list):
                        for ev in evidence_items:
                            if isinstance(ev, dict) and "chunk_id" in ev:
                                produced_chunk_ids.add(str(ev["chunk_id"]))

                payload_str = (
                    json.dumps(rec.output)
                    if not isinstance(rec.output, str)
                    else rec.output
                )
                tool_result_blocks.append(
                    ToolResultBlock(
                        tool_call_id=call_id,
                        content=payload_str,
                        is_error=False,
                    )
                )
            else:
                tool_result_blocks.append(
                    ToolResultBlock(
                        tool_call_id=call_id,
                        content=rec.error or "Tool execution error",
                        is_error=True,
                    )
                )

        messages.append(Message(role="user", content=tool_result_blocks))
    else:
        # Loop exhausted all steps without model issuing final answer
        step_limit_hit = True

    # Conclude with structured schema output
    instruction_text = (
        "Please compile your final findings into the structured SpecialistOutput schema. "
        "Ensure each finding cites valid chunk or tool_call evidence references."
    )
    if messages[-1].role == "user":
        messages[-1].content.append(TextBlock(text=instruction_text))
    else:
        messages.append(Message(role="user", content=[TextBlock(text=instruction_text)]))

    struct_res: StructuredResult = llm_provider.structured(
        messages=messages,
        schema=SpecialistOutput,
        system=system_prompt,
    )
    total_calls += 1
    total_in_tokens += struct_res.usage.input_tokens
    total_out_tokens += struct_res.usage.output_tokens

    spec_output: SpecialistOutput = struct_res.value
    valid_limitations = list(spec_output.limitations)
    if step_limit_hit:
        valid_limitations.append(f"Step limit ({max_steps}) reached before tool execution completed.")

    # Validate evidence references: record every unverified or dropped reference into limitations
    for finding in spec_output.findings:
        valid_refs: List[EvidenceRef] = []
        for ref in finding.evidence:
            if ref.kind == "chunk" and ref.ref_id in produced_chunk_ids:
                valid_refs.append(ref)
            elif ref.kind in ("tool_call", "tool_output") and ref.ref_id in produced_tool_ids:
                valid_refs.append(EvidenceRef(kind="tool_call", ref_id=ref.ref_id))
            else:
                valid_limitations.append(
                    f"Dropped unverified evidence reference ({ref.kind}:{ref.ref_id}) for finding: {finding.statement[:80]}"
                )

        finding.evidence = valid_refs

    duration_sec = time.perf_counter() - t0
    cost_usd = calculate_cost(struct_res.model, total_in_tokens, total_out_tokens) if struct_res.model else None

    return AgentExecutionResult(
        subtask_id=subtask_id,
        agent_role=role,
        output_summary=spec_output.summary,
        findings=spec_output.findings,
        limitations=valid_limitations,
        detailed_findings=[f.statement for f in spec_output.findings],
        tool_calls=all_tool_records,
        execution_time_seconds=round(duration_sec, 3),
        usage=Usage(input_tokens=total_in_tokens, output_tokens=total_out_tokens),
        metadata={
            "attempts": struct_res.attempts,
            "model": struct_res.model,
            "cost_usd": cost_usd,
            "call_count": total_calls,
        },
    )
