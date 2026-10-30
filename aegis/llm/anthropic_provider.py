"""Anthropic LLM provider adapter using the official Anthropic Python SDK."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from aegis.config import Settings, get_settings
from aegis.llm.errors import LLMRequestError, StructuredOutputError
from aegis.llm.pricing import calculate_cost
from aegis.llm.types import (
    LLMResponse,
    Message,
    StopReason,
    StructuredResult,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider:
    """Adapter for Anthropic Claude models conforming to the LLMProvider Protocol."""

    def __init__(
        self,
        model: str,
        api_key: str,
        settings: Optional[Settings] = None,
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        self.model = model
        self.settings = settings or get_settings()
        self.client = client or anthropic.Anthropic(
            api_key=api_key,
            max_retries=self.settings.llm_max_retries,
            timeout=self.settings.llm_timeout_s,
        )

    def _normalize_stop_reason(self, reason: Optional[str]) -> StopReason:
        if reason == "end_turn":
            return "end_turn"
        if reason == "tool_use":
            return "tool_use"
        if reason == "max_tokens":
            return "max_tokens"
        return "other"

    def _map_tool(self, tool: ToolSpec) -> Dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

    def _map_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []
        for msg in messages:
            content_blocks: List[Dict[str, Any]] = []
            for b in msg.content:
                if isinstance(b, TextBlock):
                    content_blocks.append({"type": "text", "text": b.text})
                elif isinstance(b, ToolUseBlock):
                    content_blocks.append({
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": b.arguments,
                    })
                elif isinstance(b, ToolResultBlock):
                    content_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": b.tool_call_id,
                        "content": b.content,
                        "is_error": b.is_error,
                    })
            mapped.append({"role": msg.role, "content": content_blocks})
        return mapped

    def chat(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Execute a conversational completion with optional tool calls."""
        anthropic_messages = self._map_messages(messages)
        effective_max_tokens = max_tokens if max_tokens is not None else self.settings.llm_max_output_tokens

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": effective_max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [self._map_tool(t) for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            if e.status_code in (400, 401, 403, 404):
                raise LLMRequestError(provider="anthropic", status_code=e.status_code, message=str(e)) from e
            raise

        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw_model = getattr(resp, "model", None)
        returned_model = raw_model if isinstance(raw_model, str) else self.model

        # Extract content blocks
        text_parts: List[str] = []
        tool_calls: List[ToolUseBlock] = []

        for block in getattr(resp, "content", []):
            b_type = getattr(block, "type", None)
            if b_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif b_type == "tool_use":
                tool_calls.append(
                    ToolUseBlock(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=getattr(block, "input", {}) or {},
                    )
                )

        text = "\n".join(text_parts) if text_parts else None
        stop_reason = self._normalize_stop_reason(getattr(resp, "stop_reason", None))

        usage_obj = getattr(resp, "usage", None)
        in_tokens = getattr(usage_obj, "input_tokens", 0) if usage_obj else 0
        out_tokens = getattr(usage_obj, "output_tokens", 0) if usage_obj else 0
        usage = Usage(input_tokens=in_tokens, output_tokens=out_tokens)

        cost_usd = calculate_cost(returned_model, in_tokens, out_tokens, self.settings.pricing_file)

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=returned_model,
            latency_ms=round(latency_ms, 2),
            cost_usd=cost_usd,
        )

    def structured(
        self,
        messages: List[Message],
        schema: Type[T],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> StructuredResult:
        """Generate schema-constrained structured output using native parse or forced-tool fallback."""
        if getattr(self.settings, "structured_mode", "native") == "tool":
            return self._structured_tool(messages=messages, schema=schema, system=system, max_tokens=max_tokens)
        return self._structured_native(messages=messages, schema=schema, system=system, max_tokens=max_tokens)

    def _structured_native(
        self,
        messages: List[Message],
        schema: Type[T],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> StructuredResult:
        anthropic_messages = self._map_messages(messages)
        effective_max_tokens = max_tokens if max_tokens is not None else self.settings.llm_max_output_tokens

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": effective_max_tokens,
            "output_format": schema,
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        validation_error_str = ""
        raw_text_1 = ""
        usage_1 = Usage()
        returned_model_1 = self.model
        cost_1 = None
        lat_1 = 0.0

        try:
            resp_1 = self.client.messages.parse(**kwargs)
            lat_1 = (time.perf_counter() - t0) * 1000.0
            raw_model_1 = getattr(resp_1, "model", None)
            returned_model_1 = raw_model_1 if isinstance(raw_model_1, str) else self.model
            usage_obj_1 = getattr(resp_1, "usage", None)
            in_1 = getattr(usage_obj_1, "input_tokens", 0) if usage_obj_1 else 0
            out_1 = getattr(usage_obj_1, "output_tokens", 0) if usage_obj_1 else 0
            usage_1 = Usage(input_tokens=in_1, output_tokens=out_1)
            cost_1 = calculate_cost(returned_model_1, in_1, out_1, self.settings.pricing_file)

            parsed_val = None
            for block in getattr(resp_1, "content", []):
                if hasattr(block, "parsed_output") and block.parsed_output is not None:
                    parsed_val = block.parsed_output
                    break
                if hasattr(block, "text") and block.text:
                    raw_text_1 = block.text

            if parsed_val is not None:
                return StructuredResult(
                    value=parsed_val,
                    usage=usage_1,
                    model=returned_model_1,
                    latency_ms=round(lat_1, 2),
                    attempts=1,
                    cost_usd=cost_1,
                )
            validation_error_str = "No parsed output returned by model."
        except anthropic.APIStatusError as e:
            if e.status_code in (400, 401, 403, 404):
                raise LLMRequestError(provider="anthropic", status_code=e.status_code, message=str(e)) from e
            raise
        except ValidationError as e:
            lat_1 = (time.perf_counter() - t0) * 1000.0
            validation_error_str = str(e)
            raw_text_1 = getattr(e, "raw_text", str(e))

        # Attempt 2: One repair turn
        repair_messages = list(messages)
        error_note = f"Schema validation error: {validation_error_str}. Please correct the output and return valid JSON conforming to the schema."
        if repair_messages and repair_messages[-1].role == "user":
            last_msg = repair_messages[-1]
            new_content = list(last_msg.content) + [TextBlock(text=f"\n\n{error_note}")]
            repair_messages[-1] = Message(role="user", content=new_content)
        else:
            repair_messages.append(Message(role="user", content=[TextBlock(text=error_note)]))

        kwargs["messages"] = self._map_messages(repair_messages)

        t1 = time.perf_counter()
        try:
            resp_2 = self.client.messages.parse(**kwargs)
            lat_2 = (time.perf_counter() - t1) * 1000.0
            raw_model_2 = getattr(resp_2, "model", None)
            returned_model_2 = raw_model_2 if isinstance(raw_model_2, str) else self.model
            usage_obj_2 = getattr(resp_2, "usage", None)
            in_2 = getattr(usage_obj_2, "input_tokens", 0) if usage_obj_2 else 0
            out_2 = getattr(usage_obj_2, "output_tokens", 0) if usage_obj_2 else 0
            usage_2 = Usage(input_tokens=in_2, output_tokens=out_2)
            cost_2 = calculate_cost(returned_model_2, in_2, out_2, self.settings.pricing_file)

            total_usage = Usage(
                input_tokens=usage_1.input_tokens + usage_2.input_tokens,
                output_tokens=usage_1.output_tokens + usage_2.output_tokens,
            )
            total_latency = round(lat_1 + lat_2, 2)
            # If any attempt is unpriced, total cost is None
            total_cost: Optional[float] = None
            if cost_1 is not None and cost_2 is not None:
                total_cost = round(cost_1 + cost_2, 6)

            parsed_val_2 = None
            raw_text_2 = ""
            for block in getattr(resp_2, "content", []):
                if hasattr(block, "parsed_output") and block.parsed_output is not None:
                    parsed_val_2 = block.parsed_output
                    break
                if hasattr(block, "text") and block.text:
                    raw_text_2 = block.text

            if parsed_val_2 is not None:
                return StructuredResult(
                    value=parsed_val_2,
                    usage=total_usage,
                    model=returned_model_2,
                    latency_ms=total_latency,
                    attempts=2,
                    cost_usd=total_cost,
                )
            raise StructuredOutputError(
                "Structured output validation failed after repair turn: no parsed output returned.",
                raw_text=raw_text_2 or raw_text_1,
            )
        except anthropic.APIStatusError as e:
            if e.status_code in (400, 401, 403, 404):
                raise LLMRequestError(provider="anthropic", status_code=e.status_code, message=str(e)) from e
            raise
        except ValidationError as err:
            raise StructuredOutputError(
                f"Structured output validation failed after repair turn: {err}",
                raw_text=getattr(err, "raw_text", str(err)),
            ) from err

    def _structured_tool(
        self,
        messages: List[Message],
        schema: Type[T],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> StructuredResult:
        tool_name = schema.__name__ if hasattr(schema, "__name__") else "StructuredOutput"
        input_schema = (
            schema.model_json_schema()
            if hasattr(schema, "model_json_schema")
            else getattr(schema, "schema", lambda: {})()
        )
        tool_spec = ToolSpec(
            name=tool_name,
            description=f"Produce output strictly conforming to the {tool_name} schema.",
            input_schema=input_schema,
        )
        tool_choice = {"type": "tool", "name": tool_name}

        # Attempt 1
        resp_1 = self.chat(
            messages=messages,
            system=system,
            tools=[tool_spec],
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        )

        matching_call = next((tc for tc in resp_1.tool_calls if tc.name == tool_name), None)
        raw_args_1 = matching_call.arguments if matching_call else {}
        raw_text_1 = json.dumps(raw_args_1) if isinstance(raw_args_1, dict) else str(raw_args_1)

        if matching_call is not None:
            try:
                val = schema.model_validate(raw_args_1)
                return StructuredResult(
                    value=val,
                    usage=resp_1.usage,
                    model=resp_1.model,
                    latency_ms=resp_1.latency_ms,
                    attempts=1,
                    cost_usd=resp_1.cost_usd,
                )
            except ValidationError as val_err:
                validation_error_str = str(val_err)
        else:
            validation_error_str = f"Model did not invoke tool '{tool_name}'."

        # Attempt 2: One repair turn
        repair_messages = list(messages)
        if matching_call:
            repair_messages.append(
                Message(
                    role="assistant",
                    content=[matching_call],
                )
            )
            repair_messages.append(
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_call_id=matching_call.id,
                            content=f"Schema validation error: {validation_error_str}. Please correct the data and output valid JSON conforming to {tool_name}.",
                            is_error=True,
                        )
                    ],
                )
            )
        else:
            # If no tool call, append error to last user message rather than creating consecutive user messages
            err_msg = f"Error: {validation_error_str}. You must call the tool '{tool_name}' with valid arguments."
            if repair_messages and repair_messages[-1].role == "user":
                last_msg = repair_messages[-1]
                new_content = list(last_msg.content) + [TextBlock(text=f"\n\n{err_msg}")]
                repair_messages[-1] = Message(role="user", content=new_content)
            else:
                repair_messages.append(Message(role="user", content=[TextBlock(text=err_msg)]))

        resp_2 = self.chat(
            messages=repair_messages,
            system=system,
            tools=[tool_spec],
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        )

        total_usage = Usage(
            input_tokens=resp_1.usage.input_tokens + resp_2.usage.input_tokens,
            output_tokens=resp_1.usage.output_tokens + resp_2.usage.output_tokens,
        )
        total_latency = round(resp_1.latency_ms + resp_2.latency_ms, 2)
        total_cost: Optional[float] = None
        if resp_1.cost_usd is not None and resp_2.cost_usd is not None:
            total_cost = round(resp_1.cost_usd + resp_2.cost_usd, 6)

        matching_call_2 = next((tc for tc in resp_2.tool_calls if tc.name == tool_name), None)
        if matching_call_2 is None:
            raw_text_2 = resp_2.text or raw_text_1
            raise StructuredOutputError(
                f"Model failed to call '{tool_name}' after repair turn.",
                raw_text=raw_text_2,
            )

        raw_args_2 = matching_call_2.arguments
        raw_text_2 = json.dumps(raw_args_2) if isinstance(raw_args_2, dict) else str(raw_args_2)

        try:
            val_2 = schema.model_validate(raw_args_2)
            return StructuredResult(
                value=val_2,
                usage=total_usage,
                model=resp_2.model,
                latency_ms=total_latency,
                attempts=2,
                cost_usd=total_cost,
            )
        except ValidationError as err:
            raise StructuredOutputError(
                f"Structured output validation failed after repair turn: {err}",
                raw_text=raw_text_2,
            ) from err
