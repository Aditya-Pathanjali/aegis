"""OpenAI LLM provider adapter using the official OpenAI Python SDK."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

import openai
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


class OpenAIProvider:
    """Adapter for OpenAI models conforming to the LLMProvider Protocol."""

    def __init__(
        self,
        model: str,
        api_key: str,
        settings: Optional[Settings] = None,
        client: Optional[openai.OpenAI] = None,
    ) -> None:
        self.model = model
        self.settings = settings or get_settings()
        self.client = client or openai.OpenAI(
            api_key=api_key,
            max_retries=self.settings.llm_max_retries,
            timeout=self.settings.llm_timeout_s,
        )

    def _normalize_stop_reason(self, reason: Optional[str]) -> StopReason:
        if reason == "stop":
            return "end_turn"
        if reason == "tool_calls":
            return "tool_use"
        if reason == "length":
            return "max_tokens"
        return "other"

    def _map_tool(self, tool: ToolSpec) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    def _map_messages(self, messages: List[Message], system: Optional[str] = None) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []
        if system:
            mapped.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "user":
                # Split text blocks and tool result blocks
                text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]

                # Add tool results as role='tool'
                for tr in tool_results:
                    mapped.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": tr.content,
                    })

                # Add text content if present
                if text_parts:
                    mapped.append({"role": "user", "content": "\n".join(text_parts)})
            elif msg.role == "assistant":
                text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                tool_uses = [b for b in msg.content if isinstance(b, ToolUseBlock)]

                content = "\n".join(text_parts) if text_parts else None
                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                if content is not None:
                    assistant_msg["content"] = content

                if tool_uses:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tu.id,
                            "type": "function",
                            "function": {
                                "name": tu.name,
                                "arguments": (
                                    tu.raw_arguments
                                    if tu.raw_arguments is not None
                                    else (
                                        json.dumps(tu.arguments)
                                        if isinstance(tu.arguments, dict)
                                        else str(tu.arguments)
                                    )
                                ),
                            },
                        }
                        for tu in tool_uses
                    ]
                mapped.append(assistant_msg)
        return mapped

    def chat(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Any] = None,
    ) -> LLMResponse:
        """Execute a conversational completion with optional tool calls."""
        oai_messages = self._map_messages(messages, system=system)
        effective_max_tokens = max_tokens if max_tokens is not None else self.settings.llm_max_output_tokens

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": effective_max_tokens,
        }
        if tools:
            kwargs["tools"] = [self._map_tool(t) for t in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.APIStatusError as e:
            if e.status_code in (400, 401, 403, 404):
                raise LLMRequestError(provider="openai", status_code=e.status_code, message=str(e)) from e
            raise

        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw_model = getattr(resp, "model", None)
        returned_model = raw_model if isinstance(raw_model, str) else self.model
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        text = choice.message.content if choice and choice.message else None
        stop_reason = self._normalize_stop_reason(choice.finish_reason if choice else None)

        tool_calls: List[ToolUseBlock] = []
        if choice and choice.message and getattr(choice.message, "tool_calls", None):
            for tc in choice.message.tool_calls:
                raw_args = tc.function.arguments if tc.function else "{}"
                parse_err = None
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(args, dict):
                        args = {}
                except Exception as e:
                    args = {}
                    parse_err = str(e)
                tool_calls.append(
                    ToolUseBlock(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                        raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                        parse_error=parse_err,
                    )
                )

        usage_obj = getattr(resp, "usage", None)
        in_tokens = getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0
        out_tokens = getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0
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
        oai_messages = self._map_messages(messages, system=system)
        effective_max_tokens = max_tokens if max_tokens is not None else self.settings.llm_max_output_tokens

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "response_format": schema,
        }
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens

        t0 = time.perf_counter()
        validation_error_str = ""
        raw_text_1 = ""
        usage_1 = Usage()
        returned_model_1 = self.model
        cost_1 = None
        lat_1 = 0.0

        try:
            resp_1 = self.client.chat.completions.parse(**kwargs)
            lat_1 = (time.perf_counter() - t0) * 1000.0
            raw_model_1 = getattr(resp_1, "model", None)
            returned_model_1 = raw_model_1 if isinstance(raw_model_1, str) else self.model
            usage_obj_1 = getattr(resp_1, "usage", None)
            in_1 = getattr(usage_obj_1, "prompt_tokens", 0) if usage_obj_1 else 0
            out_1 = getattr(usage_obj_1, "completion_tokens", 0) if usage_obj_1 else 0
            usage_1 = Usage(input_tokens=in_1, output_tokens=out_1)
            cost_1 = calculate_cost(returned_model_1, in_1, out_1, self.settings.pricing_file)

            choice_1 = resp_1.choices[0] if getattr(resp_1, "choices", None) else None
            msg_1 = choice_1.message if choice_1 else None
            if msg_1 and getattr(msg_1, "parsed", None) is not None:
                return StructuredResult(
                    value=msg_1.parsed,
                    usage=usage_1,
                    model=returned_model_1,
                    latency_ms=round(lat_1, 2),
                    attempts=1,
                    cost_usd=cost_1,
                )

            refusal = getattr(msg_1, "refusal", None) if msg_1 else None
            validation_error_str = refusal or "No parsed output returned by model."
            raw_text_1 = getattr(msg_1, "content", "") or ""
        except openai.APIStatusError as e:
            if e.status_code in (400, 401, 403, 404):
                raise LLMRequestError(provider="openai", status_code=e.status_code, message=str(e)) from e
            raise
        except ValidationError as e:
            lat_1 = (time.perf_counter() - t0) * 1000.0
            validation_error_str = str(e)
            raw_text_1 = getattr(e, "raw_text", str(e))

        # Attempt 2: One repair turn
        repair_messages = list(messages)
        error_note = (
            f"Schema validation error: {validation_error_str}. "
            f"Please correct the output and return valid JSON conforming to the schema."
        )
        if raw_text_1:
            repair_messages.append(Message(role="assistant", content=[TextBlock(text=raw_text_1)]))
            repair_messages.append(Message(role="user", content=[TextBlock(text=error_note)]))
        else:
            if repair_messages and repair_messages[-1].role == "user":
                last_msg = repair_messages[-1]
                new_content = list(last_msg.content) + [TextBlock(text=f"\n\n{error_note}")]
                repair_messages[-1] = Message(role="user", content=new_content)
            else:
                repair_messages.append(Message(role="user", content=[TextBlock(text=error_note)]))

        kwargs["messages"] = self._map_messages(repair_messages, system=system)

        t1 = time.perf_counter()
        try:
            resp_2 = self.client.chat.completions.parse(**kwargs)
            lat_2 = (time.perf_counter() - t1) * 1000.0
            raw_model_2 = getattr(resp_2, "model", None)
            returned_model_2 = raw_model_2 if isinstance(raw_model_2, str) else self.model
            usage_obj_2 = getattr(resp_2, "usage", None)
            in_2 = getattr(usage_obj_2, "prompt_tokens", 0) if usage_obj_2 else 0
            out_2 = getattr(usage_obj_2, "completion_tokens", 0) if usage_obj_2 else 0
            usage_2 = Usage(input_tokens=in_2, output_tokens=out_2)
            cost_2 = calculate_cost(returned_model_2, in_2, out_2, self.settings.pricing_file)

            total_usage = Usage(
                input_tokens=usage_1.input_tokens + usage_2.input_tokens,
                output_tokens=usage_1.output_tokens + usage_2.output_tokens,
            )
            total_latency = round(lat_1 + lat_2, 2)
            total_cost: Optional[float] = None
            if cost_1 is not None and cost_2 is not None:
                total_cost = round(cost_1 + cost_2, 6)

            choice_2 = resp_2.choices[0] if getattr(resp_2, "choices", None) else None
            msg_2 = choice_2.message if choice_2 else None
            if msg_2 and getattr(msg_2, "parsed", None) is not None:
                return StructuredResult(
                    value=msg_2.parsed,
                    usage=total_usage,
                    model=returned_model_2,
                    latency_ms=total_latency,
                    attempts=2,
                    cost_usd=total_cost,
                )

            raw_text_2 = getattr(msg_2, "content", "") if msg_2 else ""
            raise StructuredOutputError(
                "Structured output validation failed after repair turn: no parsed output returned.",
                raw_text=raw_text_2 or raw_text_1,
            )
        except openai.APIStatusError as e:
            if e.status_code in (400, 401, 403, 404):
                raise LLMRequestError(provider="openai", status_code=e.status_code, message=str(e)) from e
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
        """Generate schema-constrained structured output using forced tool call with one repair retry."""
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
        tool_choice = {"type": "function", "function": {"name": tool_name}}

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
        raw_text_1 = (
            matching_call.raw_arguments
            if (matching_call and matching_call.raw_arguments is not None)
            else (json.dumps(raw_args_1) if isinstance(raw_args_1, dict) else str(raw_args_1))
        )

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
        raw_text_2 = (
            matching_call_2.raw_arguments
            if (matching_call_2 and matching_call_2.raw_arguments is not None)
            else (json.dumps(raw_args_2) if isinstance(raw_args_2, dict) else str(raw_args_2))
        )

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
