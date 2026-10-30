"""Provider-neutral types for LLM interactions."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    text: str


class ToolUseBlock(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    raw_arguments: Optional[str] = None
    parse_error: Optional[str] = None


class ToolResultBlock(BaseModel):
    tool_call_id: str
    content: str
    is_error: bool = False


Block = Union[TextBlock, ToolUseBlock, ToolResultBlock]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: List[Block] = Field(default_factory=list)


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


StopReason = Literal["end_turn", "tool_use", "max_tokens", "other"]


class LLMResponse(BaseModel):
    text: Optional[str] = None
    tool_calls: List[ToolUseBlock] = Field(default_factory=list)
    stop_reason: StopReason = "end_turn"
    usage: Usage = Field(default_factory=Usage)
    model: str
    latency_ms: float = 0.0
    cost_usd: Optional[float] = None

    @property
    def content(self) -> List[Block]:
        blocks: List[Block] = []
        if self.text is not None:
            blocks.append(TextBlock(text=self.text))
        blocks.extend(self.tool_calls)
        return blocks


class StructuredResult(BaseModel):
    value: Any
    usage: Usage = Field(default_factory=Usage)
    model: str
    latency_ms: float = 0.0
    attempts: int = 1
    cost_usd: Optional[float] = None

