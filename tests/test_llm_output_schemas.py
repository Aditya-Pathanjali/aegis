"""
Contract tests for LLM-facing structured output schemas.
Validates that every schema passed to structured() in aegis/:
1. Passes Anthropic SDK transform_schema()
2. Passes OpenAI SDK to_strict_json_schema()
3. Contains no Any-typed (typeless) properties
4. Contains no 'format: date-time' properties
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Type

import anthropic.lib._parse._transform as ant_trans
import openai.lib._pydantic as oai_pyd
import pytest
from pydantic import BaseModel

from aegis.models.llm_outputs import ALL_LLM_OUTPUT_SCHEMAS
from aegis.models.schemas import ExecutionPlan


def discover_structured_schemas() -> List[Tuple[str, Type[BaseModel], str]]:
    """
    Scans aegis/ source files using AST to find all schema classes passed to structured().
    Returns list of (schema_name, schema_class, file_path).
    """
    aegis_dir = Path(__file__).resolve().parent.parent / "aegis"
    discovered: Set[Tuple[str, Type[BaseModel], str]] = set()

    for py_file in aegis_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))

        rel_parts = py_file.relative_to(aegis_dir.parent).with_suffix("").parts
        module_name = ".".join(rel_parts)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                is_structured = False
                if isinstance(node.func, ast.Attribute) and node.func.attr == "structured":
                    is_structured = True
                elif isinstance(node.func, ast.Name) and node.func.id == "structured":
                    is_structured = True

                if is_structured:
                    for kw in node.keywords:
                        if kw.arg == "schema":
                            mod = importlib.import_module(module_name)
                            schema_cls = None
                            schema_name = ""
                            if isinstance(kw.value, ast.Name):
                                schema_name = kw.value.id
                                schema_cls = getattr(mod, schema_name, None)
                            elif isinstance(kw.value, ast.Attribute):
                                schema_name = kw.value.attr
                                if isinstance(kw.value.value, ast.Name):
                                    parent_obj = getattr(mod, kw.value.value.id, None)
                                    schema_cls = getattr(parent_obj, schema_name, None) if parent_obj else None

                            if schema_cls and isinstance(schema_cls, type) and issubclass(schema_cls, BaseModel):
                                discovered.add((schema_name, schema_cls, str(py_file)))

    return sorted(list(discovered), key=lambda x: (x[0], x[2]))


def check_no_typeless_properties(schema_dict: Dict[str, Any], path: str = "root") -> None:
    """Recursively asserts that every property in the schema has a concrete type indicator (no Any)."""
    if not isinstance(schema_dict, dict):
        return

    if "properties" in schema_dict and isinstance(schema_dict["properties"], dict):
        for prop_name, prop_schema in schema_dict["properties"].items():
            if isinstance(prop_schema, dict):
                has_type_indicator = any(
                    k in prop_schema for k in ("type", "$ref", "anyOf", "allOf", "oneOf")
                )
                assert has_type_indicator, (
                    f"Typeless (Any) property found at {path}.properties.{prop_name}: {prop_schema}"
                )
                check_no_typeless_properties(prop_schema, f"{path}.properties.{prop_name}")

    for defs_key in ("$defs", "definitions"):
        if defs_key in schema_dict and isinstance(schema_dict[defs_key], dict):
            for def_name, def_schema in schema_dict[defs_key].items():
                if isinstance(def_schema, dict):
                    check_no_typeless_properties(def_schema, f"{path}.{defs_key}.{def_name}")

    if "items" in schema_dict and isinstance(schema_dict["items"], dict):
        check_no_typeless_properties(schema_dict["items"], f"{path}.items")

    for comp_key in ("anyOf", "allOf", "oneOf"):
        if comp_key in schema_dict and isinstance(schema_dict[comp_key], list):
            for i, sub in enumerate(schema_dict[comp_key]):
                if isinstance(sub, dict):
                    check_no_typeless_properties(sub, f"{path}.{comp_key}[{i}]")


def check_no_datetime_format(schema_dict: Dict[str, Any], path: str = "root") -> None:
    """Recursively asserts that no field specifies format: date-time."""
    if not isinstance(schema_dict, dict):
        return

    assert schema_dict.get("format") != "date-time", (
        f"Datetime format found at {path}: format={schema_dict.get('format')}"
    )

    if "properties" in schema_dict and isinstance(schema_dict["properties"], dict):
        for prop_name, prop_schema in schema_dict["properties"].items():
            if isinstance(prop_schema, dict):
                check_no_datetime_format(prop_schema, f"{path}.properties.{prop_name}")

    for defs_key in ("$defs", "definitions"):
        if defs_key in schema_dict and isinstance(schema_dict[defs_key], dict):
            for def_name, def_schema in schema_dict[defs_key].items():
                if isinstance(def_schema, dict):
                    check_no_datetime_format(def_schema, f"{path}.{defs_key}.{def_name}")

    if "items" in schema_dict and isinstance(schema_dict["items"], dict):
        check_no_datetime_format(schema_dict["items"], f"{path}.items")

    for comp_key in ("anyOf", "allOf", "oneOf"):
        if comp_key in schema_dict and isinstance(schema_dict[comp_key], list):
            for i, sub in enumerate(schema_dict[comp_key]):
                if isinstance(sub, dict):
                    check_no_datetime_format(sub, f"{path}.{comp_key}[{i}]")


SCHEMA_CASES = discover_structured_schemas()


@pytest.mark.parametrize(
    "schema_name,schema_cls,file_path",
    SCHEMA_CASES,
    ids=[f"{name}_in_{Path(fp).stem}" for name, _, fp in SCHEMA_CASES],
)
def test_all_discovered_structured_schemas(
    schema_name: str, schema_cls: Type[BaseModel], file_path: str
) -> None:
    """Asserts that each discovered schema complies with SDK requirements and has no runtime fields."""
    # 1. Anthropic SDK transform_schema must succeed
    anthropic_schema = ant_trans.transform_schema(schema_cls)
    assert anthropic_schema is not None, f"Anthropic schema transform returned None for {schema_name}"

    # 2. OpenAI SDK to_strict_json_schema must succeed
    openai_strict_schema = oai_pyd.to_strict_json_schema(schema_cls)
    assert openai_strict_schema is not None, f"OpenAI strict schema returned None for {schema_name}"

    # 3. Model JSON schema must contain no Any-typed (typeless) properties
    base_schema = schema_cls.model_json_schema()
    check_no_typeless_properties(base_schema, path=schema_name)

    # 4. Model JSON schema must contain no date-time formatting
    check_no_datetime_format(base_schema, path=schema_name)


def test_discovered_schemas_match_registry() -> None:
    """Every structured schema found by AST scan in aegis/ must be in ALL_LLM_OUTPUT_SCHEMAS."""
    discovered_classes = {cls for _, cls, _ in SCHEMA_CASES}
    registry_classes = set(ALL_LLM_OUTPUT_SCHEMAS)
    assert discovered_classes == registry_classes, (
        f"Mismatch between discovered schemas and llm_outputs registry: "
        f"discovered={discovered_classes}, registry={registry_classes}"
    )


def test_runtime_execution_plan_fails_schema_contract() -> None:
    """
    Confirms that the runtime ExecutionPlan schema fails the contract (reproducing root cause).
    1. Fails Anthropic transform_schema with ValueError due to typeless ToolCallRecord.output: Any.
    2. Contains 'date-time' formatted created_at.
    3. Contains typeless (Any) properties.
    """
    # 1. Anthropic SDK rejects ExecutionPlan
    with pytest.raises(ValueError, match="Schema must have a 'type', 'anyOf', 'oneOf', or 'allOf' field"):
        ant_trans.transform_schema(ExecutionPlan)

    # 2. Check that check_no_datetime_format fails on ExecutionPlan
    plan_schema = ExecutionPlan.model_json_schema()
    with pytest.raises(AssertionError, match="Datetime format found"):
        check_no_datetime_format(plan_schema)

    # 3. Check that check_no_typeless_properties fails on ExecutionPlan
    with pytest.raises(AssertionError, match="Typeless \\(Any\\) property found"):
        check_no_typeless_properties(plan_schema)

