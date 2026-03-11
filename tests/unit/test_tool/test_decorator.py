"""Tests for agent_harness.tool.decorator — @tool, schema generation, docstring parsing."""
from __future__ import annotations

import pytest

from agent_harness.tool.decorator import (
    FunctionTool,
    _build_schema_from_function,
    _parse_docstring_params,
    _python_type_to_json_schema,
    tool,
)


class TestPythonTypeToJsonSchema:
    def test_str(self) -> None:
        assert _python_type_to_json_schema(str) == {"type": "string"}

    def test_int(self) -> None:
        assert _python_type_to_json_schema(int) == {"type": "integer"}

    def test_float(self) -> None:
        assert _python_type_to_json_schema(float) == {"type": "number"}

    def test_bool(self) -> None:
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list_of_str(self) -> None:
        result = _python_type_to_json_schema(list[str])
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_bare_list(self) -> None:
        result = _python_type_to_json_schema(list)
        assert result == {"type": "array"}

    def test_dict(self) -> None:
        assert _python_type_to_json_schema(dict) == {"type": "object"}

    def test_unknown_type_defaults_to_string(self) -> None:
        class Custom:
            pass
        assert _python_type_to_json_schema(Custom) == {"type": "string"}


class TestParseDocstringParams:
    def test_google_style(self) -> None:
        docstring = """Search for something.

        Args:
            query: The search query
            limit: Max results to return
        """
        params = _parse_docstring_params(docstring)
        assert params["query"] == "The search query"
        assert params["limit"] == "Max results to return"

    def test_with_type_annotations(self) -> None:
        docstring = """Do something.

        Args:
            name (str): The name
            count (int): How many
        """
        params = _parse_docstring_params(docstring)
        assert params["name"] == "The name"
        assert params["count"] == "How many"

    def test_stops_at_returns_section(self) -> None:
        docstring = """Func.

        Args:
            x: First param
        Returns:
            Something
        """
        params = _parse_docstring_params(docstring)
        assert "x" in params
        assert len(params) == 1

    def test_no_docstring(self) -> None:
        assert _parse_docstring_params(None) == {}

    def test_no_args_section(self) -> None:
        assert _parse_docstring_params("Just a description.") == {}


class TestBuildSchemaFromFunction:
    def test_simple_function(self) -> None:
        def search(query: str, limit: int = 10) -> str:
            """Search the web."""
            ...

        schema = _build_schema_from_function(search, None, None)
        assert schema.name == "search"
        assert schema.description == "Search the web."
        assert "query" in schema.parameters["properties"]
        assert schema.parameters["properties"]["query"]["type"] == "string"
        assert "query" in schema.parameters["required"]
        assert "limit" in schema.parameters["properties"]
        assert "limit" not in schema.parameters["required"]
        assert schema.parameters["properties"]["limit"]["default"] == 10

    def test_custom_name_and_description(self) -> None:
        def my_fn(x: int) -> None:
            pass

        schema = _build_schema_from_function(my_fn, "custom_name", "Custom desc")
        assert schema.name == "custom_name"
        assert schema.description == "Custom desc"

    def test_no_docstring_uses_function_name(self) -> None:
        def no_doc(a: str) -> None:
            pass

        schema = _build_schema_from_function(no_doc, None, None)
        assert schema.name == "no_doc"

    def test_skips_self_and_cls(self) -> None:
        def method(self, x: str) -> str:
            """Do thing."""
            ...

        schema = _build_schema_from_function(method, None, None)
        assert "self" not in schema.parameters["properties"]
        assert "x" in schema.parameters["properties"]


class TestToolDecorator:
    def test_decorator_without_args(self) -> None:
        @tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}"

        assert isinstance(greet, FunctionTool)
        assert greet.name == "greet"
        assert greet.description == "Say hello."
        schema = greet.get_schema()
        assert "name" in schema.parameters["properties"]

    def test_decorator_with_args(self) -> None:
        @tool(name="custom_search", description="Custom search tool")
        def search(query: str, limit: int = 5) -> str:
            """Search for stuff.

            Args:
                query: The search query
                limit: Max results
            """
            return f"Results for {query}"

        assert isinstance(search, FunctionTool)
        assert search.name == "custom_search"
        assert search.description == "Custom search tool"
        schema = search.get_schema()
        assert schema.parameters["properties"]["query"]["description"] == "The search query"

    @pytest.mark.asyncio
    async def test_sync_function_execution(self) -> None:
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await add.execute(a=3, b=4)
        assert result == "7"

    @pytest.mark.asyncio
    async def test_async_function_execution(self) -> None:
        @tool
        async def async_greet(name: str) -> str:
            """Greet async."""
            return f"Hi {name}"

        result = await async_greet.execute(name="World")
        assert result == "Hi World"

    @pytest.mark.asyncio
    async def test_none_return_becomes_empty_string(self) -> None:
        @tool
        def noop() -> None:
            """Do nothing."""
            pass

        result = await noop.execute()
        assert result == ""

    def test_schema_parameters_required_list(self) -> None:
        @tool
        def fn(required_param: str, optional_param: int = 42) -> str:
            """Func."""
            return ""

        schema = fn.get_schema()
        assert "required_param" in schema.parameters["required"]
        assert "optional_param" not in schema.parameters["required"]
