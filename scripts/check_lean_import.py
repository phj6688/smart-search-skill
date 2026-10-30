"""Fail if importing the MCP sidecar loads an agent or model SDK dependency."""

import sys

import search_workflow.mcp_server  # noqa: F401

forbidden = [
    name for name in sys.modules
    if name.startswith(("langchain", "langgraph", "openai"))
]
assert not forbidden, f"MCP imported agent dependencies: {forbidden}"
print("Lean MCP import passed")
