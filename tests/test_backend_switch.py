"""Backend dispatcher tests — exercises the chroma <-> opensearch routing in
retriever.get_retriever() without standing up either backend.

These tests run in CI without langchain-chroma or opensearch-py being functional,
because they short-circuit before any real client is constructed.
"""
from __future__ import annotations

import sys
import types
from dataclasses import replace
from unittest.mock import MagicMock, patch

from src import retriever as retriever_module
from src.config import CONFIG


def test_default_backend_is_chroma():
    """Env var unset → backend is 'chroma'."""
    assert CONFIG.backend == "chroma"


def test_dispatcher_routes_opensearch_path_when_configured():
    """When CONFIG.backend == 'opensearch', get_retriever() must route to
    src.retriever_opensearch.get_opensearch_retriever, NOT to Chroma."""
    fake_cfg = replace(CONFIG, backend="opensearch")
    fake_module = types.ModuleType("src.retriever_opensearch")
    fake_module.get_opensearch_retriever = MagicMock(return_value="<opensearch-retriever>")
    fake_module.build_opensearch_index = MagicMock()

    with patch.object(retriever_module, "CONFIG", fake_cfg), \
         patch.dict(sys.modules, {"src.retriever_opensearch": fake_module}):
        result = retriever_module.get_retriever()
        fake_module.get_opensearch_retriever.assert_called_once()
        assert result == "<opensearch-retriever>"


def test_dispatcher_does_not_call_opensearch_for_chroma_backend():
    """When CONFIG.backend stays 'chroma', the opensearch helper must NOT be
    called. We patch the opensearch module with a strict mock and the chroma
    imports with stubs; if the dispatcher misroutes, the test will fail loudly."""
    fake_cfg = replace(CONFIG, backend="chroma")
    fake_os_module = types.ModuleType("src.retriever_opensearch")
    fake_os_module.get_opensearch_retriever = MagicMock(
        side_effect=AssertionError("opensearch path should not have been taken for chroma backend")
    )
    fake_os_module.build_opensearch_index = MagicMock()

    fake_chroma_module = types.ModuleType("langchain_chroma")
    fake_chroma_module.Chroma = MagicMock()
    fake_chroma_module.Chroma.return_value.as_retriever.return_value = "<chroma-retriever>"

    fake_openai_module = types.ModuleType("langchain_openai")
    fake_openai_module.OpenAIEmbeddings = MagicMock()

    with patch.object(retriever_module, "CONFIG", fake_cfg), \
         patch.dict(sys.modules, {
             "src.retriever_opensearch": fake_os_module,
             "langchain_chroma": fake_chroma_module,
             "langchain_openai": fake_openai_module,
         }):
        result = retriever_module.get_retriever()
        # Opensearch path NOT called
        fake_os_module.get_opensearch_retriever.assert_not_called()
        # Chroma path WAS called
        assert result == "<chroma-retriever>"
