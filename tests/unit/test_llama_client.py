# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_openai():
    mock_sdk = MagicMock()
    mock_sdk.OpenAI = MagicMock()
    sys.modules["openai"] = mock_sdk
    yield mock_sdk
    sys.modules.pop("openai", None)


class TestProviderArgspec:
    def test_base_url_default(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.module_utils.llama_client import (
            PROVIDER_ARGSPEC,
        )

        assert PROVIDER_ARGSPEC["base_url"]["default"] == "http://127.0.0.1:8080/v1"

    def test_api_key_default_and_no_log(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.module_utils.llama_client import (
            PROVIDER_ARGSPEC,
        )

        assert PROVIDER_ARGSPEC["api_key"]["no_log"] is True
        assert PROVIDER_ARGSPEC["api_key"]["default"] == "not-needed"

    def test_timeout_and_max_retries_defaults(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.module_utils.llama_client import (
            PROVIDER_ARGSPEC,
        )

        assert PROVIDER_ARGSPEC["timeout"]["default"] == 120.0
        assert PROVIDER_ARGSPEC["max_retries"]["default"] == 2


class TestGetClient:
    def test_builds_client_with_defaults(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.module_utils.llama_client import (
            get_client,
        )

        module = MagicMock()
        module.params = {
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "not-needed",
            "timeout": 120.0,
            "max_retries": 2,
        }

        get_client(module)
        mock_openai.OpenAI.assert_called_once()
        call_kwargs = mock_openai.OpenAI.call_args.kwargs
        assert call_kwargs["base_url"] == "http://127.0.0.1:8080/v1"
        assert call_kwargs["api_key"] == "not-needed"
        assert call_kwargs["timeout"] == 120.0
        assert call_kwargs["max_retries"] == 2

    def test_custom_base_url_and_api_key(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.module_utils.llama_client import (
            get_client,
        )

        module = MagicMock()
        module.params = {
            "base_url": "http://192.168.1.50:8090/v1",
            "api_key": "some-real-key",
            "timeout": 60.0,
            "max_retries": 5,
        }

        get_client(module)
        call_kwargs = mock_openai.OpenAI.call_args.kwargs
        assert call_kwargs["base_url"] == "http://192.168.1.50:8090/v1"
        assert call_kwargs["api_key"] == "some-real-key"

    def test_missing_timeout_and_max_retries_fall_back_to_hardcoded_defaults(self, mock_openai):
        # Regression coverage: get_client() itself must not blow up if
        # module.params ever lacks these keys (e.g. a caller that built
        # its own params dict without the full PROVIDER_ARGSPEC merge).
        from ansible_collections.aknochow.llama.plugins.module_utils.llama_client import (
            get_client,
        )

        module = MagicMock()
        module.params = {
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "not-needed",
            "timeout": None,
            "max_retries": None,
        }

        get_client(module)
        call_kwargs = mock_openai.OpenAI.call_args.kwargs
        assert call_kwargs["timeout"] == 120.0
        assert call_kwargs["max_retries"] == 2

    def test_missing_sdk_fails_cleanly(self, mock_openai, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("no openai")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from ansible_collections.aknochow.llama.plugins.module_utils.llama_client import (
            get_client,
        )

        module = MagicMock()
        module.params = {
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "not-needed",
            "timeout": 120.0,
            "max_retries": 2,
        }

        result = get_client(module)
        module.fail_json.assert_called_once()
        assert "openai" in module.fail_json.call_args.kwargs["msg"]
        assert result is None
        mock_openai.OpenAI.assert_not_called()
