# SPDX-License-Identifier: Apache-2.0

"""Lock the openai SDK surface aknochow.openai actually calls."""

from __future__ import annotations

import inspect
from importlib.metadata import version

from packaging.version import Version


def test_installed_sdk_meets_the_collection_floor():
    assert Version(version("openai")) >= Version("1.58.0")


def test_client_accepts_base_url_and_api_key():
    from openai import OpenAI, OpenAIError

    params = inspect.signature(OpenAI).parameters
    for name in ("api_key", "base_url", "timeout", "max_retries"):
        assert name in params
        assert params[name].kind is not inspect.Parameter.POSITIONAL_ONLY
    OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        timeout=1.0,
        max_retries=1,
    )
    assert issubclass(OpenAIError, Exception)
