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
    assert "api_key" in params
    assert "base_url" in params
    assert "timeout" in params
    assert "max_retries" in params
    assert issubclass(OpenAIError, Exception)
