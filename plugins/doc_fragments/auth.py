# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class ModuleDocFragment:

    DOCUMENTATION = r"""
options:
  base_url:
    description:
      - Base URL of the self-hosted llama-server instance (its OpenAI-compatible endpoint).
      - If the value is not specified, the value of the E(ANSIBLE_OPENAI_BASE_URL) environment variable will be used.
    type: str
    default: http://127.0.0.1:8080/v1
  api_key:
    description:
      - API key sent to the server. llama-server does not validate this by default -- the
        openai SDK requires a non-empty string regardless, so a dummy placeholder is used
        unless overridden.
      - If the value is not specified, the value of the E(ANSIBLE_OPENAI_API_KEY) environment variable will be used.
    type: str
    default: not-needed
  timeout:
    description:
      - Per-request timeout in seconds.
    type: float
    default: 120.0
  max_retries:
    description:
      - Maximum number of automatic retries the SDK performs on transient errors.
    type: int
    default: 2
"""
