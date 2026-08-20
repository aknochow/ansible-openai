# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from urllib.parse import urlparse

from ansible.module_utils.basic import AnsibleModule, env_fallback

PROVIDER_ARGSPEC = dict(
    base_url=dict(
        type="str",
        default="http://127.0.0.1:8080/v1",
        fallback=(env_fallback, ["ANSIBLE_LLAMA_BASE_URL"]),
    ),
    api_key=dict(
        type="str",
        no_log=True,
        default="not-needed",
        fallback=(env_fallback, ["ANSIBLE_LLAMA_API_KEY"]),
    ),
    timeout=dict(type="float", default=120.0),
    max_retries=dict(type="int", default=2),
)


def get_client(module: AnsibleModule):
    """Construct an openai SDK client pointed at a self-hosted llama-server instance."""
    try:
        from openai import OpenAI
    except ImportError:
        module.fail_json(
            msg="The openai Python SDK is required. Install it with: pip install openai"
        )
        return

    base_url = module.params["base_url"]
    scheme = urlparse(base_url).scheme
    if scheme not in ("http", "https"):
        module.fail_json(msg=f"base_url must use http or https, got: {base_url!r}")
        return

    timeout = module.params.get("timeout") or 120.0
    max_retries = module.params.get("max_retries")
    if max_retries is None:
        max_retries = 2

    return OpenAI(
        base_url=base_url,
        api_key=module.params["api_key"],
        timeout=timeout,
        max_retries=max_retries,
    )
