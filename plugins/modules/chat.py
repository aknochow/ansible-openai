#!/usr/bin/python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

DOCUMENTATION = r"""
---
module: chat
short_description: Send a chat completion request to a self-hosted llama-server and return the response
description:
  - Calls a self-hosted llama.cpp llama-server instance's OpenAI-compatible /v1/chat/completions
    endpoint directly via the official openai Python SDK.
  - Returns both the raw response and flattened convenience fields for use with O(register).
version_added: "0.1.0"
author:
  - Adam Knochowski (@aknochow)
options:
  model:
    description:
      - Model identifier as reported by the running llama-server (e.g. V(qwen3-8b)). llama-server
        serves whatever single GGUF it was started with, so this value is mostly informational --
        the actual model in the response reflects the loaded GGUF's path, not this string.
    type: str
    required: true
  messages:
    description:
      - List of message objects, each with C(role) (V(system), V(user), or V(assistant)) and C(content).
      - There is no separate system-prompt parameter -- send a C(role=system) message in this list,
        matching the actual OpenAI/llama.cpp wire protocol.
    type: list
    elements: dict
    required: true
  max_tokens:
    description:
      - Maximum number of tokens to generate. There is no default -- an explicit
        budget must be chosen for every call.
      - When O(enable_thinking=true), this budget is shared between the model's reasoning trace
        and its final answer -- see O(enable_thinking)'s own warning.
    type: int
    required: true
  temperature:
    description:
      - Sampling temperature.
    type: float
  top_p:
    description:
      - Nucleus sampling parameter.
    type: float
  top_k:
    description:
      - Top-k sampling parameter. Not part of the official OpenAI API -- a llama.cpp server
        extension, sent via the request body's extra fields.
    type: int
  stop_sequences:
    description:
      - List of strings that stop generation when encountered.
    type: list
    elements: str
  response_format:
    description:
      - 'Native structured-output configuration, e.g. C({"type": "json_schema", "json_schema":
        {"name": ..., "schema": {...}, "strict": true}}).'
      - llama-server enforces this via grammar-constrained decoding -- the model cannot sample a
        token that violates the schema, a stronger guarantee than prompt-based instruction-following.
      - When set, the response text is parsed as JSON into the RV(structured) return value.
    type: dict
  tools:
    description:
      - 'List of tool definitions in OpenAI function-calling format, e.g. C([{"type": "function",
        "function": {"name": ..., "description": ..., "parameters": {...}}}]).'
      - Passed through verbatim to the SDK's own native shape -- not reshaped.
      - Function calls the model makes are returned in the RV(tool_calls) return value.
    type: list
    elements: dict
  tool_choice:
    description:
      - 'Controls tool-use behavior. Either a string (V(auto), V(none), V(required)) or a dict,
        e.g. C({"type": "function", "function": {"name": "..."}}) to force a specific tool.'
    type: raw
  enable_thinking:
    description:
      - Whether to allow the model's internal reasoning/thinking trace before its final answer.
      - Defaults to V(false) so O(max_tokens) is a deterministic budget for the visible answer
        only -- thinking-capable models (e.g. Qwen3) spend part of O(max_tokens) on reasoning
        never returned in RV(text) when this is true, and can exhaust the whole budget on
        reasoning before producing any answer at all (RV(finish_reason)=C(length) with empty
        RV(text)). Raise O(max_tokens) substantially if you enable this.
      - If left unset here but also set inside O(extra_body)'s C(chat_template_kwargs.enable_thinking),
        that explicit value is honored instead of the V(false) default -- this parameter only
        overrides O(extra_body) when it is itself explicitly set.
    type: bool
  extra_body:
    description:
      - Arbitrary additional fields merged into the request body -- an escape hatch for
        llama.cpp-specific sampling extensions (e.g. C(min_p), C(repeat_penalty)) not exposed
        as dedicated parameters above.
    type: dict
extends_documentation_fragment:
  - aknochow.llama.auth
requirements:
  - "openai"
"""

EXAMPLES = r"""
- name: Basic chat completion
  aknochow.llama.chat:
    model: qwen3-8b
    max_tokens: 512
    messages:
      - role: user
        content: "Summarize this changelog in one sentence: {{ changelog }}"
  register: result

- name: Structured extraction with response_format
  aknochow.llama.chat:
    model: qwen3-8b
    max_tokens: 1024
    messages:
      - role: user
        content: "Extract the name and severity from this bug report: {{ bug_text }}"
    response_format:
      type: json_schema
      json_schema:
        name: bug_report
        strict: true
        schema:
          type: object
          properties:
            name: {type: string}
            severity: {type: string, enum: [low, medium, high, critical]}
          required: [name, severity]
          additionalProperties: false
  register: result

- name: Use structured result directly
  ansible.builtin.debug:
    msg: "{{ result.structured.name }} is {{ result.structured.severity }}"

- name: Allow the model to think before answering (needs a much larger max_tokens)
  aknochow.llama.chat:
    model: qwen3-8b
    max_tokens: 4096
    enable_thinking: true
    messages:
      - role: user
        content: "What is 12*13? Show your reasoning."
  register: result

- name: Force a tool call
  aknochow.llama.chat:
    model: qwen3-8b
    max_tokens: 256
    messages:
      - role: user
        content: "What's the weather in Boston?"
    tools:
      - type: function
        function:
          name: get_weather
          description: Get the current weather for a location
          parameters:
            type: object
            properties:
              city: {type: string}
            required: [city]
    tool_choice: required
  register: result

- name: Act on the model's tool call
  ansible.builtin.debug:
    msg: "Model wants to call {{ item.name }} with {{ item.args }}"
  loop: "{{ result.tool_calls }}"
"""

RETURN = r"""
response:
  description: Full raw response from the chat completions endpoint.
  type: dict
  returned: always
text:
  description: The final answer text (excludes any reasoning trace).
  type: str
  returned: always
tool_calls:
  description: List of tool calls the model made, each with C(id), C(name), and C(args).
  type: list
  returned: always
reasoning:
  description: The model's internal reasoning trace, when O(enable_thinking) produced one.
  type: str
  returned: when a reasoning trace was generated
structured:
  description: Parsed JSON object when O(response_format) requested structured output.
  type: dict
  returned: when response_format is set
finish_reason:
  description: Why generation stopped (V(stop), V(length), or others).
  type: str
  returned: always
usage:
  description: Token usage for the request.
  type: dict
  returned: always
  contains:
    prompt_tokens:
      description: Number of input tokens billed for this request.
      type: int
    completion_tokens:
      description: Number of output tokens generated, including any reasoning trace.
      type: int
    total_tokens:
      description: Total tokens billed for this request.
      type: int
"""

import json

from ansible.module_utils.basic import AnsibleModule

from ansible_collections.aknochow.llama.plugins.module_utils.llama_client import (
    PROVIDER_ARGSPEC,
    get_client,
)


def flatten_tool_calls(message):
    tool_calls = []
    for tc in message.tool_calls or []:
        # OpenAI's wire shape returns arguments as a JSON-encoded STRING,
        # not a parsed dict (unlike aknochow.claude's already-parsed
        # `input` and aknochow.gemini's already-parsed `args`) -- confirmed
        # live in design-module-interface-mirroring-siblings. Parse it
        # ourselves so `args` converges in *type*, not just name, with the
        # two existing siblings. Malformed JSON should not happen under
        # llama-server's grammar-constrained tool-call decoding, but fall
        # back to the raw string rather than crashing the whole call if it
        # ever does.
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, ValueError):
            args = tc.function.arguments
        tool_calls.append(dict(id=tc.id, name=tc.function.name, args=args))
    return tool_calls


def flatten_response(response, parse_structured=False):
    if not response.choices:
        raise ValueError("llama-server returned an empty choices list")
    choice = response.choices[0]
    if choice.message is None:
        raise ValueError("llama-server returned a choice with no message")
    text = choice.message.content or ""
    # llama-server uses reasoning_content; Ollama's OpenAI-compat endpoint
    # uses reasoning for the same concept (confirmed live -- Ollama silently
    # drops the trace under the other name, this module must check both or
    # lose it with no error, same as the field it's actually keyed by).
    # `is not None`, not `or` -- an `or` chain would treat an empty-string
    # reasoning_content as absent and fall through to Ollama's field,
    # violating the intended reasoning_content-wins precedence.
    reasoning = getattr(choice.message, "reasoning_content", None)
    if reasoning is None:
        reasoning = getattr(choice.message, "reasoning", None)
    tool_calls = flatten_tool_calls(choice.message)

    structured = None
    # Only attempt the parse when the caller actually requested structured
    # output -- otherwise plain conversational text that merely happens to
    # parse as JSON (e.g. a reply of "42") would silently gain a
    # `structured` key, contradicting the RETURN doc's own claim that it's
    # only present "when response_format is set".
    if parse_structured and text:
        try:
            structured = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            structured = None

    # Some OpenAI-compatible servers omit usage entirely under certain
    # request shapes (llama-server always includes it under normal
    # non-streaming use, but this module doesn't gate the extra_body
    # escape hatch, so a caller-injected field could change that). Keep
    # `usage` itself always a dict -- matches the RETURN doc's "returned:
    # always" -- with 0 sub-fields rather than crashing. 0, not None,
    # to match the RETURN doc's declared `type: int` and avoid breaking
    # numeric consumers (e.g. a `| map('int') | sum` filter chain).
    usage = response.usage
    usage_dict = (
        dict(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
        if usage is not None
        else dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    )
    result = dict(
        response=response.model_dump(),
        text=text,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
        usage=usage_dict,
    )
    if reasoning:
        result["reasoning"] = reasoning
    if structured is not None:
        result["structured"] = structured
    return result


def main():
    argument_spec = dict(
        model=dict(type="str", required=True),
        messages=dict(type="list", elements="dict", required=True),
        max_tokens=dict(type="int", required=True),
        temperature=dict(type="float"),
        top_p=dict(type="float"),
        top_k=dict(type="int"),
        stop_sequences=dict(type="list", elements="str"),
        response_format=dict(type="dict"),
        tools=dict(type="list", elements="dict"),
        tool_choice=dict(type="raw"),
        enable_thinking=dict(type="bool"),
        extra_body=dict(type="dict"),
    )
    argument_spec.update(PROVIDER_ARGSPEC)

    module = AnsibleModule(
        argument_spec=argument_spec,
        supports_check_mode=False,
    )

    client = get_client(module)
    if client is None:
        # Unreachable in real Ansible execution -- get_client() only
        # returns None after calling fail_json(), which calls sys.exit().
        # Guarded anyway since it costs nothing and keeps this function
        # correct even outside that real-execution guarantee (e.g. a
        # test harness with a mocked, non-exiting fail_json).
        return
    # Imported here, not at module top-level, deliberately -- get_client()
    # already fail_json()'d (which exits) if the openai SDK isn't
    # installed, so this is the earliest point the import is guaranteed
    # safe. A top-level import would defeat that clean-failure path by
    # crashing on module load instead, same reason both sibling
    # collections keep their provider SDK imports out of top-level scope.
    from openai import OpenAIError

    kwargs = dict(
        model=module.params["model"],
        messages=module.params["messages"],
        max_tokens=module.params["max_tokens"],
    )

    for key in ("temperature", "top_p", "response_format", "tools", "tool_choice"):
        value = module.params.get(key)
        if value is not None:
            kwargs[key] = value

    stop_sequences = module.params.get("stop_sequences")
    if stop_sequences is not None:
        kwargs["stop"] = stop_sequences

    # enable_thinking is a hard default, not opt-in sugar -- it's always
    # sent explicitly on every call (see
    # choose-serving-setup-and-model-artifact / design-module-interface-
    # mirroring-siblings: a thinking-capable model left on its template
    # default can silently burn the whole max_tokens budget on reasoning
    # and never emit an answer). Precedence: an explicitly-set top-level
    # enable_thinking always wins; otherwise an explicit value the caller
    # already put in extra_body.chat_template_kwargs is honored instead of
    # being silently clobbered back to the false default; only fall back
    # to false when neither was set anywhere.
    extra_body = dict(module.params.get("extra_body") or {})
    top_k = module.params.get("top_k")
    if top_k is not None:
        extra_body["top_k"] = top_k
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    enable_thinking = module.params.get("enable_thinking")
    if enable_thinking is None:
        enable_thinking = chat_template_kwargs.get("enable_thinking", False)
    chat_template_kwargs["enable_thinking"] = enable_thinking
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    kwargs["extra_body"] = extra_body

    try:
        response = client.chat.completions.create(**kwargs)
        # A chat completion call never mutates infrastructure state -- it's
        # a query, same as the aknochow.claude/aknochow.gemini modules.
        module.exit_json(
            changed=False,
            **flatten_response(response, parse_structured=bool(module.params.get("response_format"))),
        )
    except (OpenAIError, ValueError) as e:
        # Redact base_url/api_key if the SDK's exception text happens to
        # embed them verbatim (e.g. a connection error including the full
        # request URL) -- a real redaction of the two known-sensitive
        # values, not just a cosmetic message rewrap.
        msg = str(e)
        for sensitive_key in ("api_key", "base_url"):
            value = module.params.get(sensitive_key)
            if value and value in msg:
                msg = msg.replace(value, f"<{sensitive_key}>")
        module.fail_json(msg=msg)


if __name__ == "__main__":
    main()
