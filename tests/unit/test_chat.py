# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_openai():
    mock_sdk = MagicMock()
    mock_sdk.OpenAI = MagicMock()
    mock_sdk.OpenAIError = type("OpenAIError", (Exception,), {})
    sys.modules["openai"] = mock_sdk
    yield mock_sdk
    sys.modules.pop("openai", None)


def make_tool_call(call_id="call_1", name="get_weather", arguments='{"city": "Boston"}'):
    function = MagicMock()
    function.name = name
    function.arguments = arguments
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function = function
    return tool_call


def make_message(content="", reasoning_content=None, tool_calls=None):
    message = MagicMock()
    message.content = content
    # MagicMock auto-vivifies attributes as truthy MagicMocks, so both of
    # these must be set explicitly -- otherwise every response would look
    # like it produced a reasoning trace and/or a non-iterable tool_calls
    # blob (same hazard aknochow.gemini's test suite already flags for its
    # own function_call attribute).
    message.reasoning_content = reasoning_content
    message.tool_calls = tool_calls
    return message


def make_choice(content="", reasoning_content=None, finish_reason="stop", tool_calls=None):
    choice = MagicMock()
    choice.message = make_message(content, reasoning_content, tool_calls)
    choice.finish_reason = finish_reason
    return choice


def make_response(choices, usage_kwargs=None, usage_is_none=False):
    response = MagicMock()
    response.choices = choices
    if usage_is_none:
        response.usage = None
    else:
        usage = MagicMock()
        defaults = dict(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        defaults.update(usage_kwargs or {})
        for k, v in defaults.items():
            setattr(usage, k, v)
        response.usage = usage
    response.model_dump.return_value = {"id": "chatcmpl_123", "object": "chat.completion"}
    return response


class TestFlattenResponse:
    def test_text_only_response(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        response = make_response([make_choice(content="hello world")])
        result = flatten_response(response)

        assert result["text"] == "hello world"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15
        assert "reasoning" not in result
        assert "structured" not in result

    def test_reasoning_present_when_returned(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        response = make_response(
            [make_choice(content="4", reasoning_content="Let me think... 2+2=4")]
        )
        result = flatten_response(response)

        assert result["reasoning"] == "Let me think... 2+2=4"
        assert result["text"] == "4"

    def test_reasoning_absent_by_default(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        response = make_response([make_choice(content="hi", reasoning_content=None)])
        result = flatten_response(response)

        assert "reasoning" not in result

    def test_structured_output_parsed_when_requested(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        response = make_response(
            [make_choice(content='{"name": "bug-1", "severity": "high"}')]
        )
        result = flatten_response(response, parse_structured=True)

        assert result["structured"] == {"name": "bug-1", "severity": "high"}

    def test_valid_json_text_not_parsed_when_not_requested(self, mock_openai):
        # Regression check, mirrors aknochow.claude's identical fix: text
        # that HAPPENS to parse as JSON must not silently gain a
        # `structured` key when response_format wasn't set.
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        response = make_response(
            [make_choice(content='{"name": "bug-1", "severity": "high"}')]
        )
        result = flatten_response(response)

        assert "structured" not in result

    def test_empty_content_when_finish_reason_length(self, mock_openai):
        # The thinking-budget-exhaustion failure mode confirmed live this
        # session: finish_reason=length with empty text when reasoning
        # burns the whole max_tokens budget.
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        response = make_response(
            [make_choice(content="", reasoning_content="still thinking...", finish_reason="length")]
        )
        result = flatten_response(response, parse_structured=True)

        assert result["text"] == ""
        assert result["finish_reason"] == "length"
        assert "structured" not in result
        assert result["reasoning"] == "still thinking..."

    def test_tool_calls_empty_when_none(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        response = make_response([make_choice(content="hi")])
        result = flatten_response(response)

        assert result["tool_calls"] == []

    def test_tool_calls_arguments_parsed_to_dict(self, mock_openai):
        # The core convergence guarantee: arguments comes back as a JSON
        # STRING on the wire (confirmed live against a real llama-server
        # in design-module-interface-mirroring-siblings) and must be
        # parsed to a dict here, matching Claude's `input` and Gemini's
        # `args` in *type*, not just field name.
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        tool_call = make_tool_call(call_id="call_1", name="get_weather", arguments='{"city": "Boston"}')
        response = make_response([make_choice(content="", finish_reason="tool_calls", tool_calls=[tool_call])])
        result = flatten_response(response)

        assert result["tool_calls"] == [{"id": "call_1", "name": "get_weather", "args": {"city": "Boston"}}]
        assert isinstance(result["tool_calls"][0]["args"], dict)
        assert result["finish_reason"] == "tool_calls"

    def test_multiple_tool_calls(self, mock_openai):
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        calls = [
            make_tool_call(call_id="call_1", name="get_weather", arguments='{"city": "Boston"}'),
            make_tool_call(call_id="call_2", name="get_time", arguments='{"tz": "EST"}'),
        ]
        response = make_response([make_choice(content="", tool_calls=calls)])
        result = flatten_response(response)

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][1]["name"] == "get_time"

    def test_none_usage_does_not_crash(self, mock_openai):
        # Regression check for a real review finding: response.usage.
        # prompt_tokens was accessed unconditionally, which would raise
        # an unhandled AttributeError on a server that omits usage
        # entirely. usage itself must stay a dict (matches the RETURN
        # doc's "returned: always"), with 0 sub-fields (matching the
        # declared type: int) instead of crashing or going null.
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        response = make_response([make_choice(content="hi")], usage_is_none=True)
        result = flatten_response(response)

        assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_empty_choices_raises_value_error(self, mock_openai):
        # Regression check for a real review finding: response.choices[0]
        # was accessed unconditionally, which would raise an unhandled
        # IndexError (escaping main()'s OpenAIError-only except clause) on
        # an empty choices list. Must raise a clean, catchable ValueError
        # instead.
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        response = make_response([])

        with pytest.raises(ValueError):
            flatten_response(response)

    def test_none_message_raises_value_error(self, mock_openai):
        # Regression check for a real review finding: choice.message.content
        # was accessed without checking whether message itself is None,
        # which some OpenAI-compatible servers could theoretically return.
        # Same treatment as the empty-choices guard -- a clean, catchable
        # ValueError instead of an unhandled AttributeError.
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        choice = MagicMock()
        choice.message = None
        response = make_response([choice])

        with pytest.raises(ValueError):
            flatten_response(response)

    def test_malformed_arguments_falls_back_to_raw_string(self, mock_openai):
        # Defensive fallback -- should not happen under grammar-constrained
        # tool-call decoding, but must not crash the whole call if it did.
        from ansible_collections.aknochow.llama.plugins.modules.chat import (
            flatten_response,
        )

        tool_call = make_tool_call(arguments="not valid json{")
        response = make_response([make_choice(content="", tool_calls=[tool_call])])
        result = flatten_response(response)

        assert result["tool_calls"][0]["args"] == "not valid json{"


class TestMainReportsChanged:
    def test_main_reports_changed_false(self, mock_openai, monkeypatch):
        from ansible_collections.aknochow.llama.plugins.modules import chat as chat_module

        fake_module = MagicMock()
        fake_module.params = {
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "stop_sequences": None,
            "response_format": None,
            "tools": None,
            "tool_choice": None,
            "enable_thinking": None,
            "extra_body": None,
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "not-needed",
            "timeout": 120.0,
            "max_retries": 2,
        }
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = make_response(
            [make_choice(content="hi")]
        )
        monkeypatch.setattr(chat_module, "AnsibleModule", lambda **kwargs: fake_module)

        chat_module.main()

        fake_module.exit_json.assert_called_once()
        assert fake_module.exit_json.call_args.kwargs["changed"] is False


class TestMainHandlesEmptyChoicesCleanly:
    def test_empty_choices_fails_cleanly_not_a_crash(self, mock_openai, monkeypatch):
        from ansible_collections.aknochow.llama.plugins.modules import chat as chat_module

        fake_module = MagicMock()
        fake_module.params = {
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "stop_sequences": None,
            "response_format": None,
            "tools": None,
            "tool_choice": None,
            "enable_thinking": None,
            "extra_body": None,
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "not-needed",
            "timeout": 120.0,
            "max_retries": 2,
        }
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = make_response([])
        monkeypatch.setattr(chat_module, "AnsibleModule", lambda **kwargs: fake_module)

        chat_module.main()

        fake_module.fail_json.assert_called_once()
        fake_module.exit_json.assert_not_called()


class TestMainHandlesNoneClientCleanly:
    def test_none_client_returns_without_crashing(self, mock_openai, monkeypatch):
        # Defensive guard for a recurring review theme: get_client() only
        # ever returns None in real Ansible execution after fail_json()
        # has already called sys.exit() -- unreachable there -- but must
        # not blow up with an AttributeError on client.chat... if it ever
        # did return None (e.g. a mocked, non-exiting fail_json).
        from ansible_collections.aknochow.llama.plugins.modules import chat as chat_module

        fake_module = MagicMock()
        fake_module.params = {
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "stop_sequences": None,
            "response_format": None,
            "tools": None,
            "tool_choice": None,
            "enable_thinking": None,
            "extra_body": None,
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "not-needed",
            "timeout": 120.0,
            "max_retries": 2,
        }
        monkeypatch.setattr(chat_module, "AnsibleModule", lambda **kwargs: fake_module)
        monkeypatch.setattr(chat_module, "get_client", lambda module: None)

        chat_module.main()

        fake_module.exit_json.assert_not_called()
        fake_module.fail_json.assert_not_called()
        mock_openai.OpenAI.return_value.chat.completions.create.assert_not_called()


class TestMainRedactsSensitiveValuesFromExceptionMessages:
    def _run_main_with_exception(self, mock_openai, monkeypatch, exception_message):
        from ansible_collections.aknochow.llama.plugins.modules import chat as chat_module

        fake_module = MagicMock()
        fake_module.params = {
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "stop_sequences": None,
            "response_format": None,
            "tools": None,
            "tool_choice": None,
            "enable_thinking": None,
            "extra_body": None,
            "base_url": "http://internal-llama-host.example.com:8080/v1",
            "api_key": "real-secret-key-value",
            "timeout": 120.0,
            "max_retries": 2,
        }
        monkeypatch.setattr(chat_module, "AnsibleModule", lambda **kwargs: fake_module)
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = (
            mock_openai.OpenAIError(exception_message)
        )

        chat_module.main()
        return fake_module

    def test_base_url_redacted_from_exception_message(self, mock_openai, monkeypatch):
        # Regression check for a real review finding: OpenAI SDK exception
        # text can embed the full request URL. This is an actual
        # redaction, not a cosmetic message rewrap.
        fake_module = self._run_main_with_exception(
            mock_openai,
            monkeypatch,
            "Connection error to http://internal-llama-host.example.com:8080/v1: refused",
        )

        fake_module.fail_json.assert_called_once()
        msg = fake_module.fail_json.call_args.kwargs["msg"]
        assert "internal-llama-host.example.com" not in msg
        assert "<base_url>" in msg

    def test_api_key_redacted_from_exception_message(self, mock_openai, monkeypatch):
        fake_module = self._run_main_with_exception(
            mock_openai,
            monkeypatch,
            "Authorization failed for key real-secret-key-value",
        )

        msg = fake_module.fail_json.call_args.kwargs["msg"]
        assert "real-secret-key-value" not in msg
        assert "<api_key>" in msg

    def test_message_without_sensitive_values_passes_through_unchanged(self, mock_openai, monkeypatch):
        fake_module = self._run_main_with_exception(mock_openai, monkeypatch, "Request timed out")

        assert fake_module.fail_json.call_args.kwargs["msg"] == "Request timed out"


class TestMainRequestConstruction:
    def _run_main(self, mock_openai, monkeypatch, params_overrides):
        from ansible_collections.aknochow.llama.plugins.modules import chat as chat_module

        fake_module = MagicMock()
        fake_module.params = {
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "stop_sequences": None,
            "response_format": None,
            "tools": None,
            "tool_choice": None,
            "enable_thinking": None,
            "extra_body": None,
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "not-needed",
            "timeout": 120.0,
            "max_retries": 2,
        }
        fake_module.params.update(params_overrides)
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = make_response(
            [make_choice(content="hi")]
        )
        monkeypatch.setattr(chat_module, "AnsibleModule", lambda **kwargs: fake_module)

        chat_module.main()

        return mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs

    def test_enable_thinking_always_sent_explicitly(self, mock_openai, monkeypatch):
        # Hard-default requirement: every call must explicitly declare
        # enable_thinking, defaulting to false, regardless of whether the
        # caller touched extra_body at all.
        call_kwargs = self._run_main(mock_openai, monkeypatch, {})

        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_enable_thinking_true_is_passed_through(self, mock_openai, monkeypatch):
        call_kwargs = self._run_main(mock_openai, monkeypatch, {"enable_thinking": True})

        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    def test_extra_body_enable_thinking_honored_when_top_level_unset(self, mock_openai, monkeypatch):
        # Regression check for a real review finding: a caller setting
        # enable_thinking inside extra_body.chat_template_kwargs while
        # leaving the top-level param untouched must NOT have that
        # explicit value silently clobbered back to the false default.
        call_kwargs = self._run_main(
            mock_openai,
            monkeypatch,
            {"enable_thinking": None, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        )

        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    def test_top_level_enable_thinking_wins_over_extra_body_when_both_set(self, mock_openai, monkeypatch):
        # The dedicated param is still authoritative when the caller
        # explicitly sets both -- extra_body only fills the gap when the
        # top-level param was left unset.
        call_kwargs = self._run_main(
            mock_openai,
            monkeypatch,
            {"enable_thinking": False, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        )

        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_top_k_routes_through_extra_body(self, mock_openai, monkeypatch):
        call_kwargs = self._run_main(mock_openai, monkeypatch, {"top_k": 20})

        assert call_kwargs["extra_body"]["top_k"] == 20
        assert "top_k" not in call_kwargs

    def test_stop_sequences_maps_to_stop_kwarg(self, mock_openai, monkeypatch):
        call_kwargs = self._run_main(mock_openai, monkeypatch, {"stop_sequences": ["END"]})

        assert call_kwargs["stop"] == ["END"]
        assert "stop_sequences" not in call_kwargs

    def test_callers_extra_body_is_preserved_alongside_injected_fields(self, mock_openai, monkeypatch):
        call_kwargs = self._run_main(
            mock_openai,
            monkeypatch,
            {"extra_body": {"min_p": 0.05, "chat_template_kwargs": {"some_other_var": True}}},
        )

        assert call_kwargs["extra_body"]["min_p"] == 0.05
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["some_other_var"] is True
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_no_response_format_key_when_not_requested(self, mock_openai, monkeypatch):
        call_kwargs = self._run_main(mock_openai, monkeypatch, {})

        assert "response_format" not in call_kwargs

    def test_response_format_passed_through_verbatim(self, mock_openai, monkeypatch):
        schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        call_kwargs = self._run_main(mock_openai, monkeypatch, {"response_format": schema})

        assert call_kwargs["response_format"] == schema

    def test_no_tools_key_when_not_requested(self, mock_openai, monkeypatch):
        call_kwargs = self._run_main(mock_openai, monkeypatch, {})

        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    def test_tools_and_tool_choice_passed_through_verbatim(self, mock_openai, monkeypatch):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        call_kwargs = self._run_main(
            mock_openai, monkeypatch, {"tools": tools, "tool_choice": "required"}
        )

        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "required"

    def test_tool_choice_accepts_dict_shape(self, mock_openai, monkeypatch):
        # tool_choice is type=raw specifically to accept either a plain
        # string (auto/none/required) or a dict forcing a specific tool.
        forced = {"type": "function", "function": {"name": "get_weather"}}
        call_kwargs = self._run_main(mock_openai, monkeypatch, {"tool_choice": forced})

        assert call_kwargs["tool_choice"] == forced
