# aknochow.openai

Ansible collection for calling a self-hosted [llama.cpp](https://github.com/ggml-org/llama.cpp)
`llama-server` instance directly via the official
[openai Python SDK](https://pypi.org/project/openai/) — llama-server exposes
an OpenAI-compatible `/v1/chat/completions` endpoint, so the official OpenAI
SDK pointed at a local `base_url` is the natural client, the same way
`aknochow.claude`'s `base_url` override targets a self-hosted/proxied
endpoint. Built for deterministic, structured invocation from Ansible tasks
(`register`, `set_fact`, `when`, loops), the same shape as its companion
collections [`aknochow.claude`](https://github.com/aknochow/ansible-claude)
and [`aknochow.gemini`](https://github.com/aknochow/ansible-gemini).

`base_url` targets any OpenAI-compatible endpoint, not only llama-server —
this collection also works against
[Ollama](https://github.com/ollama/ollama)'s OpenAI-compat endpoint with zero
code changes, with one documented limitation. See
[Ollama compatibility](#ollama-compatibility) below.

## Why this exists

Qwen3-class local models are plausibly the first local/free models capable
of real review/agentic work — this collection is how to find out, both as
a standalone local-dev capability and as a free-vs-paid comparison point
against Claude/Gemini for the same tasks. Unlike its two siblings, there's
no per-token billing here: the real cost is your own compute time, not a
metered API bill — see `examples/benchmark_tasks.yml`'s tokens/sec
reporting for the more meaningful "cost" axis for this provider.

Scoped to Apple Silicon (M-series unified memory) for now — llama.cpp's
Metal backend makes multi-billion-parameter local models genuinely usable
without a discrete GPU, which most Linux dev laptops don't have. A future
`aknochow.vllm` collection targeting OpenShift/KServe-hosted GPU infra is a
separate, later effort — not started here.

## Modules

| Module | Purpose |
|---|---|
| `chat` | Call `/v1/chat/completions` — structured output (`response_format`), tool calling (`tools`/`tool_choice`), reasoning control (`enable_thinking`), flattened return values |

## Requirements

```
pip install openai
```

Plus a running `llama-server` instance — this collection does not manage
its lifecycle, start it yourself first:

```bash
# Homebrew's stable bottle works for established model architectures.
brew install llama.cpp

# Download a GGUF. Q4_K_M is a good default balance of quality/size;
# official Qwen-org releases are recommended over community requants to
# avoid conversion-quality confounds.
huggingface-cli download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir ~/models

# Start the server. --reasoning off is NOT optional for reliable
# response_format results -- see "Reasoning / thinking control" below.
llama-server \
  -m ~/models/Qwen3-8B-Q4_K_M.gguf \
  -c 4096 \
  --port 8080 \
  --host 127.0.0.1 \
  -ngl 99 \
  --reasoning off
```

### Very new model architectures may need a from-source build

Brand-new model families (e.g. hybrid SSM/attention architectures) can lag
Homebrew's stable bottle by weeks. If `llama-server` fails to load a model
with a "tensor not found" or unrecognized-architecture error, try:

```bash
brew unlink llama.cpp && brew install --HEAD llama.cpp
```

Confirm Metal is genuinely active — don't assume it from throughput alone:

```bash
llama-server --list-devices                 # confirms a Metal device exists
llama-server -m <path> -v ...                # -v prints "assigned to device MTL0" per layer on load
```

## Ollama compatibility

`chat`'s `base_url` targets any OpenAI-compatible endpoint, not only
llama-server. Confirmed live against a real [Ollama](https://github.com/ollama/ollama)
instance (`base_url: http://127.0.0.1:11434/v1`, no code changes required):
basic calls, `response_format` (structured output), and `tools`/`tool_choice`
(function calling) all work correctly. Ollama also names its reasoning-trace
field differently from llama-server (`reasoning` vs. `reasoning_content`) --
this module checks both, so `result.reasoning` is populated correctly
regardless of which backend served the request.

One documented limitation: **`enable_thinking` has no effect via Ollama's
OpenAI-compat endpoint.** Ollama's `/v1/chat/completions` shim does not
expose any thinking-control parameter, so this module's `enable_thinking`
(and the `chat_template_kwargs.enable_thinking` extra-body field it sets)
is silently ignored there -- Qwen3's thinking trace runs at whatever the
loaded model's template default is, regardless of what this module sends.
If you need thinking-trace control against Ollama, use Ollama's native
`/api/chat` endpoint directly (its own `think: false` parameter, confirmed
working) rather than this module.

## Auth

```bash
export ANSIBLE_OPENAI_BASE_URL=http://127.0.0.1:8080/v1   # this is already the default
export ANSIBLE_OPENAI_API_KEY=whatever                     # optional; llama-server doesn't validate it
```

Deliberately collection-specific env var names, not the openai SDK's own
`OPENAI_API_KEY`/`OPENAI_BASE_URL` — a dev machine may already have those
set for unrelated real OpenAI usage, and silently inheriting them here
would be a confusing cross-tool collision for a self-hosted-only
collection.

**Migration note (collection renamed `aknochow.llama` → `aknochow.openai`):**
`ANSIBLE_LLAMA_BASE_URL`/`ANSIBLE_LLAMA_API_KEY` are no longer read. If
you have either set from before this rename, the module will silently
fall back to its hardcoded default (`http://127.0.0.1:8080/v1`) instead
of your configured value, with no warning — update your shell exports
(or CI config) to `ANSIBLE_OPENAI_BASE_URL`/`ANSIBLE_OPENAI_API_KEY`.

### `chat` — basic call

```yaml
- name: Basic chat completion
  aknochow.openai.chat:
    model: qwen3-8b
    max_tokens: 512
    messages:
      - role: user
        content: "Summarize this changelog in one sentence: {{ changelog }}"
  register: result
# result.text, result.usage.{prompt_tokens,completion_tokens,total_tokens}
```

There is no separate `system` parameter — send a `role: system` message in
`messages` instead. That's the actual OpenAI/llama.cpp wire protocol
(unlike Anthropic's and Gemini's APIs, which do have a dedicated system
field), so a separate parameter here would just be translation-layer sugar
with no wire-accuracy benefit.

### Structured output

```yaml
- aknochow.openai.chat:
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
            name: { type: string }
            severity: { type: string, enum: [low, medium, high, critical] }
          required: [name, severity]
          additionalProperties: false
  register: result
# result.structured.name, result.structured.severity
```

llama-server enforces `response_format` via grammar-constrained decoding —
the model literally cannot sample a token that violates the schema, a
stronger guarantee than Claude/Gemini's function-calling-based structured
output. Validated live against both Qwen3-8B and Qwen3.8-27B with
`enable_thinking: false` — see "Reasoning / thinking control" below for
why that matters.

### Tool calling

```yaml
- aknochow.openai.chat:
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
              city: { type: string }
            required: [city]
    tool_choice: required
  register: result
# result.tool_calls -> [{id, name, args}, ...]
```

`args` is already a parsed dict here, not the raw JSON string the wire
protocol actually sends (`tool_calls[].function.arguments` is a
JSON-encoded string on the wire, confirmed live) — this module parses it
for you so it converges in *type*, not just field name, with
`aknochow.claude`'s `input` and `aknochow.gemini`'s `args`.

### Reasoning / thinking control

```yaml
- aknochow.openai.chat:
    model: qwen3-8b
    max_tokens: 4096          # needs real headroom when thinking is on
    enable_thinking: true
    messages:
      - role: user
        content: "What is 12*13? Show your reasoning."
  register: result
# result.reasoning (present only when a trace was generated), result.text
```

`enable_thinking` defaults to `false` and is sent explicitly on every
single call — not optional sugar. Qwen3's whole family can silently burn
the entire `max_tokens` budget on the reasoning trace and never emit an
answer at all (`finish_reason: length`, empty `text`) if left at the
template default. Confirmed live on both Qwen3-8B and Qwen3.8-27B: leaving
reasoning on with a modest `max_tokens` reliably produces this failure;
`enable_thinking: false` reliably avoids it (a full 5-call sweep against
Qwen3.8-27B went from 0/5 to 5/5 successful once disabled).

A large `max_tokens` + `enable_thinking: true` call can also run well past
the default 120-second `timeout` (Qwen3.8-27B generates at ~10 tok/s, so a
2048-token reasoning-heavy call can take several minutes) — raise `timeout`
too (e.g. `400`) or you'll get a clean but unhelpful `Request timed out.`
failure after the SDK's own retries.

## Ad-hoc usage (no playbook needed)

For quick prompt testing without writing a playbook, install the collection
locally once:

```bash
ansible-galaxy collection install . --force
```

This copies the collection into `~/.ansible/collections/` — it's a
snapshot, not a symlink, so re-run this after editing the source before
your next ad-hoc call picks up the change.

Then run any prompt directly from the command line:

```bash
ansible localhost -m aknochow.openai.chat -a '{
  "model": "qwen3-8b",
  "max_tokens": 400,
  "messages": [{"role": "user", "content": "your prompt here"}]
}'
```

`-a` takes a JSON object here, not flat `key=value` args, since `messages`
is a list of dicts. Add `response_format`, `tools`/`tool_choice`, or
`enable_thinking` to the same JSON blob exactly as in the playbook examples
above — e.g. for structured output:

```bash
ansible localhost -m aknochow.openai.chat -a '{
  "model": "qwen3-8b",
  "max_tokens": 512,
  "messages": [{"role": "user", "content": "Write a Python function that checks if a string is a palindrome."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "code_gen",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "language": {"type": "string"},
          "code": {"type": "string"},
          "explanation": {"type": "string"}
        },
        "required": ["language", "code", "explanation"],
        "additionalProperties": false
      }
    }
  }
}'
```

If port 8080 is already in use by something else on your machine (a real
example hit during development: a local `jira_emulator` service squatting
on it, producing a confusing `Error code: 404` rather than a connection
error), either start `llama-server` on a different port and pass
`base_url` explicitly, or set `ANSIBLE_OPENAI_BASE_URL` once instead of
repeating it on every call.

## Examples

See `examples/benchmark_tasks.yml` — a small math/sentiment/extraction
benchmark, live-verified end to end via a real `ansible-playbook` run
against a real llama-server (3/3 correct). Its `benchmark_tasks` list is
identical to `aknochow.claude`'s own comparison playbook, so it can be
reused directly in a future 3-way Claude/Gemini/Qwen comparison rather
than rewritten.

## Testing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ansible-core openai pytest
python -m pytest tests/unit/
```

Unit tests mock the `openai` SDK — no network access or a running
llama-server required. Live-verified during development against real
llama-server instances (Qwen3-8B on Homebrew's stable build, Qwen3.8-27B
requiring a from-source HEAD build for its newer architecture) for basic
calls, structured output, and tool calling — see
`examples/benchmark_tasks.yml` for a runnable live smoke test.
