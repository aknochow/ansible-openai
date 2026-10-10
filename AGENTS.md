# ansible-openai: Project Context

## Read first

- [CONTRIBUTING.md](CONTRIBUTING.md): branch names, commits, test and lint commands
- [README.md](README.md): project overview

## What this repo is

Ansible collection for calling a hosted OpenAI API or any OpenAI-compatible endpoint via the official OpenAI Python SDK. Built for deterministic, structured invocation from Ansible tasks, the same shape as `aknochow.claude` and `aknochow.gemini`.

## Commands

Test and sanity commands, copied from [CONTRIBUTING.md](CONTRIBUTING.md):

### Quick test run:
```bash
uv run pytest
```

### Syncing dependencies:
```bash
uv sync --extra dev
uv run pytest -v
```

### Running sanity tests:
Ansible sanity tests require the repository to be within an `ansible_collections/aknochow/openai` directory hierarchy:
```bash
uv run ansible-test sanity --local --python 3.13 -v
```
