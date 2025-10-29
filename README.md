# ReAgent - Task Planning Agent

ReAgent turns a natural-language request into a concise, sequential plan of actionable tasks. It scans your repository for relevant files, extracts high-signal keywords, and optionally uses an Ollama-hosted LLM to refine the plan. When available, it also anchors steps to external automation tools (via ToolRouter) and suggests helpful commands.

The generated plan is deterministic when LLMs are unavailable, and always validated to be linear, compact, and executable by a coding agent or developer.

## Highlights
- Produces a compact, ordered list of tasks with explicit dependencies and phases.
- Mines lightweight project context by scanning your repo for keyword matches.
- Optional LLM planning and refinement through Ollama (default: `llama3.1`).
- Optional ToolRouter integration to surface `fs.*`, `git.*`, `github.*` style tools.
- Robust fallbacks when external dependencies are unavailable.

## Quick Start

Prerequisites:
- Python 3.11+
- Optional: Ollama with the `llama3.1` model pulled
- Optional: ToolRouter

Install (editable) with `uv` or pip:

```bash
# using uv
uv venv
uv pip install -e .

# or using pip (ensure CUDA/torch settings match your machine or adjust pyproject)
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

CLI usage:

```bash
# Basic
reagent "Add password reset with token expiry"

# With 1 LLM refinement round (if Ollama is available)
reagent "Refactor signup flow to add analytics" --refine 1
```

The CLI prints a numbered plan with each task's phase, dependency, description, and a suggested command when applicable.

## What It Generates

The agent returns a list of Task objects with these fields:
- `name`: short title
- `description`: concise explanation of the action
- `phase`: one of `context`, `analysis`, `execution`, `validation`
- `depends_on`: the exact name of a prior task this task consumes
- `suggested_command` (optional): a shell/tool hint to kick off the step

Example (abridged, actual content depends on your prompt and repo):

```text
1. Survey repository layout
   Phase: context
   List top-level directories and files to anchor discovery.
   Command hint: ls

2. Inspect core package
   Phase: context
   Depends on: Survey repository layout
   Drill into src/reagent to catalogue modules and entry points.
   Command hint: ls src/reagent

3. Search for relevant code paths
   Phase: context
   Depends on: Inspect core package
   Ripgrep for high-signal keywords to build a review checklist.
   Command hint: rg -e "reset" -e "token" src

...

N. Run validation suite
   Phase: validation
   Depends on: Refresh automated coverage
   Execute project checks to confirm the updates integrate cleanly.
   Command hint: pytest && git status -sb
```

## How It Works

Top-level entrypoint: `reagent` (see `src/reagent/__init__.py`). The CLI collects a prompt and calls `PlanningAgent.plan()`.

Core components (in `src/reagent/agent.py`):

- Task (dataclass)
  - Represents a single step with `name`, `description`, `phase`, `depends_on`, and `suggested_command`.

- CodeMatch (dataclass)
  - A lightweight pointer to a file and line that matched the prompt's keywords, used to anchor context.

- CodeContextBuilder
  - Scans candidate source files under the repo (skips common build/venv directories).
  - Searches for prompt-derived keywords and captures an excerpt per match.
  - Provides fallbacks when no matches are found (high-level repo structure hints).
  - Formats a compact "project context" string for prompts/refinement.

- LLMTaskPlanner (optional)
  - Uses LangChain + `langchain_ollama.ChatOllama` to ask an LLM for a JSON task list.
  - Builds a planning prompt with your request, keywords, and project context.
  - Parses the returned JSON into `Task` objects with strict validation.
  - If Ollama is unavailable, the system falls back to deterministic planning.

- PlanRefiner (optional)
  - Iteratively refines an existing plan using an LLM chain (LangChain `LLMChain`).
  - Preserves ordering, phases, and task count unless clearly incorrect.

- PlanningAgent (orchestrator)
  - Extracts up to three high-signal keywords using NLTK's tokenizer and stopwords (with safe fallbacks and quiet download if needed).
  - Gathers code context via `CodeContextBuilder`.
  - Enumerates tools via ToolRouter if installed; otherwise uses a small static catalog. When tools exist, builds a tool-anchored plan with explicit `fs.*`, `git.*`, or `github.*` usage hints; otherwise builds a heuristic plan.
  - Optionally refines action tasks via `PlanRefiner` when `--refine` is provided and Ollama is live.
  - Verifies the final plan: ensures 10 or fewer tasks, includes at least one `execution` task, phases are valid, no duplicate names, and dependency references are coherent. Names are normalized and a linear dependency chain is enforced.

## Optional Integrations

- Ollama LLM
  - Install and run Ollama, then pull `llama3.1`.
  - If not present, ReAgent still works (deterministic plan; no refinement).

- ToolRouter
  - If the `tool-router` package is importable, `PlanningAgent` will instantiate a router and list available tools for plan anchoring.
  - When unavailable or initialization fails, a default static tool catalog is used for suggestions only.

## Programmatic Usage

```python
from reagent.agent import PlanningAgent

agent = PlanningAgent()
tasks = agent.plan("Add password reset with token expiry", refine_rounds=1)
for t in tasks:
    print(t.phase, t.name, '->', t.depends_on)
```

## Notes & Limitations
- The agent plans; it does not execute tasks or modify code.
- Suggested commands assume tools like `rg` and `pytest` are available. They are hints, not hard requirements.
- Repository scanning is lightweight by design and may miss deep semantics; use LLM refinement for improved shaping if Ollama is available.
- `torch` is pinned in `pyproject.toml` for environments that need it; this project itself does not import `torch` in the planner. Adjust as needed for your machine.

## Development
- Code lives in `src/reagent/agent.py` and `src/reagent/__init__.py`.
- Logging is on by default at INFO; raise to DEBUG for more detail.
- Tests are not included; you can run `pytest` where relevant in your repo.


