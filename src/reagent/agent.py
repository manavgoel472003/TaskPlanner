from __future__ import annotations

import json
import logging
import pathlib
import re
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords as nltk_stopwords  # type: ignore
import nltk  # type: ignore

try:  # pragma: no cover - optional dependency
    from tool_router.router import ToolRouter
    from tool_router.settings import RouterConfig
except ImportError:  # pragma: no cover - optional dependency
    ToolRouter = None  # type: ignore[assignment]
    RouterConfig = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from tool_router.router import ToolRouter as ToolRouterType
    from tool_router.settings import RouterConfig as RouterConfigType
else:  # pragma: no cover - types only used for hints
    ToolRouterType = Any  # type: ignore[assignment]
    RouterConfigType = Any  # type: ignore[assignment]


@dataclass(slots=True)
class Task:
    """Represents a single task in the planner's workflow."""

    name: str
    description: str
    phase: str = "execution"
    suggested_command: Optional[str] = None
    depends_on: Optional[str] = None


@dataclass(slots=True)
class CodeMatch:
    """Simple representation of a file location relevant to the prompt."""

    path: str
    line: int
    snippet: str


class CodeContextBuilder:
    """Gathers lightweight project context to help the planner tailor tasks."""

    _SOURCE_EXTENSIONS: Sequence[str] = (
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".json",
        ".yml",
        ".yaml",
    )

    def __init__(self, root: Optional[pathlib.Path] = None) -> None:
        project_root = root or pathlib.Path(__file__).resolve().parents[2]
        self._root = project_root
        self._src_root = self._root / "src"
        logging.info(
            "CodeContextBuilder initialised (root=%s, src=%s)", self._root, self._src_root
        )

    def build_snapshot(self, keywords: Sequence[str], limit: int = 5) -> List[CodeMatch]:
        matches: List[CodeMatch] = []
        normalized = [kw.lower() for kw in keywords]
        logging.info(
            "Building code context snapshot (keywords=%s, limit=%d)", normalized, limit
        )
        for path in self._iter_candidate_files():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            lowered = text.lower()
            for keyword in normalized:
                idx = lowered.find(keyword)
                if idx == -1:
                    continue
                line_no = text.count("\n", 0, idx) + 1
                snippet = self._extract_line(text, idx)
                matches.append(
                    CodeMatch(
                        path=str(path.relative_to(self._root)),
                        line=line_no,
                        snippet=snippet.strip(),
                    )
                )
                break

            if len(matches) >= limit:
                break

        if matches:
            logging.info("Snapshot discovery produced %d match(es)", len(matches))
            return matches

        # Fallback: return high-level structure hints.
        fallback = self._fallback_structure(limit)
        logging.info(
            "No keyword matches found; using fallback structure with %d entrie(s)",
            len(fallback),
        )
        return fallback

    def format_for_prompt(self, matches: Sequence[CodeMatch]) -> str:
        if not matches:
            return ""
        lines = [
            f"- {match.path}:{match.line} â†’ {match.snippet}"
            for match in matches
        ]
        logging.info("Formatted project context for prompt (%d entrie(s))", len(matches))
        return "\n".join(lines[:10])

    def highlight_summary(self, matches: Sequence[CodeMatch]) -> str:
        if not matches:
            return "Focus on the files identified during search."
        highlighted = ", ".join(match.path for match in matches[:5])
        return f"Prioritise inspecting: {highlighted}"

    def _iter_candidate_files(self) -> Iterable[pathlib.Path]:
        search_paths: List[pathlib.Path] = []
        if self._src_root.exists():
            search_paths.append(self._src_root)
        search_paths.append(self._root)

        seen: set[pathlib.Path] = set()
        for base in search_paths:
            max_files = 0
            for path in base.rglob("*"):
                if path in seen or not path.is_file():
                    continue
                rel_parts = path.relative_to(self._root).parts
                if any(
                    part in {".venv", "venv", ".git", "__pycache__", "node_modules"} or part.startswith(".")
                    for part in rel_parts
                ):
                    continue
                if path.suffix.lower() not in self._SOURCE_EXTENSIONS and path.name not in {"pyproject.toml", "README.md"}:
                    continue
                seen.add(path)
                yield path
                max_files += 1
                if max_files >= 200:
                    break

    @staticmethod
    def _extract_line(text: str, index: int) -> str:
        start = text.rfind("\n", 0, index)
        end = text.find("\n", index)
        if start == -1:
            start = 0
        else:
            start += 1
        if end == -1:
            end = len(text)
        return text[start:end].strip()

    def _fallback_structure(self, limit: int) -> List[CodeMatch]:
        entries: List[CodeMatch] = []
        skip_names = {".venv", "venv", ".git", "__pycache__", "node_modules", ".mypy_cache"}
        for idx, child in enumerate(sorted(self._root.iterdir(), key=lambda p: p.name)):
            if child.name in skip_names or child.name.startswith("."):
                continue
            if len(entries) >= limit:
                break
            entries.append(
                CodeMatch(
                    path=str(child.relative_to(self._root)),
                    line=1,
                    snippet="directory" if child.is_dir() else "file",
                )
            )
            if child.is_dir() and child.name == "src":
                reagent_dir = child / "reagent"
                if reagent_dir.exists():
                    entries.append(
                        CodeMatch(
                            path=str(reagent_dir.relative_to(self._root)),
                            line=1,
                            snippet="directory",
                        )
                    )
                    for sub in sorted(reagent_dir.glob("*")):
                        if sub.is_file() and sub.suffix.lower() in self._SOURCE_EXTENSIONS and len(entries) < limit:
                            entries.append(
                                CodeMatch(
                                    path=str(sub.relative_to(self._root)),
                                    line=1,
                                    snippet="file",
                                )
                            )
                        if len(entries) >= limit:
                            break
            if len(entries) >= limit:
                break
        return entries


class LLMTaskPlanner:
    """Leverages an Ollama-hosted model to derive sequential execution tasks."""

    def __init__(self, model: Optional[ChatOllama] = None) -> None:
        self._startup_error: Optional[Exception] = None
        if model is not None:
            self._model = model
            logging.info("PlanningAgent using provided LLM: %s", getattr(model, "model", "unknown"))
            return
        try:
            self._model = ChatOllama(model="llama3.1", temperature=0.0)
            logging.info("PlanningAgent initialised with Ollama model: llama3.1")
        except Exception as exc:  # pragma: no cover - defensive; model may be unavailable
            self._startup_error = exc
            self._model = None
            logging.warning("PlanningAgent could not initialise Ollama model llama3.1: %s", exc)

    def generate_tasks(self, prompt: str, keywords: List[str], project_context: str) -> List[Task]:
        if self._model is None:
            return []

        keyword_list = ", ".join(keywords) if keywords else "none"
        instructions = (
            "You are a senior software planning assistant. "
            "Given a request from a developer, produce the smallest ordered list of follow-up tasks that another "
            "coding agent should perform to implement the request. "
            "Each task must build on the previous one; avoid parallel tracks or vague guidance. "
            "Do not write code, focus only on planning. "
            "Whenever possible, reference the most relevant files or modules from the provided project context, "
            "merge overlapping work into a single task, and never invent file names or directories that are not "
            "listed in the project context. If dependencies need to be added, describe the action generically unless the "
            "specific package name already appears in the project context. The repository uses Python packaging "
            "(pyproject.toml); avoid suggesting npm/yarn commands or JavaScript-only libraries. "
            "Important: include a version-control safety requirement — create or switch to a dedicated feature branch "
            "(e.g., 'feat/<short-slug>') at the start, and after each major implementation step ensure the plan calls out "
            "to stage, commit, and push changes to that branch so rollbacks are possible.\n\n"
            "Keep the total number of tasks at or below 10.\n\n"
            "Example plan (for a password reset feature) to illustrate the expected structure:\n"
            "[\n"
            "  {\n"
            "    \"name\": \"Capture repository map\",\n"
            "    \"description\": \"Run ls at the repo root and record the directory layout in a scratch doc.\",\n"
            "    \"phase\": \"context\",\n"
            "    \"depends_on\": null,\n"
            "    \"suggested_command\": \"ls\"\n"
            "  },\n"
            "  {\n"
            "    \"name\": \"Inventory auth package\",\n"
            "    \"description\": \"Inspect src/auth to list modules, entry points, and helpers relevant to credentials; append findings to the scratch doc.\",\n"
            "    \"phase\": \"context\",\n"
            "    \"depends_on\": \"Capture repository map\",\n"
            "    \"suggested_command\": \"ls src/auth\"\n"
            "  },\n"
            "  {\n"
            "    \"name\": \"Review reset flows\",\n"
            "    \"description\": \"Use the module inventory to grep for reset-related handlers, opening matching files and noting responsibilities and gaps.\",\n"
            "    \"phase\": \"context\",\n"
            "    \"depends_on\": \"Inventory auth package\",\n"
            "    \"suggested_command\": \"rg \\\"reset\\\" src/auth\"\n"
            "  },\n"
            "  {\n"
            "    \"name\": \"Outline password reset changes\",\n"
            "    \"description\": \"Summarise the reviewed files into a change list covering data models, routes, and validation updates.\",\n"
            "    \"phase\": \"analysis\",\n"
            "    \"depends_on\": \"Review reset flows\"\n"
            "  },\n"
            "  {\n"
            "    \"name\": \"Update reset handler implementation\",\n"
            "    \"description\": \"Modify the identified files to add token generation and expiry handling according to the outlined plan.\",\n"
            "    \"phase\": \"execution\",\n"
            "    \"depends_on\": \"Outline password reset changes\"\n"
            "  }\n"
            "]\n\n"
            f"Request: {prompt}\n"
            f"High-signal keywords: {keyword_list}\n\n"
            f"Project context (files, tools, and hints):\n{project_context or 'No relevant files discovered.'}\n\n"
            "Return the tasks as a JSON array. Each element must contain:\n"
            "  - name: short action title\n"
            "  - description: 1-2 sentence explanation of the action to take\n"
            "  - phase: one of ['context', 'analysis', 'execution', 'validation']\n"
            "  - depends_on: the exact name of the task whose output feeds this task (omit or null for the first task)\n"
            "  - suggested_command (optional): shell command to kick-start the task\n\n"
            "Respond with JSON only."
        )

        logging.info(
            "PlanningAgent invoking planning LLM (prompt_chars=%d, keyword_count=%d)",
            len(prompt),
            len(keywords),
        )
        try:
            logging.info(
                "LLMTaskPlanner.generate_tasks dispatching to model (prompt_chars=%d, keywords=%d)",
                len(prompt),
                len(keywords),
            )
            response = self._model.invoke(instructions)
        except Exception as exc:  # pragma: no cover - model may error at runtime
            logging.warning("PlanningAgent planning LLM call failed: %s", exc)
            return []

        content = self._extract_content(response)
        logging.info(
            "LLMTaskPlanner received response (chars=%d)", len(content) if isinstance(content, str) else -1
        )
        tasks = self._parse_task_json(content)
        logging.info("LLMTaskPlanner parsed %d task(s)", len(tasks))
        return tasks

    @property
    def model(self) -> Optional[ChatOllama]:
        return self._model

    @staticmethod
    def _extract_content(message: Any) -> str:
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for chunk in content:
                    if isinstance(chunk, dict):
                        text = chunk.get("text")
                        if text:
                            parts.append(text)
                        else:
                            parts.append(json.dumps(chunk))
                    else:
                        parts.append(str(chunk))
                return "\n".join(parts)
        return str(message)

    @staticmethod
    def _parse_task_json(raw: str) -> List[Task]:
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            return []

        snippet = raw[start : end + 1]
        try:
            payload = json.loads(snippet)
        except json.JSONDecodeError:
            return []

        allowed_phases = {"context", "analysis", "execution", "validation"}
        tasks: List[Task] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("title")
            description = item.get("description") or item.get("details")
            if not name or not description:
                continue

            phase_raw = item.get("phase") or item.get("category") or "execution"
            phase_norm = phase_raw.strip().lower() if isinstance(phase_raw, str) else "execution"
            phase = phase_norm if phase_norm in allowed_phases else "execution"

            suggested = item.get("suggested_command") or item.get("command")
            if isinstance(suggested, str):
                suggested = suggested.strip() or None
            else:
                suggested = None

            depends = item.get("depends_on") or item.get("depends") or None
            if isinstance(depends, str):
                depends = depends.strip() or None
            else:
                depends = None

            tasks.append(
                Task(
                    name=name.strip(),
                    description=description.strip(),
                    phase=phase,
                    suggested_command=suggested,
                    depends_on=depends,
                )
            )

        return tasks


class PlanRefiner:
    """Iteratively refines execution tasks using a LangChain pipeline."""

    def __init__(self, planner: LLMTaskPlanner) -> None:
        self._planner = planner
        self._model = planner.model
        if self._model is not None:
            template = PromptTemplate(
                input_variables=["prompt", "tasks_json", "project_context"],
                template=textwrap.dedent(
                    """\
Refine the ordered task list for implementing the following software request.

Request: {prompt}
Existing tasks (JSON):
{tasks_json}
Project context:
{project_context}

Guidelines:
- Ensure a strictly sequential flow; each task must depend on the previous one.
- You may add, remove, split, or merge tasks to improve clarity and correctness.
- Clarify intent and deliverables; remove redundancies and vague steps.
- Focus recommendations on files from the project context; do not introduce paths outside that list.
- Describe new dependencies generically unless already mentioned in the project context.
- Keep the plan concise (7 tasks or fewer when possible) and never exceed 10 tasks; avoid writing code.
- Add version-control safety: start by creating/switching to a dedicated feature branch (e.g., 'feat/<short-slug>').
- After each execution step, explicitly stage, commit, and push changes to that branch so incremental rollbacks are possible.
- Use phases from ['context', 'analysis', 'execution', 'validation'] as appropriate.
- Return a JSON array of task objects with keys: name, description, phase, depends_on, suggested_command (optional).

Respond with JSON only."""
                ),
            )
            self._chain = LLMChain(llm=self._model, prompt=template)
            logging.info("PlanRefiner initialised (LLM chain ready)")
        else:  # pragma: no cover - depends on runtime availability
            self._chain = None
            logging.info("PlanRefiner initialised (no LLM available)")

    def refine(self, tasks: List[Task], prompt: str, project_context: str, rounds: int) -> List[Task]:
        if rounds <= 0 or self._chain is None:
            return tasks

        current = tasks
        logging.info(
            "PlanRefiner starting (rounds=%d, initial_tasks=%d)", rounds, len(current)
        )
        model_name = getattr(self._model, "model", "unknown") if self._model else "unavailable"
        for iteration in range(1, rounds + 1):
            payload = [
                {
                    "name": task.name,
                    "description": task.description,
                    "phase": task.phase,
                    "depends_on": task.depends_on,
                    "suggested_command": task.suggested_command,
                }
                for task in current
            ]
            logging.info(
                "PlanningAgent refining task list (iteration %d/%d) with model %s",
                iteration,
                rounds,
                model_name,
            )
            try:
                chain_result = self._chain.invoke(
                    {
                        "prompt": prompt,
                        "tasks_json": json.dumps(payload, ensure_ascii=False, indent=2),
                        "project_context": project_context,
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime dependency on llm
                logging.warning("PlanningAgent refinement iteration %d failed: %s", iteration, exc)
                return current

            content = self._extract_text(chain_result)
            refined = self._planner._parse_task_json(content)  # reuse planner parsing logic
            if not refined:
                logging.warning("PlanningAgent refinement iteration %d returned no usable tasks", iteration)
                return current
            current = refined
        logging.info("PlanRefiner finished (tasks=%d)", len(current))
        return current

    @staticmethod
    def _extract_text(result: Any) -> str:
        if isinstance(result, dict):
            text = result.get("text")
            if text:
                return text
            if "output_text" in result:
                return result["output_text"]
            return json.dumps(result)
        return str(result)


class PlanningAgent:
    """Turns a natural-language feature request into a sequential task list."""

    _PATH_PATTERN = re.compile(r"(?:src|tests|app|config)[\\/][A-Za-z0-9_.\\/-]+")

    def __init__(
        self,
        llm: Optional[ChatOllama] = None,
        tool_router: Optional["ToolRouterType"] = None,
        router_config: Optional["RouterConfigType"] = None,
    ) -> None:
        self._tokenizer = RegexpTokenizer(r"[A-Za-z0-9_]+")
        self._stop_words = self._load_stopwords()
        self._llm_planner = LLMTaskPlanner(llm)
        self._refiner = PlanRefiner(self._llm_planner)
        self._project_root = pathlib.Path(__file__).resolve().parents[2]
        self._context_builder = CodeContextBuilder(self._project_root)
        self._static_tool_catalog: Optional[List[str]] = None
        self._tool_router: Optional["ToolRouterType"] = tool_router or self._create_tool_router(router_config)
        logging.info(
            "PlanningAgent initialised (project_root=%s, tool_router=%s)",
            self._project_root,
            "enabled" if self._tool_router else "disabled",
        )

    def _create_tool_router(
        self, router_config: Optional["RouterConfigType"]
    ) -> Optional["ToolRouterType"]:
        if ToolRouter is None or RouterConfig is None:
            logging.debug("PlanningAgent running without tool_router integration; dependency not installed.")
            self._static_tool_catalog = self._default_tool_catalog()
            logging.info(
                "PlanningAgent using default tool catalog (%d tool(s))",
                len(self._static_tool_catalog or []),
            )
            return None

        config: Optional["RouterConfigType"] = router_config
        if config is None:
            try:
                config = RouterConfig(repo_root=str(self._project_root))
            except Exception as exc:  # pragma: no cover - defensive for configuration errors
                logging.warning("PlanningAgent could not prepare ToolRouter config: %s", exc)
                return None

        try:
            instance = ToolRouter(config)
            logging.info("PlanningAgent ToolRouter initialised and ready")
            return instance
        except Exception as exc:  # pragma: no cover - depends on environment setup
            logging.warning("PlanningAgent could not initialise ToolRouter: %s", exc)
            self._static_tool_catalog = self._default_tool_catalog()
            logging.info(
                "PlanningAgent falling back to default tool catalog (%d tool(s))",
                len(self._static_tool_catalog or []),
            )
            return None

    def plan(self, prompt: str, refine_rounds: int = 0) -> List[Task]:
        logging.info(
            "PlanningAgent.plan started (refine_rounds=%d)", refine_rounds
        )
        clean_prompt = prompt.strip()
        logging.info("Planning prompt length=%d", len(clean_prompt))
        available_tools = self._list_available_tools()
        logging.info("Available tools discovered: %d", len(available_tools))
        keywords = self._extract_keywords(clean_prompt)
        logging.info("Extracted keywords: %s", ", ".join(keywords) if keywords else "<none>")
        matches = self._context_builder.build_snapshot(keywords)
        logging.info("Context matches found: %d", len(matches))
        project_context = self._context_builder.format_for_prompt(matches)
        tools_context = self._format_tools_for_prompt(available_tools)
        if tools_context:
            if project_context:
                project_context = f"{project_context}\n\nAvailable tools:\n{tools_context}"
            else:
                project_context = f"Available tools:\n{tools_context}"

        # Generate the entire plan using the Ollama model only (no heuristics).
        logging.info("Invoking LLM planner to generate tasks...")
        action_tasks = self._llm_planner.generate_tasks(
            clean_prompt, keywords, project_context
        )

        if not action_tasks:
            logging.warning("PlanningAgent received no tasks from LLM planner.")
            return []

        # Optionally refine with the LLM for better sequencing or clarity.
        tasks = list(action_tasks)
        logging.info("Initial LLM task count: %d", len(tasks))
        max_refinements = max(0, refine_rounds)
        verified, reason = self._verify_plan(tasks)
        logging.info(
            "Initial plan verification: %s%s",
            "ok" if verified else "failed",
            "" if verified else f" ({reason})",
        )
        if self._refiner is not None:
            refinements_used = 0
            # If verification fails and no refinements requested, try one best-effort pass.
            extra_first_pass = max_refinements == 0 and not verified
            total_allowed = max_refinements + (1 if extra_first_pass else 0)
            while refinements_used < total_allowed and (not verified):
                refinements_used += 1
                logging.info(
                    "PlanningAgent running refinement pass (%d/%d): %s",
                    refinements_used,
                    total_allowed,
                    reason or "verification failed",
                )
                tasks = self._refiner.refine(
                    tasks,
                    clean_prompt,
                    project_context,
                    rounds=1,
                )
                verified, reason = self._verify_plan(tasks)

        if not verified:
            logging.warning(
                "PlanningAgent returning plan that failed verification: %s", reason
            )
        else:
            logging.info("Final plan verification ok (%d task(s))", len(tasks))

        # Inject branch/commit/push requirements into execution steps without adding new tasks.
        try:
            branch = self._feature_branch_name(prompt=clean_prompt, keywords=keywords)
            tasks = self._inject_branch_commit_requirements(tasks, branch)
        except Exception as exc:  # pragma: no cover - defensive; should not block planning
            logging.warning("PlanningAgent could not inject branch/commit requirements: %s", exc)

        # Write out a sequential JSON file representation of the plan.
        try:
            self._export_sequential_plan(tasks)
        except Exception as exc:  # pragma: no cover - filesystem could be readonly
            logging.warning("PlanningAgent could not write plan.json: %s", exc)

        return tasks

    def _feature_branch_name(self, prompt: str, keywords: List[str]) -> str:
        """Derive a concise feature branch name from keywords/prompt.

        Falls back to 'feat/change' when no signal is available.
        """
        base_source = "-".join(keywords) if keywords else prompt.lower()
        # Replace non-alphanumeric with hyphens; collapse and trim.
        cleaned = re.sub(r"[^a-z0-9]+", "-", base_source.lower())
        cleaned = re.sub(r"-+", "-", cleaned).strip("-")
        if not cleaned:
            cleaned = "change"
        # Keep reasonably short to avoid long branch names.
        cleaned = cleaned[:40].strip("-") or "change"
        return f"feat/{cleaned}"

    def _inject_branch_commit_requirements(self, tasks: List[Task], branch: str) -> List[Task]:
        """Append branch/commit/push requirements to each execution step.

        - First execution task: also instruct to create/switch to the feature branch.
        - Every execution task: instruct to stage, commit, and push to the branch.
        """
        first_exec_index: Optional[int] = None
        for idx, task in enumerate(tasks):
            if task.phase == "execution":
                first_exec_index = idx
                break
        if first_exec_index is not None:
            first_exec = tasks[first_exec_index]
            prefix = (
                f"Create or switch to feature branch '{branch}' to isolate changes. "
            )
            if prefix.strip() not in first_exec.description:
                first_exec.description = f"{prefix}{first_exec.description}".strip()

        commit_suffix = (
            f" After completing this step, stage, commit, and push to '{branch}' to enable rollbacks."
        )
        for task in tasks:
            if task.phase == "execution":
                # Avoid duplicating if already present.
                if "stage, commit, and push" not in task.description.lower():
                    task.description = f"{task.description.rstrip()}" + commit_suffix
        return tasks

    def _build_context_tasks(
        self,
        prompt: str,
        keywords: List[str],
        matches: Sequence[CodeMatch],
        available_tools: Optional[Sequence[str]] = None,
    ) -> List[Task]:
        keyword_phrase = self._format_keyword_phrase(keywords)
        search_command = self._search_command(keywords)
        highlight = self._context_builder.highlight_summary(matches)
        tasks = [
            Task(
                name="Survey repository layout",
                description=(
                    "List top-level directories and files (README.md, pyproject.toml, src/) and record them in a scratch "
                    "note so downstream steps understand which areas of the repo are relevant."
                ),
                phase="context",
                suggested_command="ls",
            ),
            Task(
                name="Inspect core package",
                description=(
                    "Using the scratch note, drill into src/reagent to catalogue modules, entry points, and utilities the "
                    "signup flow touches, appending these details to the note."
                ),
                phase="context",
                depends_on="Survey repository layout",
                suggested_command="ls src/reagent",
            ),
            Task(
                name="Search for relevant code paths",
                description=(
                    "Leverage the module inventory captured in the previous step to craft ripgrep patterns, storing the "
                    f"matching paths for {keyword_phrase} alongside the note so the review step has a concrete checklist."
                ),
                phase="context",
                depends_on="Inspect core package",
                suggested_command=search_command if search_command else 'rg "<keyword>" src',
            ),
            Task(
                name="Review candidate files",
                description=(
                    "Walk each path from the stored checklist to capture responsibilities, key functions, and integration "
                    f"seams, producing annotated notes that will drive implementation. {highlight}."
                ),
                phase="context",
                depends_on="Search for relevant code paths",
            ),
            Task(
                name="Summarize required changes",
                description=(
                    f"Convert the annotated notes into a concrete change list for \"{prompt}\", including the risks, "
                    "dependencies, and questions identified during review."
                ),
                phase="analysis",
                depends_on="Review candidate files",
            ),
        ]
        if available_tools:
            summary = self._summarize_tools(available_tools)
            tool_task = Task(
                name="Review available automation tools",
                description=(
                    "Call tool_router to enumerate accessible automation helpers and capture their capabilities for the "
                    f"upcoming plan. Tools detected: {summary}."
                ),
                phase="context",
            )
            tasks.insert(0, tool_task)
            if len(tasks) > 1:
                tasks[1].depends_on = tool_task.name
        return tasks

    def _build_tool_anchored_plan(
        self,
        prompt: str,
        keywords: List[str],
        matches: Sequence[CodeMatch],
        available_tools: Sequence[str],
    ) -> List[Task]:
        tool_groups = self._group_tools_by_namespace(available_tools)
        focus_files = ", ".join(match.path for match in matches[:3]) if matches else "the relevant modules"
        highlight = self._context_builder.highlight_summary(matches)
        summary = self._summarize_tools(available_tools)

        fs_list_tool = self._select_tool(tool_groups.get("fs", []), ["list", "stat"])
        fs_read_tool = self._select_tool(tool_groups.get("fs", []), ["read"])
        fs_write_tool = self._select_tool(tool_groups.get("fs", []), ["write", "apply_patch"])
        fs_ensure_tool = self._select_tool(tool_groups.get("fs", []), ["ensure", "dir"])
        git_status_tool = self._select_tool(tool_groups.get("git", []), ["status"])
        git_commit_tool = self._select_tool(tool_groups.get("git", []), ["commit"])
        github_tool = self._select_tool(tool_groups.get("github", []), ["pull", "issue", "comment"])

        context_hint = (
            f"use tool_router.route_name('{fs_list_tool}', path='src')" if fs_list_tool else "enumerate directories"
        )

        # Sample files to anchor suggested commands
        sample_paths = [m.path for m in matches[:2]] if matches else ["README.md"]
        sample_read = sample_paths[0]
        sample_write = sample_paths[0] if sample_paths else "<target>"

        # Tailor steps if the prompt looks code-related
        is_code = self._is_code_issue(prompt, keywords)

        tasks: List[Task] = []
        tasks.append(
            Task(
                name="Map repo and tools via tool_router",
                description=(
                    f"Call tool_router.list_tools to catalogue automation ({summary}), then {context_hint} and document "
                    f"which assets support \"{prompt}\". {highlight}"
                ),
                phase="context",
                suggested_command=(
                    f"tool_router.route_name('{fs_list_tool}', path='src')" if fs_list_tool else None
                ),
            )
        )

        # Code-focused audit of matched files
        tasks.append(
            Task(
                name=("Audit suspect code paths with fs tools" if is_code else "Audit relevant files with fs tools"),
                description=(
                    (
                        f"Open and review matched files (e.g., {focus_files}) using "
                        f"{f'`{fs_read_tool}`' if fs_read_tool else 'fs.read_*'} to confirm the behaviour related to \"{prompt}\"."
                    )
                    if is_code
                    else (
                        f"Inspect key files and configs (e.g., {focus_files}) using "
                        f"{f'`{fs_read_tool}`' if fs_read_tool else 'fs.read_*'} to capture current behaviour relevant to \"{prompt}\"."
                    )
                ),
                phase="context",
                depends_on="Map repo and tools via tool_router",
                suggested_command=(
                    f"tool_router.route_name('{fs_read_tool}', path='{sample_read}')" if fs_read_tool else None
                ),
            )
        )

        tasks.append(
            Task(
                name=("Outline fix/change strategy" if is_code else "Outline required changes for request"),
                description=(
                    f"Summarise findings into a concrete fix plan for \"{prompt}\", noting functions/classes to change across {focus_files}."
                    if is_code
                    else f"Summarise the audit into a concrete change list for \"{prompt}\", mapping edits across {focus_files} and highlighting where automation helps."
                ),
                phase="analysis",
                depends_on=("Audit suspect code paths with fs tools" if is_code else "Audit relevant files with fs tools"),
            )
        )

        tasks.append(
            Task(
                name=("Implement code changes with fs automation" if is_code else "Apply changes with fs automation"),
                description=(
                    f"Implement the plan using {f'`{fs_write_tool}`' if fs_write_tool else 'fs.write_*'}; create directories via "
                    f"{f'`{fs_ensure_tool}`' if fs_ensure_tool else 'fs.ensure_*'} if needed."
                ),
                phase="execution",
                depends_on=("Outline fix/change strategy" if is_code else "Outline required changes for request"),
                suggested_command=(
                    f"tool_router.route_name('{fs_write_tool}', path='{sample_write}', content='<updated content>')" if fs_write_tool else None
                ),
            )
        )

        tasks.append(
            Task(
                name=("Add/update tests and documentation" if is_code else "Update tests and documentation"),
                description=(
                    "Extend or adjust tests, fixtures, and docs to reflect the changes, persisting updates with "
                    f"{f'`{fs_write_tool}`' if fs_write_tool else 'fs.write_*'}."
                ),
                phase="execution",
                depends_on=("Implement code changes with fs automation" if is_code else "Apply changes with fs automation"),
            )
        )

        tasks.append(
            Task(
                name="Validate and finalise with git/GitHub",
                description=(
                    f"Run tests, inspect diffs, and prepare commits using "
                    f"{f'`{git_status_tool}`' if git_status_tool else 'git status'} and "
                    f"{f'`{git_commit_tool}`' if git_commit_tool else 'git commit'}; track follow-ups with "
                    f"{f'`{github_tool}`' if github_tool else 'GitHub tools'} if available."
                ),
                phase="validation",
                depends_on=("Add/update tests and documentation" if is_code else "Update tests and documentation"),
                suggested_command=f"tool_router.route_name('{git_status_tool}')" if git_status_tool else None,
            )
        )

        return tasks

    def _build_action_tasks(
        self,
        prompt: str,
        keywords: List[str],
        matches: Sequence[CodeMatch],
        available_tools: Optional[Sequence[str]] = None,
    ) -> List[Task]:
        focus_terms = keywords if keywords else ["feature"]
        focus_files = ", ".join(match.path for match in matches[:3]) if matches else "the inspected modules"
        tool_hint = self._tool_usage_hint(available_tools)
        tasks: List[Task] = [
            Task(
                name="Shape execution outline",
                description=(
                    f"Transform the consolidated findings into a step-by-step plan for \"{prompt}\" inside notes/planning.md, "
                    f"mapping the required edits across {focus_files} with owners, assumptions, and risks."
                    + (f" {tool_hint}" if tool_hint else "")
                ),
                phase="analysis",
                depends_on="Summarize required changes",
            )
        ]

        previous_update = "Shape execution outline"
        for term in focus_terms:
            tasks.append(
                Task(
                    name=f"Update components for '{term}'",
                    description=(
                        f"Apply the outlined plan to modify the '{term}'-related code within {focus_files}, reuse utilities "
                        "revealed during file review, and tick the corresponding checklist items."
                    ),
                    phase="execution",
                    depends_on=previous_update,
                )
            )
            previous_update = tasks[-1].name

        tasks.append(
            Task(
                name="Refresh automated coverage",
                description=(
                    "Extend or adjust tests, fixtures, and documentation to reflect the updated components, recording "
                    "evidence in notes/planning.md."
                ),
                phase="execution",
                depends_on=previous_update if focus_terms else "Shape execution outline",
            )
        )
        tasks.append(
            Task(
                name="Run validation suite",
                description="Execute project checks to confirm the updates integrate cleanly.",
                phase="validation",
                suggested_command=self._validation_command(),
                depends_on="Refresh automated coverage",
            )
        )
        tasks.append(
            Task(
                name="Review and prepare for handoff",
                description="Inspect diffs, document notable decisions, and ready the branch for review or deployment.",
                phase="validation",
                suggested_command="git status -sb",
                depends_on="Run validation suite",
            )
        )
        return tasks

    def _assemble_plan(
        self,
        context_tasks: Sequence[Task],
        action_tasks: Sequence[Task],
        matches: Sequence[CodeMatch],
    ) -> List[Task]:
        final_execution = self._ensure_quality_tasks(list(action_tasks), matches)
        combined = list(context_tasks) + self._deduplicate_tasks(final_execution)
        normalized = self._normalize_task_names(combined)
        return self._enforce_dependency_chain(normalized)

    def _list_available_tools(self) -> List[str]:
        if self._tool_router is None:
            catalog = list(self._static_tool_catalog or [])
            if catalog:
                logging.info(
                    "PlanningAgent using default tool catalog with %d tool(s).", len(catalog)
                )
            return catalog
        try:
            tools = list(self._tool_router.list_tools())
            logging.info(
                "PlanningAgent retrieved %d tool(s) from tool_router.", len(tools)
            )
            return tools
        except Exception as exc:  # pragma: no cover - runtime dependency on tool_router
            logging.warning("PlanningAgent could not fetch available tools: %s", exc)
            return []

    @staticmethod
    def _format_tools_for_prompt(tools: Sequence[str]) -> str:
        if not tools:
            return ""
        return "\n".join(f"- {tool}" for tool in tools)

    @staticmethod
    def _summarize_tools(tools: Sequence[str], limit: int = 5) -> str:
        if not tools:
            return "none"
        if len(tools) <= limit:
            return ", ".join(tools)
        remaining = len(tools) - limit
        displayed = ", ".join(tools[:limit])
        suffix = "tool" if remaining == 1 else "tools"
        return f"{displayed}, and {remaining} more {suffix}"

    def _tool_usage_hint(self, tools: Optional[Sequence[str]]) -> str:
        if not tools:
            return ""
        summary = self._summarize_tools(tools)
        return (
            f"Note how the available tools ({summary}) can automate discovery, editing, or validation as you refine the plan."
        )

    @staticmethod
    def _verify_plan(tasks: Sequence[Task]) -> tuple[bool, str]:
        if not tasks:
            return False, "plan contains no tasks"
        if len(tasks) > 10:
            return False, f"plan contains {len(tasks)} tasks which exceeds limit of 10"
        allowed_phases = {"context", "analysis", "execution", "validation"}
        seen_names: set[str] = set()
        has_execution = False
        for index, task in enumerate(tasks):
            if task.name in seen_names:
                return False, f"duplicate task name '{task.name}'"
            if task.phase not in allowed_phases:
                return False, f"task '{task.name}' has unexpected phase '{task.phase}'"
            if index == 0 and task.depends_on:
                return False, "first task must not depend on another task"
            if index > 0 and task.depends_on and task.depends_on not in seen_names:
                return False, f"task '{task.name}' depends on unknown task '{task.depends_on}'"
            if task.phase == "execution":
                has_execution = True
            seen_names.add(task.name)
        if not has_execution:
            return False, "plan is missing an execution-phase task"
        return True, ""

    @staticmethod
    def _group_tools_by_namespace(tools: Sequence[str]) -> dict[str, List[str]]:
        grouped: dict[str, List[str]] = {}
        for tool in tools:
            namespace, _, _ = tool.partition(".")
            grouped.setdefault(namespace, []).append(tool)
        return grouped

    @staticmethod
    def _select_tool(tools: Sequence[str], preferred_keywords: Optional[Sequence[str]] = None) -> Optional[str]:
        if not tools:
            return None
        if preferred_keywords:
            for keyword in preferred_keywords:
                for tool in tools:
                    if keyword.lower() in tool.lower():
                        return tool
        return tools[0]

    @staticmethod
    def _is_code_issue(prompt: str, keywords: Sequence[str]) -> bool:
        text = (prompt or "").lower()
        code_markers = [
            "traceback",
            "exception",
            "stack",
            "error",
            "bug",
            "fail",
            "test",
            "unit",
            "function",
            "class",
            "module",
            ".py",
        ]
        if any(m in text for m in code_markers):
            return True
        return any(k in {"bug", "error", "exception", "test", "refactor", "fix"} for k in keywords)

    @staticmethod
    def _default_tool_catalog() -> List[str]:
        return [
            "fs.list_dir",
            "fs.read_file",
            "fs.write_file",
            "fs.ensure_dir",
            "fs.delete",
            "fs.stat",
            "git.status",
            "git.add",
            "git.commit",
            "git.branch",
            "git.checkout",
            "git.push",
            "github.create_issue",
            "github.search_issues",
        ]

    def _ensure_quality_tasks(self, tasks: List[Task], matches: Sequence[CodeMatch]) -> List[Task]:
        has_validation = any(task.phase == "validation" for task in tasks)
        has_review = any(
            task.phase == "validation" and ("review" in task.name.lower() or "handoff" in task.name.lower())
            for task in tasks
        )

        if has_validation and has_review:
            return tasks

        focus_files = ", ".join(match.path for match in matches[:3]) if matches else "the updated modules"
        augmented = list(tasks)

        if not has_validation:
            augmented.append(
                Task(
                    name="Validate analytics tracking end-to-end",
                    description=(
                        "Run automated checks and a manual signup flow to confirm analytics events fire correctly for "
                        f"{focus_files}."
                    ),
                    phase="validation",
                    suggested_command=self._validation_command(),
                    depends_on=tasks[-1].name if tasks else None,
                )
            )

        if not has_review:
            augmented.append(
                Task(
                    name="Prepare change for review",
                    description=(
                        "Summarise analytics instrumentation decisions, update documentation, and ready the branch for "
                        "review."
                    ),
                    phase="validation",
                    suggested_command="git status -sb",
                    depends_on=augmented[-1].name if augmented else None,
                )
            )

        return augmented

    def _deduplicate_tasks(self, tasks: List[Task]) -> List[Task]:
        seen: set[tuple[str, str]] = set()
        unique: List[Task] = []
        for task in tasks:
            key = (task.name.strip().lower(), task.phase)
            if key in seen:
                continue
            seen.add(key)
            unique.append(task)
        return unique

    def _normalize_task_names(self, tasks: List[Task]) -> List[Task]:
        for task in tasks:
            cleaned = task.name.strip()
            cleaned = re.sub(r":\d+$", "", cleaned)
            if cleaned != task.name:
                task.name = cleaned
        return tasks

    def _enforce_dependency_chain(self, tasks: List[Task]) -> List[Task]:
        seen_names = {task.name for task in tasks}
        previous_name: Optional[str] = None
        for task in tasks:
            if previous_name is None:
                task.depends_on = task.depends_on if task.depends_on in seen_names else None
            else:
                if not task.depends_on or task.depends_on not in seen_names:
                    task.depends_on = previous_name
                if task.depends_on and task.depends_on.lower() not in task.description.lower():
                    task.description = (
                        f"{task.description.rstrip()} (Consumes the output from \"{task.depends_on}\".)"
                    )
            previous_name = task.name
        return tasks

    def _extract_paths(self, text: str) -> set[str]:
        if not text:
            return set()
        matches = self._PATH_PATTERN.findall(text)
        return {match.replace("\\", "/") for match in matches}

    def _extract_keywords(self, prompt: str, limit: int = 3) -> List[str]:
        words = self._tokenizer.tokenize(prompt.lower())
        stop_words = self._stop_words
        keywords: List[str] = []
        for word in words:
            if len(word) <= 2:
                continue
            if word in stop_words or word.isdigit():
                continue
            if word not in keywords:
                keywords.append(word)
            if len(keywords) >= limit:
                break
        return keywords

    def _load_stopwords(self) -> set[str]:
        """Load NLTK English stopwords with a safe fallback.

        Tries to use the NLTK corpus; if unavailable, attempts a quiet download.
        Falls back to a minimal set if resources remain unavailable.
        """
        try:
            return set(nltk_stopwords.words("english"))
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                return set(nltk_stopwords.words("english"))
            except Exception:
                logging.warning(
                    "NLTK stopwords not available; using minimal fallback list."
                )
                return {
                    "the",
                    "and",
                    "for",
                    "with",
                    "that",
                    "this",
                    "from",
                    "have",
                    "should",
                    "like",
                    "make",
                    "would",
                    "could",
                    "there",
                    "please",
                }

    @staticmethod
    def _search_command(keywords: List[str]) -> Optional[str]:
        if not keywords:
            return None
        if len(keywords) == 1:
            return f'rg "{keywords[0]}" src'
        patterns = " ".join(f'-e "{word}"' for word in keywords)
        return f"rg {patterns} src"

    @staticmethod
    def _validation_command() -> str:
        return "pytest && git status -sb"

    @staticmethod
    def _format_keyword_phrase(keywords: List[str]) -> str:
        if not keywords:
            return "the request"
        if len(keywords) == 1:
            return f"'{keywords[0]}'"
        if len(keywords) == 2:
            return f"'{keywords[0]}' and '{keywords[1]}'"
        return f"'{keywords[0]}', '{keywords[1]}', and '{keywords[2]}'"

    def _export_sequential_plan(self, tasks: List[Task], path: Optional[pathlib.Path] = None) -> pathlib.Path:
        """Write the plan to a JSON file as a sequential list with step numbers.

        The file is written to repo root as 'plan.json' by default.
        """
        output_path = path or (self._project_root / "plan.json")
        payload = [
            {
                "step": index,
                "name": task.name,
                "description": task.description,
                "phase": task.phase,
                "depends_on": task.depends_on,
                "suggested_command": task.suggested_command,
            }
            for index, task in enumerate(tasks, start=1)
        ]
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        output_path.write_text(text, encoding="utf-8")
        logging.info("PlanningAgent wrote sequential JSON plan to %s", output_path)
        return output_path


def _print_plan(prompt: str, plan: List[Task]) -> None:
    print(f'Plan for: "{prompt}"\n')
    for index, task in enumerate(plan, start=1):
        print(f"{index}. {task.name}")
        print(f"   Phase: {task.phase}")
        if task.depends_on:
            print(f"   Depends on: {task.depends_on}")
        print(f"   {task.description}")
        if task.suggested_command:
            print(f"   Command hint: {task.suggested_command}")
        print()


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse as _argparse

    parser = _argparse.ArgumentParser(
        description="Generate an implementation plan using the Ollama-backed planner."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Natural-language description of the feature to plan. If omitted, uses a sample query.",
    )
    parser.add_argument(
        "--refine",
        type=int,
        default=1,
        help="Refinement rounds to run with the LLM (default: 1).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    prompt_text = " ".join(args.prompt).strip() if args.prompt else ""
    if not prompt_text:
        prompt_text = "Design a Login page using only HTML and JavaScript"
        logging.info("No prompt provided; using sample query: %s", prompt_text)

    agent = PlanningAgent()
    plan = agent.plan(prompt_text, refine_rounds=args.refine)
    _print_plan(prompt_text, plan)


if __name__ == "__main__":
    main()

