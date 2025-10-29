from __future__ import annotations

import argparse
import logging
from typing import Iterable, List, Optional

from .agent import PlanningAgent, Task


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an implementation plan for a product request."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Natural-language description of the feature you want to build.",
    )
    parser.add_argument(
        "--show-tool-output",
        action="store_true",
        help="Execute tools referenced in the plan and display their output.",
    )
    parser.add_argument(
        "--refine",
        type=int,
        default=0,
        help="Number of refinement passes to run with the LLM (default: 0).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _collect_prompt(cli_prompt_parts: List[str]) -> str:
    prompt = " ".join(cli_prompt_parts).strip()
    if prompt:
        return prompt

    try:
        prompt = input("What feature would you like to plan? ").strip()
    except EOFError:
        prompt = ""

    return prompt


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


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    prompt = _collect_prompt(args.prompt)

    if not prompt:
        print("No prompt supplied. Exiting.")
        return

    agent = PlanningAgent()
    plan = agent.plan(prompt, refine_rounds=args.refine)
    _print_plan(prompt, plan)

    if args.show_tool_output:
        print("No automated tool invocations are configured for this planner.")
