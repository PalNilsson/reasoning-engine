#!/usr/bin/env python3
"""
Test harness for PanDAReasoningEngine using configurable prompts.

This script:

  * Loads a JSON file mapping questions to expected handler names.
  * Runs each question through PanDAReasoningEngine.
  * Compares the selected handler with the expected one.
  * Writes detailed per-test output either to stdout or to a file.
  * Always prints a summary report to stdout.
  * Exits with a non-zero status code if any test fails.

Usage:
    python test_panda_reasoning_engine.py --input prompts.json
    python test_panda_reasoning_engine.py --input prompts.json --output results.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, TextIO, List

from panda_reasoning_engine import PanDAReasoningEngine, InteractionResult


class BaseDummyHandler:
    """Base dummy handler that echoes what it received.

    Subclasses should only override the ``name`` attribute to identify
    themselves. All of them emulate a PanDA handler's ``handle_request``
    interface without performing any real work.
    """

    name: str = "BaseDummyHandler"

    def handle_request(
        self,
        *,
        prompt: str,
        entities: Dict[str, Any],
        goal: str,
        confidence: float,
    ) -> str:
        """Return a debug-style string describing how the handler was called.

        Args:
            prompt: User prompt text that was passed to the handler.
            entities: Extracted entities from the reasoning engine.
            goal: High-level goal identified by the reasoning engine.
            confidence: Confidence score in the routing decision.

        Returns:
            str: Human-readable description of the handler call.
        """
        return (
            f"[{self.name}] Handling request\n"
            f"  goal: {goal}\n"
            f"  confidence: {confidence:.2f}\n"
            f"  entities: {entities}\n"
            f"  prompt: {prompt!r}"
        )


class DocumentQuery(BaseDummyHandler):
    """Dummy handler emulating documentation / how-to queries."""

    name = "DocumentQuery"


class QueueQuery(BaseDummyHandler):
    """Dummy handler emulating queue and site status queries."""

    name = "QueueQuery"


class TaskQuery(BaseDummyHandler):
    """Dummy handler emulating task / job status queries."""

    name = "TaskQuery"


class LogAnalysis(BaseDummyHandler):
    """Dummy handler emulating log and failure analysis."""

    name = "LogAnalysis"


class PilotMonitor(BaseDummyHandler):
    """Dummy handler emulating pilot monitoring queries."""

    name = "PilotMonitor"


class MetadataAnalysis(BaseDummyHandler):
    """Dummy handler emulating metadata-level analysis for tasks and jobs."""

    name = "MetadataAnalysis"

class PandaMCP(BaseDummyHandler):
    """Dummy handler emulating PandaMCP."""

    name = "PandaMCP"


class Selection:
    """Dummy selection component that may refine the chosen handler.

    In this test harness, the selection component simply returns the
    heuristic candidate by default. The implementation can be extended to
    override routing decisions for specific phrases if desired.
    """

    def select_handler(
        self,
        *,
        prompt: str,
        heuristic_candidate: str,
        entities: Dict[str, Any],
    ) -> Dict[str, str]:
        """Return the handler name to use.

        Args:
            prompt: Original user prompt text.
            heuristic_candidate: Handler name suggested by heuristics.
            entities: Extracted entities from the reasoning engine.

        Returns:
            Dict[str, str]: Dictionary with a ``handler_name`` key indicating
            which handler should be used.
        """
        # Example override hook:
        # if "force metadata" in prompt.lower():
        #     return {"handler_name": "MetadataAnalysis"}
        return {"handler_name": heuristic_candidate}


def load_prompts(path: str) -> Dict[str, str]:
    """Load a mapping of prompts to expected handlers from a JSON file.

    The JSON file is expected to contain a single object mapping each
    question string to the expected handler name, for example:

    .. code-block:: json

        {
          "What is PanDA?": "DocumentQuery",
          "Why did job 123456 fail?": "LogAnalysis"
        }

    Args:
        path: Path to the JSON file.

    Returns:
        Dict[str, str]: Mapping from question to expected handler name.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        TypeError: If the JSON does not contain a top-level object of string
            keys and string values.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("Input JSON must contain a top-level object (mapping).")

    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("All keys and values in the input JSON must be strings.")

    return data


def run_tests(
    engine: PanDAReasoningEngine,
    prompts: Dict[str, str],
    out: TextIO,
) -> int:
    """Run all prompts through the reasoning engine and write detailed output.

    Args:
        engine: Configured PanDAReasoningEngine instance to test.
        prompts: Mapping from question text to expected handler name.
        out: Text stream to which detailed per-test output will be written.
            This may be ``sys.stdout`` or a file-like object.

    Returns:
        int: Number of failed tests.
    """
    total = len(prompts)
    failed = 0

    print("\n=== PanDAReasoningEngine Automated Test ===\n", file=out)

    for i, (prompt, expected_handler) in enumerate(prompts.items(), start=1):
        print(f"\n--- Test {i}/{total} ---", file=out)
        print(f"Prompt: {prompt!r}", file=out)
        print(f"Expected handler: {expected_handler}", file=out)

        result: InteractionResult = engine.handle_text(prompt)
        actual_handler = result.reasoning.handler_name

        print(f"Actual handler:   {actual_handler}", file=out)

        if actual_handler == expected_handler:
            print("Result: ✓ PASS", file=out)
        else:
            print("Result: ✗ FAIL", file=out)
            failed += 1

        print("\n[Perception]", file=out)
        print(f"  intent:   {result.perception.intent}", file=out)
        print(f"  entities: {result.perception.entities}", file=out)

        print("\n[Reasoning]", file=out)
        print(f"  goal:       {result.reasoning.goal}", file=out)
        print(f"  handler:    {actual_handler}", file=out)
        print(f"  confidence: {result.reasoning.confidence:.2f}", file=out)

        print("\n[Formatted Answer]", file=out)
        print(result.formatted_answer, file=out)

        print("\n" + "-" * 72, file=out)

    print("\n=== Test run complete ===", file=out)
    print(f"Total tests:   {total}", file=out)
    print(f"Passed tests:  {total - failed}", file=out)
    print(f"Failed tests:  {failed}", file=out)

    return failed


def build_engine() -> PanDAReasoningEngine:
    """Construct a PanDAReasoningEngine wired to dummy handlers.

    Returns:
        PanDAReasoningEngine: Engine instance configured with dummy handlers
        and a dummy Selection component.
    """
    document_query = DocumentQuery()
    queue_query = QueueQuery()
    task_query = TaskQuery()
    log_analysis = LogAnalysis()
    pilot_monitor = PilotMonitor()
    metadata_analysis = MetadataAnalysis()
    panda_mcp = PandaMCP()
    selection = Selection()

    engine = PanDAReasoningEngine(
        document_query=document_query,
        queue_query=queue_query,
        task_query=task_query,
        log_analysis=log_analysis,
        pilot_monitor=pilot_monitor,
        metadata_analysis=metadata_analysis,
        selection=selection,
        panda_mcp=panda_mcp,
        stt_callable=None,  # audio not tested in this harness
    )
    return engine


def _step_to_tool_name(step: str) -> Optional[str]:
    """Map high-level plan step names to concrete tool names.

    The returned name should match the tools namespace used in your
    fine-tuning schema, e.g. 'functions.metadata_search'.

    Steps that do not correspond to an actual tool call should return None.
    """
    mapping = {
        # Documentation / context retrieval
        "retrieve_relevant_docs": "functions.context_retrieve",
        # Metadata / status lookups
        "fetch_metadata": "functions.metadata_search",
        "fetch_task_metadata": "functions.metadata_search",
        "fetch_task_job_summary": "functions.metadata_search",
        "identify_failing_jobs": "functions.metadata_search",
        # Log queries
        "fetch_logs_for_jobs": "functions.log_query",
        # You can extend this dictionary as your toolset grows
    }
    return mapping.get(step)


def build_conversation_for_prompt(
    engine: PanDAReasoningEngine,
    prompt: str,
) -> Dict[str, Any]:
    """Construct a synthetic conversation object for a given user prompt.

    The returned dict has a single key 'messages' containing a list of
    message dicts suitable for JSONL training.
    """
    interaction = engine.handle_text(prompt)
    plan = interaction.reasoning.plan
    handler_name = interaction.reasoning.handler_name

    messages: List[Dict[str, Any]] = []

    # 1) System message
    system_content = (
        "You are the AskPanDA Triage Assistant. You help ATLAS users diagnose "
        "and understand PanDA tasks and jobs, including queue status and pilot "
        "issues.\n"
        "Knowledge cutoff: 2024-06\n"
        "Reasoning: high\n"
        "# Valid channels: analysis, commentary, final.\n"
        "Calls to these tools must go to the commentary channel: 'functions'."
    )
    messages.append(
        {
            "role": "system",
            "content": system_content,
        }
    )

    # 2) Developer message
    developer_content = (
        "# Instructions\n"
        "You are the AskPanDA Triage Assistant. Use the provided tool "
        "functions to look up PanDA metadata, retrieve context, and inspect "
        "logs before answering.\n"
        "# Tools\n"
        "## functions\n"
        "namespace functions {\n"
        "  // (Schemas are defined externally; the tool names are:\n"
        "  //  - metadata_search\n"
        "  //  - context_retrieve\n"
        "  //  - log_query\n"
        "}\n"
    )
    messages.append(
        {
            "role": "developer",
            "content": developer_content,
        }
    )

    # 3) User message
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # 4) Tool calls derived from the reasoning plan
    # For each step in the plan that maps to a tool, we emit:
    #   - assistant (commentary) message with name + empty arguments
    #   - tool message with empty content
    for step in plan:
        tool_name = _step_to_tool_name(step)
        if not tool_name:
            continue

        messages.append(
            {
                "role": "assistant",
                "channel": "commentary",
                "name": tool_name,
                "arguments": {},  # arguments unknown at data-generation time
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "content": "",  # tool output omitted / unknown
            }
        )

    # 5) Final assistant message
    # You can choose either the raw handler_output or the formatted_answer.
    # formatted_answer includes routing and plan; handler_output is cleaner.
    final_content = interaction.handler_output

    messages.append(
        {
            "role": "assistant",
            "channel": "final",
            "content": final_content,
        }
    )

    return {"messages": messages}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the test harness.

    Args:
        argv: Optional list of argument strings. If omitted, ``sys.argv[1:]``
            will be used.

    Returns:
        argparse.Namespace: Parsed arguments with ``input``, ``output`` and
        ``jsonl_output`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Run automated tests for PanDAReasoningEngine.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON file mapping prompts to expected handler names.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional path to a file where detailed per-test output will be "
            "written. If omitted, output is written to stdout."
        ),
    )
    parser.add_argument(
        "--jsonl-output",
        help=(
            "Optional path to a JSONL file where synthetic fine-tuning "
            "conversations will be written. Each line will contain a single "
            "JSON object with a 'messages' array."
        ),
    )
    return parser.parse_args(argv)



def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the test harness.

    This function:

      * Parses command-line arguments.
      * Loads prompts from the JSON input file.
      * Builds a dummy-configured PanDAReasoningEngine.
      * Runs tests and writes detailed output to either stdout or a file.
      * Prints a summary to stdout.
      * Exits with a non-zero status if any test failed.

    Args:
        argv: Optional list of argument strings. If omitted, ``sys.argv[1:]``
            will be used.
    """
    args = parse_args(argv)

    try:
        prompts = load_prompts(args.input)
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading input file '{args.input}': {exc}", file=sys.stderr)
        sys.exit(1)

    engine = build_engine()

    # Run tests, writing detailed logs to the requested destination.
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as out_f:
                failed = run_tests(engine, prompts, out_f)
        except OSError as exc:  # noqa: BLE001
            print(
                f"Error opening output file '{args.output}': {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        failed = run_tests(engine, prompts, sys.stdout)

    # Optionally generate JSONL dataset for fine-tuning
    if args.jsonl_output:
        try:
            with open(args.jsonl_output, "w", encoding="utf-8") as jsonl_f:
                for prompt in prompts.keys():
                    conv = build_conversation_for_prompt(engine, prompt)
                    json.dump(conv, jsonl_f, ensure_ascii=False)
                    jsonl_f.write("\n")
        except OSError as exc:  # noqa: BLE001
            print(
                f"Error writing JSONL file '{args.jsonl_output}': {exc}",
                file=sys.stderr,
            )
            # Do not override test failure exit code, but fail if we cannot
            # produce the dataset.
            sys.exit(1)

    # Always print a short summary report to stdout
    total = len(prompts)
    print("\n=== Summary (stdout) ===")
    print(f"Total tests:   {total}")
    print(f"Passed tests:  {total - failed}")
    print(f"Failed tests:  {failed}")

    # Exit with non-zero status if there were failures
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
