"""
PanDA reasoning engine module.

This module implements a lightweight, rule-based reasoning layer for Ask PanDA.
It sits between the user interface (text or audio input) and a set of concrete
clients (DocumentQuery, QueueQuery, TaskQuery, LogAnalysis, PilotMonitor,
MetadataAnalysis, Selection, PandaMCP).

The engine performs three main steps:

1. Perception:
   - Normalize the raw user input (text).
   - Classify the user's intent (documentation, queue status, log analysis, etc.).
   - Extract simple entities (job IDs, task IDs, queue names).

2. Reasoning:
   - Select which client should handle the request.
   - Infer a high-level goal for that client.
   - Estimate a lightweight confidence score.
   - Build a symbolic, high-level execution plan for the selected client
     (e.g. ["identify_failing_jobs", "fetch_logs_for_jobs", ...]).

3. Action:
   - Call the selected client with the original prompt, entities and goal.
   - Collect and wrap the result into an InteractionResult that can be logged
     or rendered by the front-end.

The clients themselves contain the actual domain-specific logic (MCP tools,
PanDA API calls, log analysis, etc.). This engine is deliberately simple and
transparent, so its decisions can be inspected, tested and evolved over time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AudioInput = Union[str, bytes]
"""Audio input representation for STT callables.

In practice this is usually a filesystem path to a WAV file, but the type
is kept generic to allow alternative implementations.
"""

STTCallable = Callable[[AudioInput], str]
"""Callable type for speech-to-text (STT) functions."""


# ---------------------------------------------------------------------------
# Client / handler protocols
# ---------------------------------------------------------------------------


class Handler(Protocol):
    """Protocol for all concrete Ask PanDA handler clients."""

    name: str

    def handle_request(
        self,
        *,
        prompt: str,
        entities: Dict[str, Any],
        goal: str,
        confidence: float,
    ) -> str:
        """Handle a user request routed by the reasoning engine.

        Args:
            prompt: The original user prompt (plain text).
            entities: Entities extracted during perception (job IDs, queues, etc.).
            goal: High-level goal for this handler (e.g. "diagnose job failure").
            confidence: Heuristic confidence in the routing decision.

        Returns:
            A formatted string answer ready to render to the user.
        """
        ...


class Selection(Protocol):
    """Protocol for a selection client that may override handler choice."""

    def select_handler(
        self,
        *,
        prompt: str,
        heuristic_candidate: str,
        entities: Dict[str, Any],
    ) -> Dict[str, str]:
        """Optionally override the heuristic handler choice.

        Args:
            prompt: Original user prompt.
            heuristic_candidate: Name of the handler chosen by built-in rules.
            entities: Extracted entities.

        Returns:
            A mapping with at least the key "handler_name", potentially
            containing additional information in the future.
        """
        ...


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Perception:
    """Result of the perception step.

    Attributes:
        raw_text: Original user query as text.
        intent: Classified intent label.
        entities: Extracted entities (job/task IDs, queue names, etc.).
        metadata: Miscellaneous metadata (token count, flags, etc.).
    """

    raw_text: str
    intent: str
    entities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Reasoning:
    """Result of the reasoning step.

    Attributes:
        goal: High-level goal for the selected handler.
        handler_name: Name of the handler that should process the request.
        confidence: Heuristic confidence score in [0.0, 1.0].
        plan: Symbolic ordered steps the handler is expected to perform.
    """

    goal: str
    handler_name: str
    confidence: float
    plan: List[str] = field(default_factory=list)


@dataclass
class InteractionResult:
    """Full outcome of handling a user interaction.

    Attributes:
        perception: Perception result (intent + entities).
        reasoning: Reasoning result (goal + handler + plan).
        handler_output: Raw output string from the handler client.
        formatted_answer: Final formatted answer for rendering/logging.
    """

    perception: Perception
    reasoning: Reasoning
    handler_output: str
    formatted_answer: str


# ---------------------------------------------------------------------------
# PanDA Reasoning Engine
# ---------------------------------------------------------------------------


class PanDAReasoningEngine:
    """Rule-based reasoning engine for Ask PanDA.

    The engine wires together several domain-specific clients and provides
    a uniform interface for text and (optional) audio-based queries.
    """

    def __init__(
        self,
        *,
        document_query: Handler,
        queue_query: Handler,
        task_query: Handler,
        log_analysis: Handler,
        pilot_monitor: Handler,
        metadata_analysis: Handler,
        panda_mcp: Handler,
        selection: Selection,
        stt_callable: Optional[STTCallable] = None,
    ) -> None:
        """Initialize the reasoning engine with concrete handler clients.

        Args:
            document_query: Client used for documentation and how-to questions.
            queue_query: Client used for queue/site status and properties.
            task_query: Client used for task-level queries.
            log_analysis: Client used for job failure and log analysis.
            pilot_monitor: Client used for pilot status and anomalies.
            metadata_analysis: Client used for job/task metadata analysis.
            selection: Client responsible for optional handler overrides.
            stt_callable: Optional speech-to-text function for audio input.
        """
        self._document_query = document_query
        self._queue_query = queue_query
        self._task_query = task_query
        self._log_analysis = log_analysis
        self._pilot_monitor = pilot_monitor
        self._metadata_analysis = metadata_analysis
        self._selection = selection
        self._stt_callable = stt_callable

        self._handlers: Dict[str, Handler] = {
            "DocumentQuery": document_query,
            "QueueQuery": queue_query,
            "TaskQuery": task_query,
            "LogAnalysis": log_analysis,
            "PilotMonitor": pilot_monitor,
            "MetadataAnalysis": metadata_analysis,
            "PandaMCP": panda_mcp,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_text(self, prompt: str) -> InteractionResult:
        """Handle a text query.

        Args:
            prompt: User query as plain text.

        Returns:
            InteractionResult containing perception, reasoning and handler output.
        """
        perception = self._perceive(prompt)
        reasoning = self._reason(perception)
        handler_output = self._execute_handler(
            reasoning.handler_name, prompt, perception, reasoning
        )
        formatted = self._format_answer(perception, reasoning, handler_output)
        return InteractionResult(
            perception=perception,
            reasoning=reasoning,
            handler_output=handler_output,
            formatted_answer=formatted,
        )

    def handle_audio(self, audio: AudioInput) -> InteractionResult:
        """Handle an audio query via the configured STT callable.

        Args:
            audio: Audio input (typically a path to a WAV file).

        Returns:
            InteractionResult as for :meth:`handle_text`.

        Raises:
            RuntimeError: If no STT callable is configured.
        """
        if self._stt_callable is None:
            raise RuntimeError("No STT callable configured for audio handling.")

        text = self._stt_callable(audio)
        return self.handle_text(text)

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def _perceive(self, prompt: str) -> Perception:
        """Perform simple perception: intent classification and entity extraction.

        Args:
            prompt: User query as plain text.

        Returns:
            Perception object with intent, entities and metadata.
        """
        text = prompt.strip()
        lowered = text.lower()

        entities = self._extract_entities(lowered)
        intent = self._classify_intent(lowered, entities)

        metadata: Dict[str, Any] = {
            "length": len(text.split()),
            "has_task_id": bool(entities.get("task_ids")),
            "has_queue": bool(entities.get("queues")),
        }

        return Perception(
            raw_text=text,
            intent=intent,
            entities=entities,
            metadata=metadata,
        )

    def _extract_entities(self, lowered: str) -> Dict[str, Any]:
        """Extract simple entities (job/task IDs, queue names) via regex.

        Args:
            lowered: Lowercase user query.

        Returns:
            Dictionary with extracted entities.
        """
        entities: Dict[str, Any] = {}

        # Job / task IDs (simple numeric heuristic)
        id_matches = re.findall(r"\b\d{5,}\b", lowered)
        if id_matches:
            entities["task_ids"] = id_matches

        # Queue names (very simplified; adapt to your real queue naming rules)
        queue_matches = re.findall(r"\b[a-z0-9_]{4,}\b", lowered)
        # Filter out obvious non-queue tokens if needed; here we keep everything.
        entities["queues"] = queue_matches

        return entities

    def _classify_intent(
        self,
        lowered: str,
        entities: Dict[str, Any],
    ) -> str:
        """Classify the user's intent based on simple keyword rules.

        This is intentionally simple and deterministic so that it can be
        unit-tested and iterated without model calls.

        Args:
            lowered: Lowercase user query.
            entities: Extracted entities.

        Returns:
            Intent label.
        """
        has_task_id = bool(entities.get("task_ids"))

        # PanDA server / MCP health checks
        if any(kw in lowered for kw in ["panda", "pandaserver", "panda server"]) and any(
                kw in lowered for kw in ["alive", "running", "up"]
        ):
            return "panda_mcp"

        # Log / failure questions about *jobs*
        if has_task_id and "job" in lowered and any(
            kw in lowered
            for kw in [
                "why did",
                "why has",
                "why is",
                "went wrong",
                "what happened",
                "has happened",
                "fail",
                "failed",
                "failing",
                "error",
            ]
        ):
            return "log_analysis"


        # Simple task summary requests (route to TaskQuery)
        if has_task_id and "task" in lowered and any(
            kw in lowered
            for kw in [
                "summarize",
                "summary",
                "overview",
            ]
        ):
            return "task_query"

        # Task-level explanation / status (metadata analysis)
        if has_task_id and "task" in lowered and any(
            kw in lowered
            for kw in [
                "explain",
                "what is happening",
                "what's happening",
                "status",
            ]
        ):
            return "metadata_analysis"

        # Queue / site / pilot status
        if any(kw in lowered for kw in ["queue", "site", "pilot"]):
            if "pilot" in lowered:
                return "pilot_status"
            # Avoid hijacking conceptual "how does PanDA ..." questions
            if not ("how does" in lowered and "panda" in lowered):
                return "queue_status"


        # Documentation / how-to (no specific task/job ID)
        if not has_task_id and any(
            kw in lowered
            for kw in [
                "how do i",
                "how can i",
                "how to",
                "how does panda",
                "how does pand",
                "explain",
                "what is panda",
                "what is pand",
                "prun",
                "pathena",
                "submit a job",
            ]
        ):
            return "documentation"


        # Task/job status questions that are not clearly failure-related
        if has_task_id and any(
            kw in lowered
            for kw in [
                "what is happening",
                "what's happening",
                "status",
                "explain job",
                "explain task",
                "analyze job",
                "analyze task",
            ]
        ):
            return "metadata_analysis"


        # Fallback generic
        return "generic_question"

    # ------------------------------------------------------------------
    # Reasoning and planning
    # ------------------------------------------------------------------

    def _reason(self, perception: Perception) -> Reasoning:
        """Select a handler, infer a goal, estimate confidence, and build a plan.

        Args:
            perception: Perception result from :meth:`_perceive`.

        Returns:
            Reasoning object.
        """
        heuristic_handler = self._heuristic_handler_choice(perception)
        overridden = self._selection.select_handler(
            prompt=perception.raw_text,
            heuristic_candidate=heuristic_handler,
            entities=perception.entities,
        )

        handler_name = overridden.get("handler_name", heuristic_handler)
        goal = self._infer_goal(handler_name, perception.intent)
        confidence = self._estimate_confidence(handler_name, perception)
        plan = self._build_plan(handler_name, perception)

        return Reasoning(
            goal=goal,
            handler_name=handler_name,
            confidence=confidence,
            plan=plan,
        )

    def _heuristic_handler_choice(self, perception: Perception) -> str:
        """Choose a handler based on intent and entities using simple rules.

        Args:
            perception: Perception result.

        Returns:
            Name of the handler that should handle the request.
        """
        intent = perception.intent
        has_task_id = bool(perception.entities.get("task_ids"))

        if intent == "documentation":
            return "DocumentQuery"

        if intent == "queue_status":
            return "QueueQuery"

        if intent == "task_query":
            return "TaskQuery"

        if intent == "pilot_status":
            return "PilotMonitor"

        if intent == "log_analysis":
            return "LogAnalysis"

        if intent == "metadata_analysis":
            return "MetadataAnalysis"

        if intent == "panda_mcp":
            return "PandaMCP"

        # Generic questions with job/task IDs → treat as metadata analysis
        if intent == "generic_question" and has_task_id:
            # Distinguish job vs task wording
            lowered = perception.raw_text.lower()
            if "task" in lowered:
                return "MetadataAnalysis"
            if "job" in lowered:
                return "LogAnalysis"

        # Default fallback
        return "DocumentQuery"

    def _infer_goal(self, handler_name: str, intent: str) -> str:
        """Infer a high-level goal string for the selected handler.

        Args:
            handler_name: Name of the selected handler.
            intent: Classified intent label.

        Returns:
            Human-readable goal description.
        """
        if handler_name == "DocumentQuery":
            return "Answer a documentation or how-to question about PanDA."

        if handler_name == "QueueQuery":
            return "Summarize the status and properties of the requested queues or sites."

        if handler_name == "TaskQuery":
            return "Summarize the current state and health of the requested task."

        if handler_name == "LogAnalysis":
            return "Diagnose failing jobs and identify likely error patterns."

        if handler_name == "PilotMonitor":
            return "Describe recent pilot activity and detect anomalies at the requested sites."

        if handler_name == "MetadataAnalysis":
            return "Summarize current metadata and activity for the specified task or job."

        if handler_name == "PandaMCP":
            return "Execute a tool or function on the PanDA server via PanDA MCP."

        return "Provide a helpful answer to a general PanDA-related question."

    def _estimate_confidence(self, handler_name: str, perception: Perception) -> float:
        """Estimate a heuristic confidence score in [0.0, 1.0].

        This score is intentionally simple; it is mainly useful for:
        - logging and monitoring routing quality over time;
        - deciding when to fall back to an LLM-based router if desired.

        Args:
            handler_name: Name of the selected handler.
            perception: Perception result.

        Returns:
            Confidence score.
        """
        base = 0.6

        # Small bonuses for clear signals
        if perception.intent in ("log_analysis", "metadata_analysis", "queue_status"):
            base += 0.1

        # If we have a task/job ID for failure or metadata questions, increase confidence
        if perception.entities.get("task_ids") and perception.intent in (
            "log_analysis",
            "metadata_analysis",
        ):
            base += 0.1

        # Clip to [0.0, 1.0]
        return max(0.0, min(1.0, base))

    def _build_plan(self, handler_name: str, perception: Perception) -> List[str]:
        """Construct a symbolic high-level plan for the selected handler.

        The plan is a list of conceptual step names the handler is expected
        to perform. The handler implementation is responsible for mapping
        these steps to actual tool calls and logic.

        Args:
            handler_name: Name of the selected handler (e.g. "LogAnalysis").
            perception: Perception describing the request.

        Returns:
            List of step names in execution order.
        """
        # Documentation / how-to
        if handler_name == "DocumentQuery":
            return [
                "interpret_question",
                "retrieve_relevant_docs",
                "synthesize_answer",
            ]

        # Queue / site status
        if handler_name == "QueueQuery":
            return [
                "identify_queues",
                "fetch_queue_status",
                "summarize_status",
            ]

        # Task-level queries (if you later introduce a dedicated TaskQuery handler)
        if handler_name == "TaskQuery":
            return [
                "identify_task",
                "fetch_task_metadata",
                "fetch_task_job_summary",
                "summarize_task_health",
            ]

        # Job log analysis / failure diagnosis
        if handler_name == "LogAnalysis":
            return [
                "identify_failing_jobs",
                "fetch_logs_for_jobs",
                "analyze_failure_patterns",
                "produce_diagnostic_summary",
            ]

        # Pilot state / pilot anomalies
        if handler_name == "PilotMonitor":
            return [
                "identify_relevant_sites",
                "fetch_pilot_metrics",
                "detect_anomalies",
                "summarize_pilot_activity",
            ]

        # Task/job metadata analysis (non-log)
        if handler_name == "MetadataAnalysis":
            return [
                "identify_task_or_job",
                "fetch_metadata",
                "analyze_state_transitions",
                "summarize_metadata_status",
            ]

        #
        if handler_name == "PandaMCP":
            return [
                "is_alive",
            ]

        # Generic fallback
        return ["interpret_question", "route_to_best_handler", "synthesize_answer"]

    # ------------------------------------------------------------------
    # Execution and formatting
    # ------------------------------------------------------------------

    def _execute_handler(
        self,
        handler_name: str,
        prompt: str,
        perception: Perception,
        reasoning: Reasoning,
    ) -> str:
        """Execute the selected handler client.

        Args:
            handler_name: Name of the handler to invoke.
            prompt: Original user prompt.
            perception: Perception result.
            reasoning: Reasoning result.

        Returns:
            Raw output from the handler's :meth:`handle_request`.
        """
        handler = self._handlers.get(handler_name)
        if handler is None:
            # This should not happen in normal operation; treat as a fallback.
            return (
                f"[Error] No handler registered under name '{handler_name}'. "
                f"Cannot process request."
            )

        return handler.handle_request(
            prompt=prompt,
            entities=perception.entities,
            goal=reasoning.goal,
            confidence=reasoning.confidence,
        )

    def _format_answer(
        self,
        perception: Perception,
        reasoning: Reasoning,
        handler_output: str,
    ) -> str:
        """Format a final answer string for logging or display.

        Args:
            perception: Perception result.
            reasoning: Reasoning result.
            handler_output: Raw output from the handler.

        Returns:
            Formatted multi-line string.
        """
        lines: List[str] = []

        lines.append(
            f"[Routing] intent={perception.intent}, "
            f"handler={reasoning.handler_name}, "
            f"confidence={reasoning.confidence:.2f}"
        )

        if reasoning.plan:
            lines.append("")
            lines.append("[Plan]")
            for idx, step in enumerate(reasoning.plan, start=1):
                lines.append(f"  {idx}. {step}")

        lines.append("")
        lines.append(handler_output)

        return "\n".join(lines)
