from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

TextInput = str
AudioInput = Union[str, Path, bytes]

Intent = Literal[
    "documentation",
    "queue_status",
    "task_status",
    "log_analysis",
    "pilot_monitoring",
    "metadata_analysis",
    "generic_question",
]

STTCallable = Callable[[AudioInput], str]


@dataclass
class Perception:
    """Structured view of the user request after the perception step.

    Attributes:
        raw_text: Canonical textual representation of the user request.
        intent: High-level intent label inferred by the perception step.
        entities: Extracted PanDA-related entities (task IDs, queues, sites, etc.).
        metadata: Additional perception metadata (e.g., length, flags).
    """

    raw_text: str
    intent: Intent
    entities: Dict[str, List[str]]
    metadata: Dict[str, Any]


@dataclass
class ReasoningState:
    """Reasoning state for a single Ask PanDA interaction.

    Attributes:
        goal: Natural-language statement of what the assistant is trying to achieve.
        handler_name: Name of the selected PanDA handler (e.g., "TaskQuery").
        confidence: Confidence score in the selected intent/handler.
        perception: Perception instance that this reasoning state is based on.
    """

    goal: str
    handler_name: str
    confidence: float
    perception: Perception


@dataclass
class InteractionResult:
    """Container for the full result of one Ask PanDA reasoning pass.

    Attributes:
        perception: Perception object describing how the request was understood.
        reasoning: ReasoningState describing the selected handler and goal.
        handler_response: Raw response object returned by the selected handler.
        formatted_answer: Optional formatted answer ready for display in the UI.
    """

    perception: Perception
    reasoning: ReasoningState
    handler_response: Any
    formatted_answer: Optional[str] = None


class PanDAReasoningEngine:
    """Reasoning engine for Ask PanDA with optional audio input.

    This engine sits in front of the various PanDA handlers and implements a
    simple perceive → reason → act loop:

      * Perceive: infer intent and extract basic entities from the request text.
      * Reason: map intent + entities to a goal and select the appropriate handler.
      * Act: delegate the actual work to the selected handler.

    The engine supports both text and audio input. Audio input is converted
    to text via an injected speech-to-text callable, after which all processing
    is text-based. Handlers encapsulate their own internal logic; the reasoning
    engine only decides which handler to use and exposes a transparent routing
    explanation that can be shown in the UI.
    """

    def __init__(
        self,
        *,
        document_query: Any,
        queue_query: Any,
        task_query: Any,
        log_analysis: Any,
        pilot_monitor: Any,
        metadata_analysis: Any,
        selection: Optional[Any] = None,
        stt_callable: Optional[STTCallable] = None,
    ) -> None:
        """Initialize the reasoning engine with PanDA handlers and STT.

        Args:
            document_query: Handler used for documentation / static knowledge
                queries (for example, ``DocumentQuery``).
            queue_query: Handler that answers queue/site information queries
                (for example, ``QueueQuery``).
            task_query: Handler that answers task and job status queries
                (for example, ``TaskQuery``).
            log_analysis: Handler that performs log and failure analysis
                (for example, ``LogAnalysis``).
            pilot_monitor: Handler that answers pilot-related questions
                (for example, ``PilotMonitor``).
            metadata_analysis: Handler that performs metadata-level analysis for
                tasks or jobs (for example, ``MetadataAnalysis``).
            selection: Optional Selection component used as a fallback to refine
                heuristic routing decisions. It is expected to expose a
                ``select_handler`` method.
            stt_callable: Optional function that converts audio input to text.
                It should accept an ``AudioInput`` and return the transcribed
                text as a string.
        """
        self.document_query = document_query
        self.queue_query = queue_query
        self.task_query = task_query
        self.log_analysis = log_analysis
        self.pilot_monitor = pilot_monitor
        self.metadata_analysis = metadata_analysis
        self.selection = selection
        self.stt_callable = stt_callable

    # -------------------------------------------------------------------------
    # Public entry points
    # -------------------------------------------------------------------------

    def handle_text(self, prompt: TextInput) -> InteractionResult:
        """Handle a single Ask PanDA request from a text prompt.

        Args:
            prompt: User's textual query.

        Returns:
            InteractionResult: Full interaction result with perception, reasoning
            and handler response.
        """
        perception = self._perceive_text(prompt)
        reasoning = self._reason(perception)
        handler_response = self._act(reasoning)
        formatted_answer = self._format_answer(handler_response, reasoning)
        return InteractionResult(
            perception=perception,
            reasoning=reasoning,
            handler_response=handler_response,
            formatted_answer=formatted_answer,
        )

    def handle_audio(self, audio: AudioInput) -> InteractionResult:
        """Handle a single Ask PanDA request from audio input.

        Audio is converted to text via the configured STT callable before running
        the normal perceive → reason → act pipeline.

        Args:
            audio: Audio input (for example, a file path or raw bytes) to be
                transcribed.

        Returns:
            InteractionResult: Full interaction result with perception, reasoning
            and handler response.

        Raises:
            RuntimeError: If no STT callable has been configured.
        """
        if self.stt_callable is None:
            raise RuntimeError(
                "No STT callable configured on PanDAReasoningEngine; "
                "audio input handling is not available."
            )

        text = self.stt_callable(audio)
        return self.handle_text(text)

    # -------------------------------------------------------------------------
    # Perception
    # -------------------------------------------------------------------------

    def _perceive_text(self, prompt: str) -> Perception:
        """Derive intent and entities from the raw text prompt.

        This method uses lightweight, deterministic rules. It can be extended or
        combined with a Selection component if richer understanding is required
        for ambiguous queries.

        Args:
            prompt: User's textual query.

        Returns:
            Perception: Structured perception object.
        """
        normalized = prompt.strip()
        intent = self._infer_intent(normalized)
        entities = self._extract_entities(normalized)
        metadata: Dict[str, Any] = {
            "length": len(normalized.split()),
            "has_task_id": bool(entities.get("task_ids")),
            "has_queue": bool(entities.get("queues")),
        }
        return Perception(
            raw_text=normalized,
            intent=intent,
            entities=entities,
            metadata=metadata,
        )

    def _infer_intent(self, text: str) -> Intent:
        """Infer a high-level intent label from the given text.

        The rules in this method are intentionally explicit and biased towards
        clarity rather than cleverness. They distinguish carefully between:

        * Job vs task questions (jobs have logs; tasks do not).
        * Failure-oriented queries (routed to log analysis for jobs, but to
          metadata analysis for tasks).
        * Documentation-style questions about how PanDA works.
        * Queue/site status vs pilot monitoring vs metadata vs logs.

        Args:
            text: User's normalized request text.

        Returns:
            Intent: Coarse intent label used for routing to handlers.
        """
        lower = text.lower()

        # Basic helpers
        job_present = "job" in lower
        task_present = "task" in lower

        import re

        ids = re.findall(r"\b\d{4,}\b", lower)
        has_id = len(ids) > 0

        # ------------------------------------------------------------------
        # 1) "Why ..." logic
        # ------------------------------------------------------------------
        if "why" in lower:
            if job_present:
                return "log_analysis"
            if task_present:
                return "metadata_analysis"

        # ------------------------------------------------------------------
        # 2) "What happened / what went wrong"
        #     Jobs -> logs, tasks -> metadata
        # ------------------------------------------------------------------
        what_happened_phrases = [
            "what happened",
            "what has happened",
            "what exactly happened",
            "check what happened",
            "know what happened",
        ]
        went_wrong_phrases = [
            "went wrong",
            "what went wrong",
            "wrong with",
        ]

        if any(p in lower for p in what_happened_phrases) or any(
            p in lower for p in went_wrong_phrases
        ):
            if job_present:
                return "log_analysis"
            if task_present:
                return "metadata_analysis"

        # ------------------------------------------------------------------
        # 3) Direct job/task analysis: "explain job 1234", "analyze task 1234"
        #    Only when we have an ID and the verb directly modifies job/task.
        # ------------------------------------------------------------------
        direct_analysis_patterns = [
            "explain job",
            "explain task",
            "analyze job",
            "analyze task",
            "analysis of job",
            "analysis of task",
            "describe job",
            "describe task",
            "inspect job",
            "inspect task",
        ]
        if has_id and any(p in lower for p in direct_analysis_patterns):
            return "metadata_analysis"

        # ------------------------------------------------------------------
        # 4) "What is happening with job|task" → metadata_analysis
        # ------------------------------------------------------------------
        if "what is happening with" in lower or "what's happening with" in lower:
            if job_present or task_present:
                return "metadata_analysis"

        # ------------------------------------------------------------------
        # 5) Task/job status queries ("status", "summarize", "summary")
        # ------------------------------------------------------------------
        if (job_present or task_present) and any(
            kw in lower for kw in ["status", "summarize", "summary"]
        ):
            return "task_status"

        # ------------------------------------------------------------------
        # 6) Pilot queries
        # ------------------------------------------------------------------
        if "pilot" in lower:
            return "pilot_monitoring"

        # ------------------------------------------------------------------
        # 7) Documentation / conceptual "how does / how do i / how can i / what is"
        # ------------------------------------------------------------------
        if any(
            kw in lower
            for kw in [
                "how does",
                "how do i",
                "how can i",
                "what is",
                "documentation",
                "manual",
            ]
        ):
            return "documentation"

        # ------------------------------------------------------------------
        # 8) Queue / site status questions
        # ------------------------------------------------------------------
        queue_words = ["queue", "site"]
        status_words = ["status", "state", "health", "usage", "load", "backlog"]

        if any(q in lower for q in queue_words) and any(
            s in lower for s in status_words
        ):
            return "queue_status"

        # ------------------------------------------------------------------
        # 9) Generic logs/failures (no explicit job/task routing)
        # ------------------------------------------------------------------
        if any(
            kw in lower
            for kw in ["log", "error", "failure", "crash", "aborted", "killed"]
        ):
            return "log_analysis"

        return "generic_question"

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract PanDA-relevant entities from the text.

        This is a placeholder for progressively richer entity recognition
        (task IDs, job IDs, queues, sites, experiments, etc.). It can be
        extended to support more realistic patterns or replaced by a more
        sophisticated NER component later.

        Args:
            text: User's textual request.

        Returns:
            Dict[str, List[str]]: Mapping from entity type to a list of values.
        """
        import re

        entities: Dict[str, List[str]] = {}

        # Example: numeric IDs (task/job IDs)
        task_ids = re.findall(r"\b\d{4,}\b", text)
        if task_ids:
            entities["task_ids"] = task_ids

        # Example: PanDA queues (very rough; adapt to your actual naming)
        queues = re.findall(r"\b(A-T[A-Z0-9-]+|ANALY_[A-Z0-9_-]+)\b", text)
        if queues:
            entities["queues"] = queues

        # Example: sites (placeholder pattern)
        sites = re.findall(r"\b[A-Z]{2,}[_-]SITE[_-][A-Z0-9]+\b", text)
        if sites:
            entities["sites"] = sites

        experiments = []
        for exp_name in ["ATLAS", "ePIC", "EPIC", "RUBIN", "Vera Rubin"]:
            if exp_name.lower() in text.lower():
                experiments.append(exp_name)
        if experiments:
            entities["experiments"] = experiments

        return entities

    # -------------------------------------------------------------------------
    # Reasoning
    # -------------------------------------------------------------------------

    def _reason(self, perception: Perception) -> ReasoningState:
        """Construct a reasoning state from the perception.

        Args:
            perception: Perception object describing how the request was parsed.

        Returns:
            ReasoningState: Reasoning state with goal, selected handler and
            confidence score.
        """
        goal = self._identify_goal(perception)
        handler_name = self._select_handler_name(perception)
        confidence = self._estimate_confidence(perception, handler_name)
        return ReasoningState(
            goal=goal,
            handler_name=handler_name,
            confidence=confidence,
            perception=perception,
        )

    def _identify_goal(self, perception: Perception) -> str:
        """Map the perceived intent and entities to a goal description.

        Args:
            perception: Perception associated with the current request.

        Returns:
            str: Human-readable goal statement.
        """
        intent = perception.intent
        if intent == "documentation":
            return "Answer a documentation or how-to question about PanDA."
        if intent == "queue_status":
            return (
                "Summarize the status and properties of the requested queues or sites."
            )
        if intent == "task_status":
            return "Explain the status and health of a specific PanDA task and its jobs."
        if intent == "log_analysis":
            return "Diagnose failing jobs and identify likely error patterns."
        if intent == "pilot_monitoring":
            return (
                "Describe pilot activity and any observed issues for relevant sites."
            )
        if intent == "metadata_analysis":
            return (
                "Summarize current metadata and activity for the specified task or job."
            )
        return "Provide a helpful answer to a general PanDA-related question."

    def _select_handler_name(self, perception: Perception) -> str:
        """Select the PanDA handler name to handle the request.

        This method performs heuristic routing based on the intent and entities.
        If a Selection component is configured, it may refine or override the
        heuristic choice by calling its ``select_handler`` method.

        Args:
            perception: Perception describing the request.

        Returns:
            str: Name of the selected handler (for example, ``TaskQuery``).
        """
        if perception.intent == "documentation":
            candidate = "DocumentQuery"
        elif perception.intent == "queue_status":
            candidate = "QueueQuery"
        elif perception.intent == "task_status":
            candidate = "TaskQuery"
        elif perception.intent == "log_analysis":
            candidate = "LogAnalysis"
        elif perception.intent == "pilot_monitoring":
            candidate = "PilotMonitor"
        elif perception.intent == "metadata_analysis":
            candidate = "MetadataAnalysis"
        else:
            candidate = "DocumentQuery"

        if self.selection is not None:
            try:
                selection_result = self.selection.select_handler(
                    prompt=perception.raw_text,
                    heuristic_candidate=candidate,
                    entities=perception.entities,
                )
                return selection_result.get("handler_name", candidate)
            except Exception:
                # In testing and production we prefer robustness: fall back to
                # the heuristic rather than failing the entire request.
                return candidate

        return candidate

    def _estimate_confidence(self, perception: Perception, handler_name: str) -> float:
        """Estimate the confidence in the routing decision.

        Args:
            perception: Perception object associated with the current request.
            handler_name: Name of the selected handler.

        Returns:
            float: Confidence score between 0.0 and 1.0.
        """
        score = 0.6

        if perception.entities.get("task_ids") and handler_name in (
            "TaskQuery",
            "MetadataAnalysis",
        ):
            score += 0.2

        if perception.entities.get("queues") and handler_name == "QueueQuery":
            score += 0.15

        if perception.intent == "log_analysis" and handler_name == "LogAnalysis":
            score += 0.15

        length = perception.metadata.get("length", 0)
        if length < 3 or length > 80:
            score -= 0.1

        return max(0.0, min(1.0, score))

    # -------------------------------------------------------------------------
    # Action
    # -------------------------------------------------------------------------

    def _act(self, reasoning: ReasoningState) -> Any:
        """Delegate to the selected PanDA handler.

        The selected handler is responsible for its internal steps. The engine
        passes contextual information (prompt, entities, goal, confidence) so
        that the handler can decide how to process the request.

        Args:
            reasoning: ReasoningState containing the goal, handler and confidence.

        Returns:
            Any: Raw response object returned by the selected handler.
        """
        handler = self._resolve_handler(reasoning.handler_name)
        prompt = reasoning.perception.raw_text
        entities = reasoning.perception.entities

        return handler.handle_request(
            prompt=prompt,
            entities=entities,
            goal=reasoning.goal,
            confidence=reasoning.confidence,
        )

    def _resolve_handler(self, handler_name: str) -> Any:
        """Return the concrete handler instance for a given handler name.

        Args:
            handler_name: Name of the handler (for example, ``TaskQuery``).

        Returns:
            Any: Handler instance.

        Raises:
            ValueError: If the handler name is unknown.
        """
        if handler_name == "DocumentQuery":
            return self.document_query
        if handler_name == "QueueQuery":
            return self.queue_query
        if handler_name == "TaskQuery":
            return self.task_query
        if handler_name == "LogAnalysis":
            return self.log_analysis
        if handler_name == "PilotMonitor":
            return self.pilot_monitor
        if handler_name == "MetadataAnalysis":
            return self.metadata_analysis
        raise ValueError(f"Unknown handler_name: {handler_name!r}")

    # -------------------------------------------------------------------------
    # Answer formatting
    # -------------------------------------------------------------------------

    def _format_answer(self, handler_response: Any, reasoning: ReasoningState) -> str:
        """Format a handler response into a UI-ready answer.

        In many cases, the underlying handler already returns a nicely formatted
        string. This method exists as a central place to add optional decorations
        such as routing summaries, disclaimers, or links.

        Args:
            handler_response: Raw response returned by the handler.
            reasoning: ReasoningState that led to this response.

        Returns:
            str: Final string to display in the Ask PanDA UI.
        """
        if isinstance(handler_response, str):
            core = handler_response
        else:
            core = str(handler_response)

        header = (
            f"[Routing] intent={reasoning.perception.intent}, "
            f"handler={reasoning.handler_name}, "
            f"confidence={reasoning.confidence:.2f}"
        )
        return f"{header}\n\n{core}"
