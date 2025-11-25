#!/usr/bin/env python3
"""
Streamlit UI for Ask PanDA Reasoning Engine with audio upload.

Workflow:
  1. User records audio using any tool (browser, OS, phone) and uploads file.
  2. Whisper transcribes the audio.
  3. PanDAReasoningEngine classifies and routes the query.
  4. UI shows transcript, perception, reasoning, and formatted answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import soundfile as sf
import streamlit as st
import whisper

from panda_reasoning_engine import PanDAReasoningEngine, InteractionResult


# ---------------------------------------------------------------------------
# Dummy handlers (replace with real Ask PanDA clients)
# ---------------------------------------------------------------------------

class BaseDummyHandler:
    name: str = "BaseDummyHandler"

    def handle_request(
        self,
        *,
        prompt: str,
        entities: Dict[str, Any],
        goal: str,
        confidence: float,
    ) -> str:
        return (
            f"[{self.name}] Handling request\n"
            f"  goal: {goal}\n"
            f"  confidence: {confidence:.2f}\n"
            f"  entities: {entities}\n"
            f"  prompt: {prompt!r}"
        )


class DocumentQuery(BaseDummyHandler):
    name = "DocumentQuery"


class QueueQuery(BaseDummyHandler):
    name = "QueueQuery"


class TaskQuery(BaseDummyHandler):
    name = "TaskQuery"


class LogAnalysis(BaseDummyHandler):
    name = "LogAnalysis"


class PilotMonitor(BaseDummyHandler):
    name = "PilotMonitor"


class MetadataAnalysis(BaseDummyHandler):
    name = "MetadataAnalysis"


class Selection:
    def select_handler(
        self,
        *,
        prompt: str,
        heuristic_candidate: str,
        entities: Dict[str, Any],
    ) -> Dict[str, str]:
        return {"handler_name": heuristic_candidate}


# ---------------------------------------------------------------------------
# Whisper + Reasoning Engine setup
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_whisper_model() -> whisper.Whisper:
    """Load and cache the Whisper 'small' model."""
    return whisper.load_model("small")


def whisper_stt(audio_path: Path) -> str:
    """Speech-to-text using the cached Whisper model."""
    model = load_whisper_model()
    result = model.transcribe(str(audio_path))
    return result["text"]


@st.cache_resource(show_spinner=True)
def build_engine() -> PanDAReasoningEngine:
    """Construct and cache a PanDAReasoningEngine wired to handlers and STT."""
    document_query = DocumentQuery()
    queue_query = QueueQuery()
    task_query = TaskQuery()
    log_analysis = LogAnalysis()
    pilot_monitor = PilotMonitor()
    metadata_analysis = MetadataAnalysis()
    selection = Selection()

    engine = PanDAReasoningEngine(
        document_query=document_query,
        queue_query=queue_query,
        task_query=task_query,
        log_analysis=log_analysis,
        pilot_monitor=pilot_monitor,
        metadata_analysis=metadata_analysis,
        selection=selection,
        stt_callable=lambda audio: whisper_stt(Path(audio)),
    )
    return engine


# ---------------------------------------------------------------------------
# Utility: save uploaded file to disk
# ---------------------------------------------------------------------------

def save_uploaded_audio(uploaded_file) -> Path:
    """Save an uploaded audio file to a temporary WAV file if needed.

    If the uploaded file is already WAV, we just write it out unchanged.
    Otherwise, we decode and re-encode as WAV using soundfile.
    """
    import tempfile

    suffix = Path(uploaded_file.name).suffix.lower()

    # Write bytes to a temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        raw_path = Path(tmp.name)
        tmp.write(uploaded_file.getbuffer())

    if suffix in [".wav", ".wave"]:
        # Whisper can handle WAV directly
        return raw_path

    # For other formats (e.g. .mp3, .ogg, .m4a), re-encode as WAV
    data, samplerate = sf.read(str(raw_path))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = Path(tmp_wav.name)
    sf.write(wav_path, data, samplerate)
    return wav_path


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Ask PanDA – Audio UI",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("🎙️ Ask PanDA – Audio Reasoning Demo")
    st.write(
        "Upload an audio recording of your question about PanDA. "
        "Whisper will transcribe it, and the reasoning engine will classify "
        "and route it to the appropriate client."
    )

    st.markdown("---")

    st.subheader("1. Upload your audio file")

    uploaded_file = st.file_uploader(
        "Record your question using any tool (browser / OS / phone), then upload it here.",
        type=["wav", "wave", "mp3", "ogg", "m4a"],
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Transcribe & Ask PanDA"):
            with st.spinner("Processing audio..."):
                wav_path = save_uploaded_audio(uploaded_file)
                engine = build_engine()
                result: InteractionResult = engine.handle_audio(str(wav_path))

            st.markdown("### 2. Transcription")
            st.code(result.perception.raw_text)

            st.markdown("### 3. Perception")
            st.json(
                {
                    "intent": result.perception.intent,
                    "entities": result.perception.entities,
                    "metadata": result.perception.metadata,
                }
            )

            st.markdown("### 4. Reasoning")
            st.json(
                {
                    "goal": result.reasoning.goal,
                    "handler": result.reasoning.handler_name,
                    "confidence": result.reasoning.confidence,
                }
            )

            st.markdown("### 5. Answer")
            st.code(result.formatted_answer)
    else:
        st.info("Upload a short audio question to get started.")


if __name__ == "__main__":
    main()
