#!/usr/bin/env python3
"""
Streamlit UI for Ask PanDA Reasoning Engine with audio input.

Supports:
  * 🎙️ Microphone recording via streamlit-webrtc (browser mic).
  * 📁 Audio file upload (WAV/MP3/OGG/M4A).

Workflow:
  1. Capture audio (mic or upload).
  2. Whisper transcribes the audio.
  3. PanDAReasoningEngine classifies and routes the query.
  4. UI shows transcript, perception, reasoning, and formatted answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import time 

import av
import numpy as np
import soundfile as sf
import streamlit as st
import whisper
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration
from queue import Empty

# --- Placeholder Classes ---
class InteractionResult:
    def __init__(self):
        class Perception:
            raw_text: str = "Dummy transcription result: What is the status of task X?"
            intent: str = "TASK_STATUS_QUERY"
            entities: Dict[str, Any] = {"task_id": "X1234"}
            metadata: Dict[str, Any] = {}

        class Reasoning:
            goal: str = "Monitor task status."
            handler_name: str = "TaskQuery"
            confidence: float = 0.95

        self.perception = Perception()
        self.reasoning = Reasoning()
        self.formatted_answer: str = "[TaskQuery] Task X1234 is currently running in state 'active'."

class PanDAReasoningEngine:
    def __init__(self, **kwargs):
        self.stt_callable = kwargs.get('stt_callable')

    def handle_audio(self, audio_path: str) -> InteractionResult:
        try:
            transcript = self.stt_callable(audio_path)
            result = InteractionResult()
            result.perception.raw_text = transcript
            return result
        except Exception:
            return InteractionResult()
# --- End Placeholder Classes ---


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
# Utilities for audio saving
# ---------------------------------------------------------------------------

def save_uploaded_audio(uploaded_file) -> Path:
    """Save an uploaded audio file to a temporary WAV file if needed."""
    import tempfile

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        raw_path = Path(tmp.name)
        tmp.write(uploaded_file.getbuffer())

    if suffix in [".wav", ".wave"]:
        return raw_path

    data, samplerate = sf.read(str(raw_path))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = Path(tmp_wav.name)
    sf.write(wav_path, data, samplerate)
    return wav_path


def save_mic_frames_to_wav(frames: List[av.AudioFrame]) -> Path:
    """Convert a list of av.AudioFrame objects into a mono WAV file."""
    import tempfile

    if not frames:
        raise RuntimeError("No audio frames captured from microphone.")

    # Convert each frame to numpy, then concatenate
    arrays = []
    for frame in frames:
        # shape: (channels, samples)
        a = frame.to_ndarray()
        # average channels to mono if needed
        if a.ndim == 2:
            a = a.mean(axis=0)
        arrays.append(a)

    audio_data = np.concatenate(arrays, axis=0).astype(np.float32)

    samplerate = int(frames[0].sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = Path(tmp_wav.name)

    # Write mono WAV
    sf.write(wav_path, audio_data, samplerate)
    return wav_path


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

# Use a standard STUN server
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def main() -> None:
    st.set_page_config(
        page_title="Ask PanDA – Audio UI",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("🎙️ Ask PanDA – Audio Reasoning Demo")

    mode = st.radio(
        "Choose input method:",
        ["🎙️ Microphone (browser)", "📁 Upload audio file"],
        index=0,
    )

    engine = build_engine()

    if mode == "🎙️ Microphone (browser)":
        st.subheader("1. Record with your microphone")

        st.write(
            "Click **Start** below, grant microphone access in your browser, "
            "speak your question, then click **Transcribe** while the stream is active."
        )

        webrtc_ctx = webrtc_streamer(
            key="panda-audio",
            mode=WebRtcMode.RECVONLY,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"audio": True, "video": False},
        )

        if webrtc_ctx.state.playing:
            st.info("Streaming audio… speak now. You can click Transcribe at any time.")
        else:
            st.info("Click Start above to begin streaming from your microphone.")

        if st.button("Transcribe & Ask PanDA (from mic)"):
            
            # 1. Check if the stream is playing (initial state)
            if not webrtc_ctx.state.playing:
                st.error(
                    "The microphone stream is **not yet active**. "
                    "Please click **Start**, wait for the stream status to show 'Streaming audio...', "
                    "and ensure you have allowed microphone access in your browser."
                )
                return

            # 2. FIX: Poll for a short time to wait for the audio_receiver object to be initialized
            receiver_available = False
            start_time = time.time()
            # Increased timeout for extreme stability in flaky network environments
            TIMEOUT = 10.0 

            with st.spinner(f"Waiting up to {TIMEOUT} seconds for audio stream to establish..."):
                while time.time() - start_time < TIMEOUT:
                    if webrtc_ctx.audio_receiver is not None:
                        receiver_available = True
                        break
                    time.sleep(0.1) 

            if not receiver_available:
                st.error(
                    f"**Audio receiver object is still unavailable after {TIMEOUT} seconds.** "
                    "This usually means the WebRTC connection failed. Please try these steps:\n"
                    "1. Ensure Streamlit is served over **HTTPS** (required by most browsers for microphone access).\n"
                    "2. Check if a **VPN or firewall** is blocking UDP traffic (used by WebRTC).\n"
                    "3. Try a different browser (Chrome or Firefox)."
                )
                return

            # 3. If we reach here, the receiver is available and we can proceed.
            with st.spinner("Processing microphone audio…"):
                try:
                    # Removed timeout to reliably get all buffered frames.
                    frames = webrtc_ctx.audio_receiver.get_frames()
                except Empty:
                    frames = []
                    st.warning("Audio queue was empty. Ensure you spoke for a moment before clicking the button.")
                except Exception as e:
                    st.error(f"An error occurred while getting frames: {e}")
                    frames = []

                if not frames:
                    st.error(
                        "No audio frames captured. Make sure you click **Start**, "
                        "speak for a few seconds, and then click **Transcribe & Ask PanDA** "
                        "**while the stream is still active**."
                    )
                else:
                    wav_path = save_mic_frames_to_wav(frames)
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
        st.subheader("1. Upload your audio file")

        uploaded_file = st.file_uploader(
            "Record your question using any tool (browser / OS / phone), then upload it here.",
            type=["wav", "wave", "mp3", "ogg", "m4a"],
        )

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

            if st.button("Transcribe & Ask PanDA (from file)"):
                with st.spinner("Processing audio file..."):
                    wav_path = save_uploaded_audio(uploaded_file)
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
