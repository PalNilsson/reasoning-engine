from panda_reasoning_engine import PanDAReasoningEngine
from test_panda_reasoning_engine import (
    DocumentQuery, QueueQuery, TaskQuery, LogAnalysis,
    PilotMonitor, MetadataAnalysis, Selection
)

import whisper
from pathlib import Path

# Load Whisper
model = whisper.load_model("small")

def whisper_stt(audio):
    audio_path = Path(audio)
    txt = model.transcribe(str(audio_path))["text"]
    return txt

def build_engine():
    return PanDAReasoningEngine(
        document_query=DocumentQuery(),
        queue_query=QueueQuery(),
        task_query=TaskQuery(),
        log_analysis=LogAnalysis(),
        pilot_monitor=PilotMonitor(),
        metadata_analysis=MetadataAnalysis(),
        selection=Selection(),
        stt_callable=whisper_stt,  # <<< enable audio
    )

if __name__ == "__main__":
    engine = build_engine()

    # Replace with your microphone recording function or a waveform file
    audio_file = "recorded.wav"

    result = engine.handle_audio(audio_file)

    print("=== AUDIO QUERY RESULT ===")
    print("Transcript:", result.perception.raw_text)
    print("Intent:", result.reasoning.intent)
    print("Handler:", result.reasoning.handler_name)
    print("Answer:\n", result.formatted_answer)
