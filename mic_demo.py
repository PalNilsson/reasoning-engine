# mic_demo.py

from demo_audio import whisper_stt   # reuse STT!
from demo_audio import build_engine  # reuse handler wiring

import queue
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

def record_microphone(
    duration: float = 5.0,
    samplerate: int = 16_000,
    channels: int = 1,
    verbose: bool = True,
) -> Path:
    """Record audio from the default microphone into a temporary WAV file.

    This function opens the system default microphone, captures audio for the
    specified duration, writes it to a temporary .wav file using soundfile,
    and returns the path to that file.

    Args:
        duration (float): Number of seconds to record. Defaults to 5 seconds.
        samplerate (int): Sample rate in Hz. Whisper models expect 16 kHz.
        channels (int): Number of audio channels. For speech, use 1.
        verbose (bool): If True, print recording status messages.

    Returns:
        Path: Path to the temporary WAV file containing the recording.

    Raises:
        RuntimeError: If recording fails or yields no audio frames.
    """
    if verbose:
        print(f"Recording {duration:.1f} seconds of audio... (speak now)")

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time, status):
        """sounddevice callback function for asynchronous recording."""
        if status:
            print(f"[sounddevice warning] {status}")
        audio_queue.put(indata.copy())

    try:
        # Create a temporary file path for the WAV output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        frames = []

        # Open the microphone stream
        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            callback=callback,
        ):
            # Wait for (duration) seconds, collecting frames into queue
            sd.sleep(int(duration * 1000))

            # Drain the queue
            while not audio_queue.empty():
                frames.append(audio_queue.get())

        if not frames:
            raise RuntimeError("No audio captured from microphone.")

        # Concatenate all frame batches into a single numpy array
        audio_data = np.concatenate(frames, axis=0)

        # Write to the WAV file
        sf.write(wav_path, audio_data, samplerate)

        if verbose:
            print(f"Saved recording to: {wav_path}")

        return wav_path

    except Exception as exc:
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc

if __name__ == "__main__":
    engine = build_engine()
    audio_path = record_microphone()  # your microphone recorder
    result = engine.handle_audio(audio_path)
    print(result.formatted_answer)
