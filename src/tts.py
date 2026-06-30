"""Edge TTS Backend: cloud-based Microsoft Edge text-to-speech."""

import asyncio
import io
import os
import numpy as np
import soundfile as sf
import edge_tts


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str = "zh-CN-XiaoxiaoNeural", speed: float = 1.1) -> np.ndarray:
        raise NotImplementedError


class EdgeTTSBackend(TTSBackend):
    """edge-tts backend (Microsoft Edge cloud TTS)."""

    def __init__(self):
        self.sample_rate = 24000
        # Check default voice from environment or default to zh-CN-XiaoxiaoNeural
        self.default_voice = os.environ.get("TTS_VOICE", "zh-CN-XiaoxiaoNeural")

    async def _generate_async(self, text: str, voice: str, speed: float) -> bytes:
        # Convert speed to rate string (e.g. 1.1 -> "+10%", 0.9 -> "-10%")
        percent = int((speed - 1.0) * 100)
        rate = f"+{percent}%" if percent >= 0 else f"{percent}%"
        
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data

    def generate(self, text: str, voice: str = "zh-CN-XiaoxiaoNeural", speed: float = 1.1) -> np.ndarray:
        # If the voice is a Kokoro style voice (or default "af_heart"), map it to the default voice
        if not voice or voice.startswith("af_") or voice.startswith("am_"):
            voice = self.default_voice
        
        # Run async generation in a sync context safely
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                audio_bytes = executor.submit(lambda: asyncio.run(self._generate_async(text, voice, speed))).result()
        else:
            audio_bytes = asyncio.run(self._generate_async(text, voice, speed))
        
        if not audio_bytes:
            return np.zeros(0, dtype=np.float32)
            
        # Decode MP3 to PCM float32 array
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        return data


def load() -> TTSBackend:
    """Load the Edge TTS backend."""
    backend = EdgeTTSBackend()
    print(f"TTS: edge-tts (Cloud, default_voice={backend.default_voice}, sample_rate={backend.sample_rate})")
    return backend
