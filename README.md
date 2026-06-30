# Parlor (Edge TTS Chinese Edition)

On-device real-time multimodal AI (voice + vision), updated with **Edge TTS** to support high-quality Chinese and English voice generation.

Parlor uses [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) for understanding speech and vision (running entirely locally on your Mac's GPU), and **Edge TTS** (cloud-based) for low-latency, natural speech synthesis.

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **Research preview.** This is an early experiment. Expect rough edges and bugs.

## How it works

```
Browser (mic + camera)
    │
    │  WebSocket (audio PCM + JPEG frames)
    ▼
    FastAPI server
    ├── Gemma 4 E2B via LiteRT-LM (GPU)  →  understands speech + vision (100% Local)
    └── Edge TTS (Microsoft Cloud API)   →  generates voice response (Online, multilingual)
    │
    │  WebSocket (streamed audio chunks)
    ▼
    Browser (playback + transcript)
```

- **Voice Activity Detection** in the browser ([Silero VAD](https://github.com/ricky0123/vad)). Hands-free, no push-to-talk.
- **Barge-in.** Interrupt the AI mid-sentence by speaking.
- **Sentence-level TTS streaming.** Audio starts playing before the full response is finished generating.
- **Multilingual Support.** System prompts are optimized for Chinese interaction and match the spoken language dynamically.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4 Series)
- Python 3.12 (standard for running current dependencies)
- Active internet connection (required for **Edge TTS** voice generation; LLM/vision inference runs completely offline)
- ~3 GB free RAM for the Gemma 4 E2B model

## Deployment & Quick Start (macOS)

Follow these steps to deploy and run Parlor on a Mac device:

### 1. Install uv (Python Package Manager)
We use `uv` for lightning-fast dependency resolution and virtual environment management.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Prepare the Codebase
```bash
git clone https://github.com/fikrikarim/parlor.git
cd parlor
```

### 3. Configure Environment Variables
Create your local `.env` file by copying the template:
```bash
cp .env.example .env
```
Open `.env` in your editor. You can configure:
- `TTS_VOICE`: The voice used by Edge TTS (defaults to `zh-CN-XiaoxiaoNeural` for natural Chinese female voice).
- `MODEL_PATH`: If you have already downloaded `gemma-4-E2B-it.litertlm` locally, uncomment and point it to the file path. Otherwise, leave it commented, and it will auto-download from HuggingFace on first run.

> [!TIP]
> If you prefer not to modify the `.env` file, you can directly set the environment variable in your terminal session before launching the server:
> ```bash
> 建议把模型放到当前项目的models文件夹下
> export MODEL_PATH="/Users/frank/Documents/parlor_zh/models/gemma-4-E2B-it.litertlm"
> ```

### 4. Install Dependencies & Start the Server
```bash
cd src
uv venv && source .venv/bin/activate

uv sync
# If you exported MODEL_PATH in the terminal, it will override the .env setting automatically

uv run server.py
```

### 5. Access the Web UI
Open [http://localhost:8000](http://localhost:8000) in Safari or Chrome, grant camera and microphone permissions, and start talking to the AI!

---

## Configuration Reference

You can configure the following environment variables inside the `.env` file in the root directory:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_PATH` | auto-download | Path to a local `gemma-4-E2B-it.litertlm` file (avoids downloading on launch) |
| `TTS_VOICE` | `zh-CN-XiaoxiaoNeural` | Microsoft Edge voice ID. E.g., `zh-CN-XiaoxiaoNeural` (Chinese Female), `zh-CN-YunxiNeural` (Chinese Male), `en-US-AriaNeural` (English) |
| `PORT` | `8000` | Port of the FastAPI server |

## Performance (Apple M3 Pro)

| Stage | Time |
| :--- | :--- |
| Speech + vision understanding | ~1.8 - 2.2s |
| Response generation (~25 tokens) | ~0.3s |
| Edge TTS voice synthesis | ~0.2 - 0.4s |
| **Total end-to-end** | **~2.3 - 2.8s** |

## Project Structure

```
src/
├── server.py              # FastAPI WebSocket server + Gemma 4 inference (system prompts optimized for Chinese)
├── tts.py                 # Edge TTS engine (decodes cloud MP3 streams to raw 24kHz float32 PCM on the fly)
├── index.html             # Frontend HTML (VAD, camera frame capture, audio player)
├── pyproject.toml         # Dependencies managed by uv (including edge-tts and soundfile)
└── benchmarks/
    ├── bench.py           # End-to-end WebSocket benchmark
    └── benchmark_tts.py   # TTS backend comparison
```

## License

[Apache 2.0](LICENSE)
