"""Who said what: serves index.html, and POST /diarize runs pyannote Community-1 on the uploaded audio.

Two ways to run it, from the repository root:
    uv run python app/server.py   # the diarization server, native and fast: page and /diarize on port 8001
    docker compose up --build     # the website on port 8000; its /diarize forwards to the one above (DIARIZE_URL)

The page reads results.jsonl and segments.csv in the browser. Only the full audio is sent, to POST /diarize,
which returns the turns as JSON. Nothing leaves this machine except the model download from Hugging Face.
"""
import http.client
import io
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")  # pyannote.audio's usage telemetry off


def load_dotenv(path):
    """KEY=value lines from the repository's .env (as docker compose reads it). Variables already set win."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

HOST = os.environ.get("HOST", "127.0.0.1")  # the Docker image sets 0.0.0.0; compose publishes it on 127.0.0.1 only
PORT = int(os.environ.get("PORT", "8001"))  # the Docker website sets 8000 and forwards /diarize here
DIARIZE_URL = os.environ.get("DIARIZE_URL")  # set: forward /diarize to this server instead of running pyannote here
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
MODEL = "pyannote/speaker-diarization-community-1"

_pipeline = None
_lock = threading.Lock()  # one diarization at a time: the model is not shared safely across threads


def load_pipeline():
    """Load Community-1 once, on the first request. Never print the pipeline: it can print the token."""
    global _pipeline
    if _pipeline is None:
        import torch
        torch.backends.nnpack.set_flags(False)  # NNPACK fails on emulated CPUs (Docker on Apple Silicon) and floods the log
        from pyannote.audio import Pipeline
        from pyannote.audio.core.task import Problem, Resolution, Specifications

        # torch >= 2.6 loads only the classes it is told are safe; these are pyannote's own, stored in its checkpoints
        torch.serialization.add_safe_globals([Specifications, Problem, Resolution, torch.torch_version.TorchVersion])
        # token None falls back to a cached `hf auth login`
        pipe = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
        if pipe is None:
            raise RuntimeError(f"could not load {MODEL}: accept its conditions on huggingface.co and set HF_TOKEN")
        pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        _pipeline = pipe
    return _pipeline


def read_wav(data):
    """16-bit PCM WAV bytes -> (channel, time) float tensor and sample rate. pyannote resamples to 16 kHz itself."""
    import numpy as np
    import torch

    try:
        with wave.open(io.BytesIO(data), "rb") as f:
            channels, width, sr = f.getnchannels(), f.getsampwidth(), f.getframerate()
            pcm = f.readframes(f.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError(f"not a PCM WAV file ({e}); convert it: ffmpeg -i in.mp3 -ac 1 -ar 16000 out.wav") from e
    if width != 2:
        raise ValueError(f"{8 * width}-bit WAV; only 16-bit is read: ffmpeg -i in.wav -c:a pcm_s16le out.wav")
    x = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels).T / 2**15
    return torch.from_numpy(x).float(), sr


def turns(annotation):
    return [{"start": round(s.start, 3), "end": round(s.end, 3), "speaker": label}
            for s, _, label in annotation.itertracks(yield_label=True)]


def diarize(data, count):
    waveform, sr = read_wav(data)
    with _lock:
        pipe = load_pipeline()
        t0 = time.time()
        out = pipe({"waveform": waveform, "sample_rate": sr}, **count)
        seconds = time.time() - t0
    return {
        "duration": round(waveform.shape[1] / sr, 3),
        "seconds": round(seconds, 1),
        "speakers": sorted(out.speaker_diarization.labels()),
        "turns": turns(out.speaker_diarization),                      # may overlap: two people at once
        "exclusive_turns": turns(out.exclusive_speaker_diarization),  # one speaker at a time
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path not in ("/", "/index.html"):
            return self.send_error(404)
        with open(PAGE, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/diarize":
            return self.send_error(404)
        query = parse_qs(url.query)
        count = {k: int(query[k][0]) for k in ("num_speakers", "min_speakers", "max_speakers")
                 if query.get(k, [""])[0].isdigit()}
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if DIARIZE_URL:
            return self.forward(data, url.query)
        try:
            self.send_json(200, diarize(data, count))
        except ValueError as e:
            self.send_json(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def forward(self, data, query):
        """The website's /diarize: pass the audio to the diarization server and its answer back, unchanged."""
        req = urllib.request.Request(f"{DIARIZE_URL}/diarize" + (f"?{query}" if query else ""), data=data,
                                     headers={"Content-Type": "application/octet-stream"}, method="POST")
        try:
            with urllib.request.urlopen(req) as r:  # no timeout: a long recording takes minutes
                status, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, body = e.code, e.read()
        except urllib.error.URLError as e:
            return self.send_json(503, {"error": f"the diarization server is not running ({e.reason}):"
                                                 " start it with `uv run python app/server.py`"})
        except (ConnectionError, http.client.HTTPException) as e:  # it stopped or crashed in the middle of the run
            return self.send_json(502, {"error": f"the diarization server stopped during the run ({type(e).__name__}):"
                                                 " check its terminal, restart it and press Diarize again"})
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


if __name__ == "__main__":
    if DIARIZE_URL:
        print(f"website only: /diarize goes to {DIARIZE_URL}")
    elif not os.environ.get("HF_TOKEN"):
        print("note: HF_TOKEN is not set in the environment or .env; falling back to a cached `hf auth login`, if any")
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        if e.errno not in (48, 98):  # address in use: macOS, Linux
            raise
        sys.exit(f"port {PORT} is taken: is app/server.py already running? Or set PORT=... to use another one")
    print(f"listening on {HOST}:{PORT}; open http://127.0.0.1:{PORT}")
    server.serve_forever()
