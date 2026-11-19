# Thai speech notebooks

Hands-on companions to the team's speech manual: one notebook per section, each runnable in Colab or locally.

| Notebook | Section | Status |
|---|---|---|
| [vad.ipynb](vad.ipynb) | Voice activity detection | draft |
| [ASR_transcription.ipynb](ASR_transcription.ipynb) | Speech-to-text | draft |
| [diarization.ipynb](diarization.ipynb) | Speaker diarization | draft |
| [pipeline.ipynb](pipeline.ipynb) | All three on one recording: Part 1 VAD, Part 2 ASR, Part 3 diarization, then who said what | draft |

## Rules

- **Licensed public audio only.** Every file in `data/` must carry an open license that allows redistribution, credited below. No company recordings, and nothing downloaded from YouTube or other sites without such a license.
- One notebook per section, kept running by that section's author.

## Run

**Locally** (needs [uv](https://docs.astral.sh/uv)):

```bash
git clone <this repository's URL>
cd thai-speech-notebooks
uv sync            # installs the pinned libraries from uv.lock into .venv
uv run jupyter lab
```

On Linux, TEN VAD also needs the system library `libc++1` (`sudo apt install libc++1`), as its package notes say.

**Colab:** open the notebook from GitHub; its setup cell installs what Colab lacks. Then upload `data/` with Colab's Files panel. While this repository is private, Colab cannot download from it; once it is public, set `REPO_RAW` in the notebook's second code cell to its raw URL instead.

## Who said what (local page)

`app/` joins the three steps: upload the full recording, ASR's `results.jsonl` and VAD's `segments.csv`, press **Diarize**, and each segment gets a speaker from Community-1, run once on the whole recording. Rename speakers, click a row to play it, and download the result as CSV or RTTM.

Two parts, on the same machine:

```bash
uv sync                       # once
uv run python app/server.py   # the diarization server, native: reads HF_TOKEN from .env, port 8001
docker compose up --build     # the website, port 8000: open http://127.0.0.1:8000
```

The website container only serves the page; its **Diarize** button forwards the audio to the native server, so pyannote runs at full speed and the token never enters Docker. If the native server is not running, the page says so. `uv run python app/server.py` alone also works: open http://127.0.0.1:8001.

The browser reads the transcripts and times itself; only the audio goes to the local server. A segment flagged "under 80% one speaker" may span a speaker change: listen to it. The audio must be 16-bit WAV. `data/clean/` has all three inputs.

## Data

### `data/lotusdis-S002_T062/`

- **Source:** [LOTUSDIS](https://huggingface.co/datasets/nectec/LOTUSDIS), a Thai far-field meeting corpus by NECTEC ([paper](https://arxiv.org/abs/2509.18722)), licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **Excerpt:** 60 s of session S002, from 395 s to 455 s of `Hijack_S002_T062`, converted to 16 kHz mono 16-bit WAV.
  - `distant.wav`: the 3 m Bluetooth microphone (`Hijack_S002_T062_BT3m.wav`).
  - `close.wav`: the close-talk lavalier microphone (`Hijack_S002_T062_Lav123.wav`), shifted by the measured device delay so it lines up with `distant.wav`.
- **`reference.csv`:** speech segments in seconds from the start of the excerpt. We labeled them on the close-talk microphone in Audacity. They are derived from LOTUSDIS, so they share its CC BY-SA 4.0 license.

### `data/lotusdis-S075/`

- **Source:** [LOTUSDIS](https://huggingface.co/datasets/nectec/LOTUSDIS), as above, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **`S075_300-360.wav`:** 60 s of session S075, from 300 s to 360 s, microphone Con123, cut sample-exact from the corpus's 24 kHz mono 16-bit WAV and not resampled.
- **`S075_300-360.reference.rttm`:** who spoke when, from the LOTUSDIS TextGrid of the session, cropped and shifted so the excerpt starts at 0.
- **`S075_300-360.hand-labels.txt`:** who spoke when, labeled by us in Audacity: one stretch per turn, not split at short pauses, and a second label only where a second voice is heard.
- Both label files are derived from LOTUSDIS, so they share its CC BY-SA 4.0 license.
