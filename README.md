# Thai speech notebooks

Hands-on companions to the team's speech manual: one notebook per section, each runnable in Colab or locally.

| Notebook | Section | Status |
|---|---|---|
| [vad.ipynb](vad.ipynb) | Voice activity detection | draft |
| `stt.ipynb` | Speech-to-text | not started |
| `diarization.ipynb` | Speaker diarization | not started |

## Rules

- **Licensed public audio only.** Every file in `data/` must carry an open license that allows redistribution, credited below. No company recordings, and nothing downloaded from YouTube or other sites without such a license.
- One notebook per section, kept running by that section's author.

## Run

- **Colab:** open the notebook from GitHub. Set `REPO_RAW` in its second code cell to this repository's raw URL, or upload `data/` with Colab's Files panel.
- **Locally:** from a clone, `uv venv && source .venv/bin/activate && uv pip install jupyterlab torch numpy pandas matplotlib`, then `jupyter lab`. Each notebook installs its own libraries in its first cell.

## Data

### `data/lotusdis-S002_T062/`

- **Source:** [LOTUSDIS](https://huggingface.co/datasets/nectec/LOTUSDIS), a Thai far-field meeting corpus by NECTEC ([paper](https://arxiv.org/abs/2509.18722)), licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **Excerpt:** 60 s of session S002, from 395 s to 455 s of `Hijack_S002_T062`, converted to 16 kHz mono 16-bit WAV.
  - `distant.wav`: the 3 m Bluetooth microphone (`Hijack_S002_T062_BT3m.wav`).
  - `close.wav`: the close-talk lavalier microphone (`Hijack_S002_T062_Lav123.wav`), shifted by the measured device delay so it lines up with `distant.wav`.
- **`reference.csv`:** speech segments in seconds from the start of the excerpt. We labeled them on the close-talk microphone in Audacity. They are derived from LOTUSDIS, so they share its CC BY-SA 4.0 license.
