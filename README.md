# stt — local English speech-to-text for Apple Silicon

Turn one audio or video file into a punctuated transcript and captions on your Mac. Runs entirely locally after the first model download.

## What you get

For an input like `interview.mp4`, `stt` writes four files:

| File | What it is |
|---|---|
| `interview.txt` | Full transcript |
| `interview.srt` | Captions (SubRip) — use this in editors / YouTube |
| `interview.vtt` | Captions (WebVTT) |
| `interview.words.json` | Timed words/units with punctuation (advanced) |

English only. One file per run.

## Requirements

- **Apple Silicon Mac** (M1 / M2 / M3 / M4 …) — Intel Macs are not supported
- **macOS** with a normal Metal/GPU setup
- [**uv**](https://docs.astral.sh/uv/) (installs the app and the right Python)
- **ffmpeg** on your PATH (`brew install ffmpeg`)
- About **6 GB** free disk for speech models (downloaded automatically on first use)
- Network on **first run** (or a copied model cache — see below)

## Install (any Mac user)

```bash
# 1) system tools
brew install ffmpeg uv

# 2) get this project
git clone https://github.com/maxedapps/stt.git
cd stt

# 3) install the global `stt` command (~/.local/bin/stt)
uv tool install --editable .
```

If the shell says `stt: command not found`:

```bash
uv tool update-shell
# open a new terminal, or ensure ~/.local/bin is on your PATH
```

Check:

```bash
stt --help
```

### Update

```bash
cd stt
git pull
uv tool install --force --editable .
```

### Uninstall

```bash
uv tool uninstall stt
```

## Usage

From **any folder**:

```bash
stt /path/to/video.mp4 -o ./out
```

### Options

| Flag | Meaning |
|---|---|
| `INPUT` | One audio/video file ffmpeg can read |
| `-o` / `--output-dir` | Where to write outputs (default: current folder; created if missing) |
| `--overwrite` | Replace existing output files with the same names |
| `--terms PATH` | Optional keywords file (default: `./terms.txt` if it exists) |

Examples:

```bash
# basic
stt lecture.mov -o ./captions

# replace a previous run
stt lecture.mov -o ./captions --overwrite

# bias spelling of names / product terms
stt lecture.mov -o ./captions --terms ./terms.txt
```

On success, the four output paths are printed (one per line). Progress messages go to stderr.

**First run** can take a while: it downloads ~6 GB of models, then loads them and transcribes. Later runs reuse the cache and are much snappier to start (long videos still take time to process).

### Keywords (`terms.txt`)

Optional. Helps with names and jargon (soft bias, not a guarantee).

- One term per line  
- A term may contain spaces (`Maximilian Schwarzmüller`)  
- Blank lines and lines starting with `#` are ignored  

```text
# terms.txt
Academind
App Router
Maximilian Schwarzmüller
fal.ai
```

- If you don’t pass `--terms`, `stt` looks for `./terms.txt` in the current folder  
- Missing default file → continues without terms  
- If you pass `--terms some/path.txt` and it’s missing → error  
- If Qwen emits the supplied vocabulary verbatim, `stt` rejects that contaminated pass and retries once without terms

### Offline use (after first download)

```bash
HF_HUB_OFFLINE=1 stt lecture.mov -o ./captions --overwrite
```

## Sharing this tool with other Mac users

Send them this repo and the **Install** section above.

They need:

1. An Apple Silicon Mac  
2. Access to the git repo (public URL, or invite if private)  
3. `brew install ffmpeg uv`  
4. `uv tool install --editable .` from a clone  

They do **not** need your project folders, Python expertise, or a cloud API key.

### Optional: skip the big download on their machine

After you’ve run `stt` once, models live in the Hugging Face cache (usually `~/.cache/huggingface/hub/`). You can copy these folders to the same place on another Mac:

- `models--Qwen--Qwen3-ASR-1.7B`
- `models--Qwen--Qwen3-ForcedAligner-0.6B`

Then their first transcription can run with less or no network (`HF_HUB_OFFLINE=1` once the cache is complete).

## What to expect

- **Local & private** after models are cached — audio isn’t sent to a caption SaaS  
- **English** speech works best; other languages are out of scope  
- **Punctuation and capitalization** are kept in transcript and captions  
- Long-file aligner timestamps that overshoot an internal audio-chunk boundary are safely bounded to that chunk; unrelated malformed timing still fails closed
- Output files are written safely (no half-written caption files left behind)  
- If an output name already exists, `stt` refuses to overwrite unless you pass `--overwrite`  
- It will never overwrite your input media file  

## Troubleshooting

| Problem | What to try |
|---|---|
| `stt: command not found` | `uv tool install --editable .` from the repo; ensure `~/.local/bin` is on `PATH` (`uv tool update-shell`) |
| Not Apple Silicon / MLX errors | This tool only supports Apple Silicon Macs |
| `ffmpeg` errors | `brew install ffmpeg` and confirm `ffmpeg -version` |
| First run very slow | Normal — model download + load; watch stderr progress lines |
| Cache miss with offline mode | Run once online, or copy the HF model folders (see above) |
| Existing outputs | Add `--overwrite` or choose another `-o` directory |
| Names misspelled | Add them to `terms.txt` and pass `--terms` |
| `detected verbatim domain-context echo` | `stt` automatically retries once without terms so the echoed vocabulary is not published |

## Limits (by design)

No batch folder mode, live microphone streaming, speaker labels, language selection, or cloud/API server in this tool. One English file in → four caption/transcript files out.

## License and acknowledgments

**stt** (this application’s source code) is released under the [MIT License](LICENSE).

At runtime it uses third-party components that keep their own licenses:

| Component | License | What it’s for |
|---|---|---|
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Apache 2.0 | Speech recognition model (downloaded on first use) |
| [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | Apache 2.0 | Word/unit timestamps (downloaded on first use) |
| [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr) | Apache 2.0 | MLX inference runtime on Apple Silicon |

Those model weights are **not** included in this git repo; Hugging Face serves them under Apache 2.0 when `stt` downloads them into your local cache. Using this tool does not grant Qwen/Alibaba trademark rights or imply their endorsement.

Full third-party notes: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
