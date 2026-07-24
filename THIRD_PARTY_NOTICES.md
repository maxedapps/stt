# Third-party notices

This file lists major third-party components used by **stt**.  
The **stt** application source code itself is licensed under the MIT License (see [`LICENSE`](LICENSE)).

Model weights are **not** shipped inside this git repository. On first run they are downloaded into the local Hugging Face cache under each model’s own license terms.

## Qwen3-ASR-1.7B (model weights)

- **Source:** [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) on Hugging Face  
- **Upstream project:** [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)  
- **License:** Apache License 2.0  
- **Role:** English (and multilingual) automatic speech recognition weights used at runtime  

## Qwen3-ForcedAligner-0.6B (model weights)

- **Source:** [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) on Hugging Face  
- **Upstream project:** [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)  
- **License:** Apache License 2.0  
- **Role:** Forced-alignment / timestamp weights used when producing timed captions  

## mlx-qwen3-asr (Python package)

- **Source:** [moona3k/mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr) (PyPI: `mlx-qwen3-asr`)  
- **License:** Apache License 2.0  
- **Role:** Apple Silicon (MLX) runtime used to load models and run transcription  

## Other runtime dependencies

Additional packages (for example **MLX**, **NumPy**, **huggingface-hub**, and their transitive dependencies) are installed by the package manager and retain their own licenses. See the environment created by `uv tool install` / `uv sync` for the exact set and versions.

## Trademarks

“Qwen”, “Tongyi Qianwen”, and related names are trademarks of their respective owners.  
Use of the open-weight models under Apache 2.0 does **not** grant trademark rights or imply endorsement by Alibaba or the Qwen team.

## Apache License 2.0 (reference)

Full text: https://www.apache.org/licenses/LICENSE-2.0  
Upstream LICENSE copies are also published with the Qwen3-ASR project and the `mlx-qwen3-asr` distribution.
