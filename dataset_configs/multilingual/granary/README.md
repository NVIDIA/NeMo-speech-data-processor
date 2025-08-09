## Granary Dataset Creation Pipeline

### Overview

This configuration drives the **Granary pseudo-labelling pipeline** – an open-source workflow that transforms large, noisy speech corpora into high-quality Automatic Speech Recognition (ASR) and Automatic Speech Translation (AST) training data for **25 European languages**.

The first public release of **Granary** (≈ 643 k h ASR / ≈ 351 k h AST) was built from three openly available corpora:

- [espnet/yodas2](https://huggingface.co/datasets/espnet/yodas2)
- [FBK-MT/mosel](https://huggingface.co/datasets/FBK-MT/mosel)
- [PleIAs/YouTube-Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons)

and is published as [nvidia/Granary](https://huggingface.co/datasets/nvidia/Granary).

> Note — Per-language runs
> 
> The pipeline is executed once per language pair: set
> - `source_lang` / `source_lang_full` – audio & transcript language
> - `translation.target_lang` / `target_lang_full` – translation language
> 
> For example, to obtain English audio with Italian translations choose `source_lang: en` and `translation.target_lang: it`. Separate runs are required for each additional language combination.

> Note — GPU required
> 
> All Whisper, vLLM and Comet-QE stages expect at least one CUDA-capable GPU. Multi-GPU nodes are auto-detected when `num_devices: -1` (default) is used.

### Software prerequisites

Install NeMo-speech-data-processor plus the extra wheels required by specific processors:

- `FasterWhisperInference`

```bash
pip install pytorch-lightning \
            "nvidia-cublas-cu12" \
            "nvidia-cudnn-cu12==9.*" \
            faster_whisper

export LD_LIBRARY_PATH=$(python - <<'PY'
import os, nvidia.cublas.lib, nvidia.cudnn.lib
print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))
PY)
```

- `vLLMInference`

```bash
pip install "optree>=0.13.0" vllm
```

- `CometoidWMTQualityEstimation`

```bash
pip install pymarian
```

- `FastTextLangIdClassifier`

```bash
pip install fasttext
```

- `ConvertToTarredAudioDataset` (optional, only if tar-sharding is enabled)

```bash
pip install lhotse "nemo-toolkit[common]==2.2.1"
```

### Quick start

1. **Hardware** – Linux box with NVIDIA GPU(s) and ≥ 16 GB VRAM (reference runs used A100-80 GB; smaller cards work with reduced batch sizes).
2. **Install** NeMo-speech-data-processor and the extras listed above.
3. **Prepare** the input manifest and set three mandatory YAML keys:
   - `input_manifest_file` – manifest with raw audio paths  
   - `output_dir` – working/output directory  
   - `sdp_dir` – root of the SDP tree (for prompt/regex assets)
4. **Run the pipeline**:

```bash
# Path to your local clone of NeMo-speech-data-processor
SDP_DIR=/path/to/NeMo-speech-data-processor

python ${SDP_DIR}/main.py \
    --config-path ${SDP_DIR}/dataset_configs/multilingual/granary/ \
    --config-name config.yaml \
    input_manifest_file=/path/to/input_manifest.json \
    output_dir=/path/to/output/dir \
    sdp_dir=${SDP_DIR}
```

### Input and output formats

#### Input manifest

Each line is a JSON object with the source-audio path:

```json
{"source_audio_filepath": "/path/to/file.flac"}
```

#### Key outputs

  - `${output_dir}/${source_lang}/manifest_46.json` – final bilingual manifest containing `audio_filepath`, `offset`, `duration`, `text` (source) and `answer` (translation), plus constant decoder flags.
  - `${output_dir}/${source_lang}/tarred_dataset/` – optional tarred-audio shards and `shard_manifest.json` when `convert_to_audio_tarred_dataset.should_run: True`.
  - All intermediate `manifest_XX.json` files are kept for audit/debug.

### Pipeline stages

The processors executed (indices match the config):

- **FfmpegConvert** (0) – re-encode audio to 16 kHz/mono FLAC.
- **GetAudioDuration** (1) – compute clip length.
- **RemoveFiles** (2) – optionally delete originals (`params.save_disk_space`).
- **FasterWhisperInference** (3) – pass 1 language detection.
- **LambdaExpression** (4) – probability-based LID filtering.
- **DropSpecifiedFields** (5) – remove temporary fields.
- **FasterWhisperInference** (6, 14) – two-pass transcription (second run can slice by offset).
- **Segmentation & grooming** (7–13) – split Whisper segments into atomic utterances.
- **Hallucination detection** (18–20) – drop repeated n-grams, garbage tokens and common filler phrases.
- **PnC restoration** (21–23) – `Qwen-2.5-7B` restores punctuation & capitalisation; optional regex clean-up.
- **Length & charset filtering** (27–36) – word-ratio, character histogram and FastText checks.
- **Quality estimation** (41–43) – keep pairs with `Comet-QE score ≥ min_qe_score`.
- **Constant flags** (44) – add decoder directives (`<|emo:undefined|>`, `itn`, `pnc`, etc.).
- **Tarred dataset** (46) – shard audio into `num_shards` tar files (optional).

### Tunable parameters

All knobs live under the `params` block.

- **Language**
  - `source_lang` / `source_lang_full`
  - `translation.target_lang` / `target_lang_full`

- **Audio duration**
  - `min_audio_duration` – drop very short clips (seconds)
  - `max_audio_duration` – drop very long clips (seconds)

- **Language-ID & text filtering**
  - `min_audio_lid_probability` – Whisper LID threshold
  - `translation.min_hist_token_ratio` – charset-purity ratio
  - `translation.min_text_lid_probability` – FastText LID threshold

- **Length & quality**
  - `translation.max_len_diff_ratio` – max(src / tgt) word ratio
  - `translation.min_qe_score` – Comet-QE acceptance score

- **Tarred dataset**
  - `convert_to_audio_tarred_dataset.should_run` (bool)
  - `num_shards` and `buckets_num` – shard layout

- **Misc.**
  - `use_regex` – regex preset for text normalisation
  - `save_disk_space` – delete originals after conversion
  - `use_dask` – enable distributed execution (not recommended)

### Advanced usage

- **Selective execution** – override `processors_to_run` with a range of indices, e.g. `"0:25"`.
- **Model swapping** – every inference processor exposes either `model_size_or_path` (Whisper) or an embedded `model:` block (vLLM).
- **Resource tuning** – `num_devices = -1` uses all visible GPUs; set an integer to pin workers per stage.

### References

- Koluguri et al. (2025). Granary: Speech Recognition and Translation Dataset in 25 European Languages (preprint). arXiv: [2505.13404](https://arxiv.org/abs/2505.13404),
- [nvidia/Granary](https://huggingface.co/datasets/nvidia/Granary) dataset on Hugging Face,
- NeMo-SDP source [code](https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/multilingual/granary/>). 