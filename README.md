# Speech Data Processor (SDP) Toolkit

The Speech Data Processor (SDP) is a toolkit designed to simplify the processing of speech datasets. It minimizes the boilerplate code required and allows for easy sharing of processing steps. SDP's philosophy is to represent processing operations as 'processor' classes, which take in a path to a NeMo-style data manifest as input (or a path to the raw data directory if you do not have a NeMo-style manifest to start with), apply some processing to it, and then save the output manifest file.

## Features

- **Creating Manifests:** Generate manifests for your datasets.
- **Running ASR Inference:** Automatically run ASR inference to remove utterances where the reference text differs greatly from ASR predictions.
- **Text Transformations:** Apply text-based transformations to lines in the manifest.
- **Removing Inaccurate Transcripts:** Remove lines from the manifest which may contain inaccurate transcripts.
- **Custom Processors:** Write your own processor classes if the provided ones do not meet your needs.

## Installation

SDP is officially supported for Python 3.10, but might work for other versions.

1. Clone the repository:

```bash
   git clone https://github.com/NVIDIA/NeMo-speech-data-processor.git
   cd NeMo-speech-data-processor
```
2. Install dependencies:
```bash
   pip install -r requirements/main.txt
```

3. Optional: If you need to use ASR, NLP parts, or NeMo Text Processing, follow the NeMo installation instructions:
   - [NeMo Installation](https://github.com/NVIDIA/NeMo)

## Example:
1. In this example we will load librispeech using SDP.
   * For downloading all available data - replace config.yaml with all.yaml
   * For mini dataset - replace with mini.yaml.
```bash
    python NeMo-speech-data-processor/main.py \
    --config-path="dataset_configs/english/librispeech" \
    --config-name="config.yaml" \
    processors_to_run="0:" \
    workspace_dir="librispeech_data_dir"
```
## Usage

1. Create a Configuration YAML File:

   Here is a simplified example of a `config.yaml` file:

   ```yaml
   processors:
     - _target_: sdp.processors.CreateInitialManifestMCV
       output_manifest_file: "${data_split}_initial_manifest.json"
       language_id: es
     - _target_: sdp.processors.ASRInference
       pretrained_model: "stt_es_quartznet15x5"
     - _target_: sdp.processors.SubRegex
       regex_params_list:
         - {"pattern": "¡", "repl": "."}
         - {"pattern": "ó", "repl": "o"}
       test_cases:
         - {input: {text: "hey!"}, output: {text: "hey."}}
     - _target_: sdp.processors.DropNonAlphabet
       alphabet: "abcdefghijklmnopqrstuvwxyzáéiñóúüABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÑÓÚÜ"
       test_cases:
         - {input: {text: "test Тест ¡"}, output: null}
         - {input: {text: "test"}, output: {text: "test"}}
     - _target_: sdp.processors.KeepOnlySpecifiedFields
       output_manifest_file: "${data_split}_final_manifest.json"
       fields_to_keep:
         - "audio_filepath"
         - "text"
         - "duration"
   ```

2. Run the Processor:

   Use the following command to process your dataset:

```bash
   python <SDP_ROOT>/main.py \
     --config-path="dataset_configs/<lang>/<dataset>/" \
     --config-name="config.yaml" \
     processors_to_run="all" \
     data_split="train" \
     workspace_dir="<dir_to_store_processed_data>"
```

![SDP overview](https://github.com/NVIDIA/NeMo/releases/download/v1.17.0/sdp_overview_diagram.png)

To learn more about SDP, have a look at our [documentation](https://nvidia.github.io/NeMo-speech-data-processor/).


## Contributing
We welcome community contributions! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for the process.
