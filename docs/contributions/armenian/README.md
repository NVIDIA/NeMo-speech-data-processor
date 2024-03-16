# General Info
- Contributor/Author: Ara Yeroyan | `ara_yeroyan@edu.aua.am`
- Supervisor/Reviewer: Nikolay Karpov | `nkarpov@nvidia.com`
- Project: **Armenian ASR**

# Setup

## Run:

Install the environment requirements via:

```bash
install_packages.sh 
```

If you are a contributor and/or want to use the ASR models from `sdp/processor/huggingface/speech_recognition.py` as well, then

```bash
install_packages.sh contribute asr_models
```
<hr>

Project runs through hydra => change the config through using `+` and predefined name - `config_name` to change between various datasets' configs.
If want to switch to another language or another directory (currently `dataset_configs/armenian`) overwrite the `config_path` as well.

For **Contributions**

```bash
python main.py +config_name=mcv.yaml
```

## Data

### 1) Mozilla Common Voice:
 - Extract your desired dataset (https://commonvoice.mozilla.org/en/datasets)
 - Either unzip and place in the `data/original` or change the `mcv.yaml`'s **already_extracted** to **False**
 - Create manifests folder in the `data/manifests` or adjust the **output_manifest_file** to your custom case

### 2) Armenian Books:
 - To Do: https://grqaser.org/am


## Insights:


### 1. Configure sdp.processors.DropNonAlphabet

#### <font color='red'>Problem</font> - new language => new punctuation
#### <font color='lightgreen'>Solution</font> - run `main.py` to extract the punctuations according to which the processor filtered the texts
![Example Image](imgs/debug_alphabet.JPG)

<div align="center" style="background-color: green; color: black; padding: 0.01px;">
  <p><u><b>Here on the middle section can notice the filtering from 5.12hours to 0.06</b></u></p>
</div>