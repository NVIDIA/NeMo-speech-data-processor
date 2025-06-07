# 🧠 TopIPL: Iterative Pseudo-Labeling for ASR

TopIPL is an **iterative pseudo-labeling algorithm** designed for training ASR models using both labeled and unlabeled data. It maintains a **dynamic pseudo-label cache** and leverages **top-N averaged checkpoints** as a teacher model to generate high-quality pseudo-labels across training iterations.

## 📦 Contents

- `NemoRunIPLProcessor` — Command generator and job submitter for IPL runs, compatible with local and cluster environments.
- `nemo_run_config.yaml` — Main configuration file. Users should define all required paths and parameters here.

## 🚀 Getting Started

TopIPL runs like any other processor in the `nemo_run` framework. To use it, you must pass:

- `output_manifest_file`: Path where the resulting manifest will be saved.
- `nemo_run_config`: YAML file containing IPL setup, training/inference configs, and NeMo-Run settings.

### 🔧 Training Config Requirements

Your training config must:

```yaml
exp_manager:
  create_ipl_epoch_stopper_callback: True
```
If you're not using Lhotse, also include:

```yaml
ipl_epoch_stopper_callback_params:
stop_every_n_epochs: 2

```

### Prerequisites
- nemo_run
- `pip install -r ipl.txt`

### Running the Code

```bash
python main.py --config-path=/path/to/directory/config --config-name=config.yaml