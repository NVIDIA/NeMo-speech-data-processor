import json
import os
import sys
import tempfile
from pathlib import Path

import hydra
import yaml
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict

from sdp.run_processors import run_processors, update_processor_imports
from sdp.utils import BootstrapProcessor


def _write_config(file_path: Path, dict_conf):
    with file_path.open("w") as file:
        yaml.dump(dict_conf, file)


def read_yaml(config_path=".", config_name="config"):
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
    return cfg


def make_dict(output_manifest_file, use_backend=None):
    workspace_dir = os.path.join(os.getenv('TEST_DATA_ROOT'), "armenian/audio_books/mp3")
    return {
        "processors_to_run": "0:",
        "use_backend": use_backend,
        "processors": [
            {
                "_target_": "sdp.processors.CreateInitialManifestByExt",
                "raw_data_dir": workspace_dir,
                "extension": "mp3",
                "output_file_key": "audio_filepath",
                "output_manifest_file": output_manifest_file,
            },
        ],
    }


def make_expected_output():
    workspace_dir = os.path.join(os.getenv('TEST_DATA_ROOT'), "armenian/audio_books/mp3")
    return {'audio_filepath': os.path.join(workspace_dir, "Eleonora/Eleonora30s.mp3")}


def test_curator():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output_manifest_file.jsonl")
        dict_conf = make_dict(output_manifest_file=output_path, use_backend="curator")
        conf_path = Path(tmpdir) / "config.yaml"
        _write_config(conf_path, dict_conf)

        cfg = OmegaConf.load(conf_path)

    run_processors(cfg)
    with open(output_path, "r") as f:
        output = json.load(f)

    expected_output = make_expected_output()

    assert output == expected_output, f"Expected {expected_output}, but got {output}"


def test_multiprocessing():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output_manifest_file.jsonl")
        dict_conf = make_dict(output_manifest_file=output_path, use_backend=None)
        conf_path = Path(tmpdir) / "config.yaml"
        _write_config(conf_path, dict_conf)

        cfg = OmegaConf.load(conf_path)

    run_processors(cfg)
    with open(output_path, "r") as f:
        output = json.load(f)

    expected_output = make_expected_output()
    assert output == expected_output, f"Expected {expected_output}, but got {output}"


def test_dask():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output_manifest_file.jsonl")
        dict_conf = make_dict(output_manifest_file=output_path, use_backend="dask")
        conf_path = Path(tmpdir) / "config.yaml"
        _write_config(conf_path, dict_conf)

        cfg = OmegaConf.load(conf_path)

    run_processors(cfg)
    with open(output_path, "r") as f:
        output = json.load(f)

    expected_output = make_expected_output()
    assert output == expected_output, f"Expected {expected_output}, but got {output}"
