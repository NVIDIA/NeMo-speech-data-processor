import pytest
import ndjson
import boto3
import json
import os
import tarfile
from pathlib import Path
from omegaconf import OmegaConf
from sdp.run_processors import run_processors

DATASET_CONFIGS_ROOT = Path(__file__).parents[1] / "dataset_configs"

@pytest.fixture
def get_tts_ytc_data(tmpdir: str):
    # Download the data from S3
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY")
    )
    s3.download_file(
       "sdp-test-data",
       "test_data/tts/ytc/test_data_reference.json",
       tmpdir/"test_data_reference.json",
    )

    s3.download_file(
       "sdp-test-data",
       "test_data/tts/ytc/ytc.en.tar.gz",
       tmpdir/"ytc.en.tar.gz",
    )

    # Extract the tar.gz file
    with tarfile.open(tmpdir/"ytc.en.tar.gz", "r:gz") as tar:
        tar.extractall(tmpdir)

    audio_files = Path(tmpdir).glob("audios/*")
    with open(os.path.join(tmpdir, "input_manifest.jsonl"), "w") as f:
        for audio_file in audio_files:
            data = {
                "audio_filepath": f"{tmpdir}/audios/{audio_file.name}",
                "audio_item_id": audio_file.stem,
            }
            f.write(json.dumps(data) + "\n")

    return tmpdir

def test_tts_sdp_end_to_end(get_tts_ytc_data):
    data_dir = get_tts_ytc_data
    assert os.path.exists(data_dir)
    config_path = DATASET_CONFIGS_ROOT / "tts/ytc/config.yaml"
    input_manifest_file = os.path.join(data_dir, "input_manifest.jsonl")
    reference_manifest_file = os.path.join(data_dir, "test_data_reference.json")

    cfg = OmegaConf.load(config_path)
    cfg.hf_token = os.getenv("HF_SECRET_KEY")
    cfg.final_manifest = os.path.join(data_dir, "output_manifest.jsonl")
    cfg.raw_audio_dir = os.path.join(data_dir, "audios")
    cfg.data_split = "train"
    cfg.device = "cpu"
    cfg.language_short = "en"
    cfg.processors[3].model_name = "nvidia/stt_en_fastconformer_ctc_large"
    cfg.processors[3].parakeet = False
    cfg.processors[3].ctc = True
    cfg.processors[0].input_manifest_file = input_manifest_file

    run_processors(cfg)

    assert os.path.exists(cfg.final_manifest)
    output_file_data = {}
    with open(cfg.final_manifest, "r") as f:
        output_data = ndjson.load(f)
        for item in output_data:
            output_file_data[item["audio_item_id"]] = item
    
    reference_file_data = {}
    with open(reference_manifest_file, "r") as f:
        reference_data = ndjson.load(f)
        for item in reference_data:
            reference_file_data[item["audio_item_id"]] = item
    
    assert len(output_file_data) == len(reference_file_data)
    assert len(output_file_data) == 2
    for audio_item_id in output_file_data:
        assert output_file_data[audio_item_id]["segments"] == reference_file_data[audio_item_id]["segments"]

