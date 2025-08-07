import pytest
import boto3
import json
import os
from pathlib import Path
from omegaconf import OmegaConf
from sdp.run_processors import run_processors
import shutil

DATASET_CONFIGS_ROOT = Path(__file__).parents[1] / "dataset_configs"

@pytest.fixture
def granary_data(tmp_path: Path):
    # Download the data from S3
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY")
    )

    granary_key_prefix = "test_data/granary"

    file_keys_to_download = [
        f"{granary_key_prefix}/reference_manifest.json",
        f"{granary_key_prefix}/manifest_03.json",
        f"{granary_key_prefix}/manifest_06.json",
        f"{granary_key_prefix}/manifest_14.json",
        f"{granary_key_prefix}/manifest_21.json",
        f"{granary_key_prefix}/manifest_26.json",
        f"{granary_key_prefix}/manifest_34.json",
        f"{granary_key_prefix}/manifest_39.json",
        f"{granary_key_prefix}/audio/zCW0Pa0BI4Q.wav",
        f"{granary_key_prefix}/audio/zHWk3Ae7qJ0.wav",
        f"{granary_key_prefix}/audio/zHtFdl5K8qg.wav",
        f"{granary_key_prefix}/audio/zCW9rGbaF4E.wav",
        f"{granary_key_prefix}/audio/zG3RpHaMzkQ.wav",
    ]

    input_manifest_file = tmp_path / "input_manifest.json"
    with open(input_manifest_file, 'w', encoding="utf8") as f:
        for file_key in file_keys_to_download:
            rel_path   = file_key.replace(granary_key_prefix + "/", "")
            dest_path  = tmp_path / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            s3.download_file("sdp-test-data", file_key, str(dest_path))

            if file_key.endswith(".wav"):
                f.write(json.dumps({"source_audio_filepath": str(dest_path)}) + "\n")

    return tmp_path

def test_granary_pipeline_end_to_end(granary_data):
    assert os.path.exists(granary_data)
    config_path = DATASET_CONFIGS_ROOT / "multilingual/granary/config.yaml"
    input_manifest_file = os.path.join(granary_data, "input_manifest.json")
    reference_manifest = os.path.join(granary_data, "reference_manifest.json")

    cfg = OmegaConf.load(config_path)
    
    cfg.input_manifest_file = input_manifest_file
    cfg.output_dir = os.path.join(granary_data, "sdp_output")
    cfg.sdp_dir = Path(__file__).parents[1]

    #disable some processors
    ## step 3: FasterWhisperInference 
    cfg.processors[3].should_run = False
    cfg.processors[4].input_manifest_file = os.path.join(granary_data, "manifest_03.json")

    ## step 14: FasterWhisperInference 
    cfg.processors[6].should_run = False
    cfg.processors[7].input_manifest_file = os.path.join(granary_data, "manifest_06.json")

    ## step 21: FasterWhisperInference 
    cfg.processors[14].should_run = False
    cfg.processors[15].input_manifest_file = os.path.join(granary_data, "manifest_14.json")

    ## step 21: vLLMInference 
    cfg.processors[21].should_run = False
    cfg.processors[22].input_manifest_file = os.path.join(granary_data, "manifest_21.json")

    ## step 26: vLLMInference 
    cfg.processors[26].should_run = False
    cfg.processors[27].input_manifest_file = os.path.join(granary_data, "manifest_26.json")
    
    ## steps 33-34: CharacterHistogramLangValidator
    cfg.processors[33].should_run = False
    cfg.processors[34].should_run = False
    cfg.processors[35].input_manifest_file = os.path.join(granary_data, "manifest_34.json")

    ## step 39: CometoidWMTQualityEstimation
    cfg.processors[39].should_run = False
    cfg.processors[40].input_manifest_file = os.path.join(granary_data, "manifest_39.json")

    run_processors(cfg)

    final_manifest = cfg.processors[-1].output_manifest_file
    fields_to_ignore = ['audio_filepath']

    with open(reference_manifest, "rt", encoding="utf8") as reference_fin, \
         open(final_manifest, "rt", encoding="utf8") as generated_fin:
        reference_lines = sorted(reference_fin.readlines())
        generated_lines = sorted(generated_fin.readlines())
        assert len(reference_lines) == len(generated_lines)

        for reference_line, generated_line in zip(reference_lines, generated_lines):
            reference_data = json.loads(reference_line)
            generated_data = json.loads(generated_line)
            for field in fields_to_ignore:
                reference_data.pop(field, None)
                generated_data.pop(field, None)
            assert reference_data == generated_data
    
    if os.getenv("CLEAN_UP_TMP_PATH", "0") != "0":
        shutil.rmtree(granary_data)




    

   
    
    


