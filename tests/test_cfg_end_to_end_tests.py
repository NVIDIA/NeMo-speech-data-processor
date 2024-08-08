# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import shutil
import tarfile
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple
from unittest import mock

import pytest
from omegaconf import OmegaConf

import sdp.processors.datasets.coraal.create_initial_manifest as coraal_processor
from sdp.run_processors import run_processors
from sdp.utils.common import extract_tar_with_strip_components

DATASET_CONFIGS_ROOT = Path(__file__).parents[1] / "dataset_configs"

def data_check_fn_generic(raw_data_dir: str, file_name: str, **kwargs) -> None:
    if callable(file_name):
        file_name = file_name(**kwargs)
    expected_file = Path(raw_data_dir) / file_name
    if not expected_file.exists():
        raise ValueError(f"No such file {str(expected_file)}")

data_check_fn_mls = partial(data_check_fn_generic, file_name=lambda language, **kwargs: f"mls_{language}.tar.gz")
data_check_fn_mcv = partial(data_check_fn_generic, file_name=lambda archive_file_stem, **kwargs: f"{archive_file_stem}.tar.gz")
data_check_fn_mtedx = partial(data_check_fn_generic, file_name=lambda language_id, **kwargs: f"mtedx_{language_id}.tgz")
data_check_fn_coraa = partial(data_check_fn_generic, file_name="train_dividido/train.part1.rar")
data_check_fn_slr102 = partial(data_check_fn_generic, file_name="slr102_kk.tar.gz")
data_check_fn_ksc2 = partial(data_check_fn_generic, file_name="ksc2_kk.tar.gz")
data_check_fn_librispeech = partial(data_check_fn_generic, file_name="dev-clean.tar.gz")
data_check_fn_fleurs = partial(data_check_fn_generic, file_name="dev.tar.gz")

def data_check_fn_voxpopuli(raw_data_dir: str) -> None:
    """Raises error if do not find expected data.

    Will also extract the archive as initial processor expects extracted data.
    """
    if (Path(raw_data_dir) / "transcribed_data").exists():
        return
    expected_file = Path(raw_data_dir) / "transcribed_data.tar.gz"
    if not expected_file.exists():
        raise ValueError(f"No such file {str(expected_file)}")
    with tarfile.open(expected_file, 'r:gz') as tar:
        tar.extractall(path=raw_data_dir)
        
def data_check_fn_slr140(raw_data_dir: str) -> None:
    """Raises error if do not find expected data.
    Will also extract the archive as initial processor expects extracted data.
    """
    tgt_dir = Path(raw_data_dir)

    expected_file = Path(raw_data_dir) / f"slr140_kk.tar.gz"
    if not expected_file.exists():
        raise ValueError(f"No such file {str(expected_file)}")

    extract_tar_with_strip_components(expected_file, tgt_dir, strip_components=1)

# using Mock so coraal_processor will only try to use the files listed.
# To reduce the amount of storage required by the test data, the S3 bucket contains
# modified versions of LES_audio_part01_2021.07.tar.gz and
# LES_textfiles_2021.07.tar.gz which only contain data from 2 recordings

coraal_processor.get_coraal_url_list = mock.Mock(
    return_value=[
        'http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2021.07.txt',
        'http://lingtools.uoregon.edu/coraal/les/2021.07/LES_audio_part01_2021.07.tar.gz',
        'http://lingtools.uoregon.edu/coraal/les/2021.07/LES_textfiles_2021.07.tar.gz',
    ]
)

def get_test_cases() -> List[Tuple[str, Callable]]:
    return [
        (f"{DATASET_CONFIGS_ROOT}/spanish/mls/config.yaml", partial(data_check_fn_mls, language="spanish")),
        (f"{DATASET_CONFIGS_ROOT}/spanish_pc/mcv12/config.yaml", partial(data_check_fn_mcv, archive_file_stem="cv-corpus-12.0-2022-12-07-es")),
        (f"{DATASET_CONFIGS_ROOT}/italian/voxpopuli/config.yaml", data_check_fn_voxpopuli),
        (f"{DATASET_CONFIGS_ROOT}/italian/mls/config.yaml", partial(data_check_fn_mls, language="italian")),
        (f"{DATASET_CONFIGS_ROOT}/portuguese/mls/config.yaml", partial(data_check_fn_mls, language="portuguese")),
        (f"{DATASET_CONFIGS_ROOT}/portuguese/mcv/config.yaml", partial(data_check_fn_mcv, archive_file_stem="cv-corpus-15.0-2023-09-08-pt")),
        (f"{DATASET_CONFIGS_ROOT}/portuguese/mtedx/config.yaml", partial(data_check_fn_mtedx, language_id="pt")),
        (f"{DATASET_CONFIGS_ROOT}/portuguese/coraa/config.yaml", data_check_fn_coraa),
        (f"{DATASET_CONFIGS_ROOT}/english/slr83/config.yaml", lambda raw_data_dir: True),
        (f"{DATASET_CONFIGS_ROOT}/english/coraal/config.yaml", lambda raw_data_dir: True),
        (f"{DATASET_CONFIGS_ROOT}/english/librispeech/config.yaml", data_check_fn_librispeech),
        (f"{DATASET_CONFIGS_ROOT}/armenian/fleurs/config.yaml", data_check_fn_fleurs),
        (f"{DATASET_CONFIGS_ROOT}/armenian/text_mcv/config.yaml", lambda raw_data_dir: True),
        (f"{DATASET_CONFIGS_ROOT}/armenian/audio_books/config.yaml", lambda raw_data_dir: True),
        (f"{DATASET_CONFIGS_ROOT}/kazakh/mcv/config.yaml", partial(data_check_fn_mcv, archive_file_stem="mcv_kk")),
        (f"{DATASET_CONFIGS_ROOT}/kazakh/slr140/config.yaml", data_check_fn_slr140),
        (f"{DATASET_CONFIGS_ROOT}/kazakh/slr102/config.yaml", data_check_fn_slr102),
        (f"{DATASET_CONFIGS_ROOT}/kazakh/ksc2/config.yaml", data_check_fn_ksc2),
    ]

def check_e2e_test_data() -> bool:
    """
    Checks if required environment variables are defined for e2e data.
    Either TEST_DATA_ROOT needs to be defined or both AWS_SECRET_KEY
    and AWS_ACCESS_KEY.
    """
    return bool(os.getenv("TEST_DATA_ROOT") or (os.getenv("AWS_SECRET_KEY") and os.getenv("AWS_ACCESS_KEY")))

def get_e2e_test_data_path(rel_path_from_root: str) -> str:
    """Returns path to e2e test data (downloading from AWS if necessary).
    In case of downloading from AWS, will create "test_data" folder in the
    current folder and set TEST_DATA_ROOT automatically (used by the sdp code
    to locate test data).
    """
    test_data_root = os.getenv("TEST_DATA_ROOT") 
    if test_data_root: # assume it's present locally
        return test_data_root

    import boto3
    import logging

    s3_resource = boto3.resource(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    )
    bucket = s3_resource.Bucket("sdp-test-data")
    
    logging.info(f"Downloading test data for {rel_path_from_root} from s3")
    for obj in bucket.objects.all():
        if obj.key.endswith("/"):  # do not try to "download_file" on objects which are actually directories
            continue
        if rel_path_from_root in obj.key: 
            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key)
    logging.info(f"Test data downloaded to 'test_data/{rel_path_from_root}' folder.")

    return os.path.abspath("test_data")

@pytest.mark.skipif(
    not check_e2e_test_data(),
    reason="Either TEST_DATA_ROOT needs to be defined or both AWS_SECRET_KEY "
    "and AWS_ACCESS_KEY to run e2e config tests",
)
@pytest.mark.parametrize("config_path,data_check_fn", get_test_cases())
def test_configs(config_path: str, data_check_fn: Callable, tmp_path: Path):
    # we expect DATASET_CONFIGS_ROOT and TEST_DATA_ROOT
    # to have the same structure (e.g. <lang>/<dataset>)
    rel_path_from_root = Path(config_path).parent.relative_to(DATASET_CONFIGS_ROOT)
    test_data_root = Path(get_e2e_test_data_path(str(rel_path_from_root)))
 
    # run data_check_fn - it will raise error if the expected test data is not found
    try:
        data_check_fn(raw_data_dir=str(test_data_root / rel_path_from_root))
    except ValueError as e:
        pytest.skip(f"Test data not available: {str(e)}")

    reference_manifest = test_data_root / rel_path_from_root / "test_data_reference.json"
    if not reference_manifest.exists():
        pytest.skip(f"Reference manifest not found: {reference_manifest}")

    cfg = OmegaConf.load(config_path)
    assert "processors" in cfg
    cfg.processors_to_run = "all"
    cfg.workspace_dir = str(tmp_path)
    cfg.final_manifest = str(tmp_path / "final_manifest.json")
    cfg.data_split = cfg.get("data_split", "train")
    cfg.processors[0].raw_data_dir = str(test_data_root / rel_path_from_root)

    if "already_downloaded" in cfg["processors"][0]:
        cfg["processors"][0]["already_downloaded"] = True

    run_processors(cfg)
    # additionally, let's test that final generated manifest matches the
    # reference file (ignoring the file paths)
    
    with open(reference_manifest, "rt", encoding="utf8") as reference_fin, \
         open(cfg.final_manifest, "rt", encoding="utf8") as generated_fin:
        reference_lines = sorted(reference_fin.readlines())
        generated_lines = sorted(generated_fin.readlines())
        assert len(reference_lines) == len(generated_lines)

        for reference_line, generated_line in zip(reference_lines, generated_lines):
            reference_data = json.loads(reference_line)
            generated_data = json.loads(generated_line)
            reference_data.pop("audio_filepath", None)
            generated_data.pop("audio_filepath", None)
            assert reference_data == generated_data

 # if CLEAN_UP_TMP_PATH is set to non-0 value, we will delete tmp_path
    if os.getenv("CLEAN_UP_TMP_PATH", "0") != "0":
        shutil.rmtree(tmp_path)

# Additional unit tests to increase coverage
def test_check_e2e_test_data():
    os.environ.clear()
    assert not check_e2e_test_data()
    os.environ["TEST_DATA_ROOT"] = "/path/to/test/data"
    assert check_e2e_test_data()
    os.environ.clear()
    os.environ["AWS_SECRET_KEY"] = "secret"
    os.environ["AWS_ACCESS_KEY"] = "access"
    assert check_e2e_test_data()

@pytest.mark.slow
def test_get_e2e_test_data_path(tmp_path):
    os.environ["TEST_DATA_ROOT"] = str(tmp_path)
    assert get_e2e_test_data_path("test/path") == str(tmp_path)

    os.environ.clear()
    os.environ["AWS_SECRET_KEY"] = "secret"
    os.environ["AWS_ACCESS_KEY"] = "access"
    with mock.patch("boto3.resource") as mock_resource:
        mock_bucket = mock.MagicMock()
        mock_resource.return_value.Bucket.return_value = mock_bucket
        mock_bucket.objects.all.return_value = [
            mock.MagicMock(key="test/path/file1.txt"),
            mock.MagicMock(key="test/path/file2.txt"),
        ]
        result = get_e2e_test_data_path("test/path")
        assert result == os.path.abspath("test_data")
        assert mock_bucket.download_file.call_count == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--durations=0"])
