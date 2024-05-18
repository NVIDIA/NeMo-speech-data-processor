# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import subprocess
import tarfile
import urllib
import zipfile
from pathlib import Path
from typing import Dict, List, Union

import wget

from sdp.logging import logger


def load_manifest(manifest: Path) -> List[Dict[str, Union[str, float]]]:
    # read NeMo manifest as a list of dicts
    result = []
    with manifest.open() as f:
        for line in f:
            data = json.loads(line)
            result.append(data)
    return result


def download_file(source_url: str, target_directory: str, verbose=True):
    # make sure target_directory is an absolute path to avoid bugs when we change directories to download data later
    target_directory = os.path.abspath(target_directory)

    if verbose:
        logger.info(f"Trying to download data from {source_url} and save it in this directory: {target_directory}")
    filename = os.path.basename(urllib.parse.urlparse(source_url).path)
    target_filepath = os.path.join(target_directory, filename)

    if os.path.exists(target_filepath):
        if verbose:
            logger.info(f"Found file {target_filepath} => will not be attempting download from {source_url}")
    else:
        logger.info(f"Not found file {target_filepath}")
        original_dir = os.getcwd()  # record current working directory so can cd back to it
        os.chdir(target_directory)  # cd to target dir so that temporary download file will be saved in target dir

        wget.download(source_url, target_directory)

        # change back to original directory as the rest of the code may assume that we are in that directory
        os.chdir(original_dir)
        if verbose:
            logger.info("Download completed")

    return target_filepath


def extract_archive(archive_path: str, extract_path: str, force_extract: bool = False) -> str:
    logger.info(f"Attempting to extract all contents from tar file {archive_path} and save in {extract_path}")
    if not force_extract:
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r") as archive:
                archive_extracted_dir = os.path.dirname(archive.getnames()[0])
        elif zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive_extracted_dir = archive.namelist()[0]
        else:
            raise RuntimeError(f"Unknown archive format: {archive_path}. We only support tar and zip archives.")

        archive_contents_dir = os.path.join(extract_path, archive_extracted_dir)

    if not force_extract and os.path.exists(archive_contents_dir):
        logger.info(f"Directory {archive_contents_dir} already exists => will not attempt to extract file")
    else:
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r") as archive:
                archive.extractall(path=extract_path)
        elif zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(extract_path)
        logger.info("Finished extracting")

    if force_extract:
        return None
    return archive_contents_dir


def ffmpeg_convert(jpg: str, wav: str, ar: int = 0, ac: int = 1):
    process_args = ["ffmpeg", "-nostdin", "-i", jpg, '-ac', str(ac), "-map", "0:a", "-c:a", "pcm_s16le", "-y", wav]
    if ar:
        process_args = process_args[:-1]
        process_args.extend(["-ar", str(ar), wav])
    return subprocess.run(process_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
