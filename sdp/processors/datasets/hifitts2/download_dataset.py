# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import librosa
from pathlib import Path
import soundfile as sf
import time
import urllib.error
import urllib.request

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class DownloadHiFiTTS2(BaseParallelProcessor):
    """
    Downloads HiFiTTS-2 dataset to local machine. Unsegmented audiobook chapters are first downloaded at a
    48 kHz from LibriVox. Each chapter is then split into segmented utterance files based on precomputed
    offsets and durations.

    To reduce disk use, the chapter files can be optionally deleted after they are segmented.

    Metadata for chapters which fail to download due to network errors are stored in an output manifest file,
    which can be given as input to this processor to attempt the downloads again.

    Args:
        audio_dir (str): Root directory where utterance files will be saved.
        chapter_dir (str): Root directory where audiobook chapter files will be saved.
        sample_rate (int): Sample rate to use for utterance files.
        delete_chapter_files (bool): Whether to delete each chapter file after it is done being processed.
        exit_on_error (bool): Whether to terminate the entire processor script if a single chapter downlaod fails.
        num_retries (int): Number of times to retry chapter download after encountering intermittent HTTP errors.

    Returns:
        Utterance files are stored under 'audio_dir' and chapter files are downloaded under 'chapter_dir'.

        If exit_on_error is False, then an output manifest will be saved with manifest entries that fail to downlaod,
        with error information stored under the 'error_code' and 'error_reason' fields.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.DownloadHiFiTTS2
              input_manifest_file: ${workspace_dir}/manifest_22khz.json
              output_manifest_file: ${workspace_dir}/errors_22khz.json
              audio_dir: ${workspace_dir}/audio_22khz
              chapter_dir: ${workspace_dir}/chapters
              max_workers: 8
    """

    def __init__(
        self,
        audio_dir: str,
        chapter_dir: str,
        sample_rate: int = 22050,
        delete_chapter_files: bool = True,
        exit_on_error: bool = False,
        num_retries: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_dir = Path(audio_dir)
        self.chapter_dir = Path(chapter_dir)
        self.sample_rate = sample_rate
        self.delete_chapter_files = delete_chapter_files
        self.exit_on_error = exit_on_error
        self.num_retries = num_retries

    def prepare(self):
        # Create output directory structure
        with open(self.input_manifest_file, "rt", encoding="utf-8") as fin:
            dirs = set()
            for line in fin:
                row = json.loads(line)
                audio_filepath = Path(row["utterances"][0]["audio_filepath"])
                chapter_dir = audio_filepath.parent
                dirs.add(chapter_dir)

        for dir in dirs:
            audio_dir = self.audio_dir / dir
            chapter_dir = self.chapter_dir / dir
            audio_dir.mkdir(exist_ok=True, parents=True)
            chapter_dir.mkdir(exist_ok=True, parents=True)

        return

    def process_dataset_entry(self, data_entry):
        url = data_entry["url"]
        chapter_filepath = data_entry["chapter_filepath"]
        utterances = data_entry["utterances"]

        chapter_path = self.chapter_dir / chapter_filepath
        for i in range(1, self.num_retries + 1):
            try:
                urllib.request.urlretrieve(url=url, filename=chapter_path)
                break
            except Exception as ex:
                error_msg = f"Encountered exception when downloading {url}: {ex}"
                logger.warning(error_msg)

                if i < self.num_retries:
                    logger.info(f"Retry {i} for url {url}")
                    time.sleep(10)
                    continue

                if self.exit_on_error:
                    raise RuntimeError(error_msg)

                if isinstance(ex, urllib.error.URLError):
                    error_reason = ex.reason
                else:
                    error_reason = repr(ex)

                error_data = {
                    "url": url,
                    "chapter_filepath": chapter_filepath,
                    "error_reason": error_reason,
                    "utterances": utterances,
                }
                return [DataEntry(data=error_data)]

        chapter_audio, sr = librosa.load(path=chapter_path, sr=self.sample_rate)
        chapter_duration = librosa.get_duration(y=chapter_audio, sr=sr)

        original_duration = data_entry["duration"]
        duration_diff = abs(chapter_duration - original_duration)
        if duration_diff > 0.1:
            error_msg = f"Duration mismatch for {url}: original duration={original_duration}; " \
                        f"downloaded duration={round(chapter_duration, 2)}"
            logger.warning(error_msg)

            if self.exit_on_error:
                raise RuntimeError(error_msg)

            error_data = {
                "url": url,
                "chapter_filepath": chapter_filepath,
                "error_reason": error_msg,
                "utterances": utterances,
            }
            return [DataEntry(data=error_data)]

        for utt in utterances:
            audio_filepath = utt["audio_filepath"]
            audio_path = self.audio_dir / audio_filepath
            offset = utt["offset"]
            dur = utt["duration"]
            start_sample = librosa.time_to_samples(offset, sr=sr)
            end_sample = librosa.time_to_samples(offset + dur, sr=sr)
            audio = chapter_audio[start_sample:end_sample]
            sf.write(file=audio_path, data=audio, samplerate=int(sr))

        if self.delete_chapter_files:
            chapter_path.unlink()

        return []
