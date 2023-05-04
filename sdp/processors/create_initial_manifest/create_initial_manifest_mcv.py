# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#

# Copyright (c) 2020, SeanNaren.  All rights reserved.
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
#

# To convert mp3 files to wav using sox, you must have installed sox with mp3 support
# For example sudo apt-get install libsox-fmt-mp3
import csv
import os
from pathlib import Path
from typing import Tuple, Optional

import sox
from sox import Transformer
from nemo.utils import logging
from tqdm.contrib.concurrent import process_map

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import extract_archive


class CreateInitialManifestMCV(BaseParallelProcessor):
    def __init__(
        self,
        extract_archive_dir: str,
        resampled_audio_dir: str,
        data_split: str,
        language_id: str,
        archive_filepath: Optional[str] = None,
        use_test_data: bool = False,
        relpath_from_test_data_root: Optional[str] = None,
        already_extracted: bool = False,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.archive_filepath = archive_filepath
        self.resampled_audio_dir = resampled_audio_dir
        self.extract_archive_dir = extract_archive_dir
        self.data_split = data_split
        self.language_id = language_id
        self.already_extracted = already_extracted
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels
        self.use_test_data = use_test_data
        self.relpath_from_test_data_root = relpath_from_test_data_root

    def prepare(self):
        """Extracting data (unless already done).

        If use_test_data is True, then will not process data, instead will
        copy the included test data (mainly useful for quick development or
        CI pipeline).
        """
        if self.use_test_data:
            try:
                __TEST_DATA_ROOT = os.environ["TEST_DATA_ROOT"]
                logging.info(f"Found 'TEST_DATA_ROOT' environment variable:{repr(__TEST_DATA_ROOT)}")
            except KeyError:
                raise KeyError(
                    f"Tried to look for os.environ['TEST_DATA_ROOT'] but it was not set."
                    f" Please set 'TEST_DATA_ROOT' as an environment variable and try again."
                )

            if not self.relpath_from_test_data_root:
                raise ValueError(f"relpath_from_test_data_root needs to be specified")

            self.test_data_path = str(Path(__TEST_DATA_ROOT) / self.relpath_from_test_data_root / "data.tar.gz")

            if not os.path.exists(self.test_data_path):
                raise ValueError(
                    f"No such file {self.test_data_path}. Are you sure you specified the "
                    f" 'TEST_DATA_ROOT' environment variable correctly?"
                )
            data_folder = extract_archive(str(self.test_data_path), str(self.extract_archive_dir))
            self.transcription_file = Path(data_folder)
        else:
            if not self.already_extracted:
                data_folder = extract_archive(self.archive_filepath, self.extract_archive_dir)
                self.transcription_file = Path(data_folder)
            else:
                self.transcription_file = Path(self.extract_archive_dir) / self.language_id
        self.audio_path_prefix = str(self.transcription_file / "clips")
        self.transcription_file = str(self.transcription_file / (self.data_split + ".tsv"))
        os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def read_manifest(self):
        if self.transcription_file is None:
            raise RuntimeError("self.process has to be called before processing the data.")

        with open(self.transcription_file, "rt", encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter="\t")
            next(reader, None)  # skip the headers
            dataset_entries = [(row["path"], row["sentence"]) for row in reader]
        return dataset_entries

    def process_dataset_entry(self, data_entry: Tuple[str, str]):
        file_path, text = data_entry
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        transcript_text = text.strip()

        audio_path = os.path.join(self.audio_path_prefix, file_path)
        output_wav_path = os.path.join(self.resampled_audio_dir, file_name + ".wav")

        if not os.path.exists(output_wav_path):
            tfm = Transformer()
            tfm.rate(samplerate=self.target_samplerate)
            tfm.channels(n_channels=self.target_nchannels)
            tfm.build(input_filepath=audio_path, output_filepath=output_wav_path)

        data = {
            "audio_filepath": output_wav_path,
            "duration": float(sox.file_info.duration(output_wav_path)),
            "text": transcript_text,
        }

        return [DataEntry(data=data)]
