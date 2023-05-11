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
    """
    Extracts raw MCV data for the specified language and creates an initial manifest
    using the transcripts provided in the raw data.

    Args:
        extract_archive_dir: directory where the extracted data will be saved
        resampled_audio_dir: directory where the resampled audio will be saved
        data_split: the data_split to create
        language_id: the ID of the language of the data
        already_extracted: bool (default False) - if True (and raw_data_override_archive is False),
            we will not try to extract the raw data.
        target_samplerate: sample rate (Hz) to use for resampling (default: 16000)
        target_nchannels: number of channels to create during resampling process (default: 1)
        archive_filepath: the path to the raw data archive
        raw_data_override_archive: (default None) - if specified, the processor will extract the 
            archive at this filepath, and use that as the data source for the remained of the processing
            (this is helpful for e.g. running tests on a subset of the data)
    
    """

    def __init__(
        self,
        extract_archive_dir: str,
        resampled_audio_dir: str,
        data_split: str,
        language_id: str,
        already_extracted: bool = False,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        archive_filepath: Optional[str] = None,
        raw_data_override_archive: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.extract_archive_dir = extract_archive_dir
        self.resampled_audio_dir = resampled_audio_dir
        self.data_split = data_split
        self.language_id = language_id
        self.already_extracted = already_extracted
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

        self.archive_filepath = archive_filepath
        self.raw_data_override_archive = raw_data_override_archive

    def prepare(self):
        """Extracting data (unless already done).

        If use_test_data is True, then will not process data, instead will
        copy the included test data (mainly useful for quick development or
        CI pipeline).
        """
        if self.raw_data_override_archive:
            data_folder = extract_archive(str(self.raw_data_override_archive), str(self.extract_archive_dir))
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
