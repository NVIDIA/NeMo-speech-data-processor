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

"""
Class which will process the data for Fisher Spanish and create an initial manifest.
The raw Fisher Spanish data is provided in .sph files and with 2 channels (1 for each speaker).
The data needs to be converted in .wav and trimmed+segmented. This script will do the processing
required, and result in the following tree:

<ROOT_DATA_DIR>
├── fisher_spa_LDC2010S01.tgz
├── LDC2010T04.tgz
├── downloaded
│   ├── fisher_spa
│   │   ├── data
│   │   │   └── speech
│   │   └── docs
│   └── fisher_spa_tr
│       ├── data
│       │   └── transcripts
│       └── docs
└── processed
    ├── manifests
    └── wavs
        ├── original_duration
        └── trimmed_and_segmented

"""

import glob
import json
import os
from pathlib import Path
from sox import Transformer
import subprocess
from tqdm import tqdm
from typing import List, Optional

from nemo.utils import logging

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import extract_archive

AUDIO_TGZ_FILE = "fisher_spa_LDC2010S01.tgz"
TRANSCRIPT_TGZ_FILE = "LDC2010T04.tgz"


class CreateInitialManifestFisherSpanish(BaseParallelProcessor):
    """
    Class which will create an initial manifest from the initial Fisher Spanish 
    data files, which must be located in root_data_dir.

    Args:
        raw_data_dir: path to where the initial data archive files are located. This will 
            also be where the new audio files are processed and where the manifests are saved.
        path_to_sph2pipe: the path to the sph2pipe tool, which will be used to convert
            the sph audio files to wav files.
    """

    def __init__(self, raw_data_dir: str, path_to_sph2pipe: str, **kwargs):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.path_to_sph2pipe = path_to_sph2pipe

        self.extracted_path = str(Path(self.raw_data_dir) / "extracted")
        self.processed_path = str(Path(self.raw_data_dir) / "processed")

    def prepare(self):
        audio_archive_path = self.raw_data_dir / AUDIO_TGZ_FILE
        transcript_archive_path = self.raw_data_dir / TRANSCRIPT_TGZ_FILE

        if audio_archive_path.exists() and transcript_archive_path.exists():
            extract_archive(audio_archive_path, self.extracted_path)
            extract_archive(transcript_archive_path, self.extracted_path)

        elif (self.raw_data_dir / "data.tar.gz").exists():
            extract_archive(str(self.raw_data_dir / "data.tar.gz"), self.extracted_path)

        else:
            raise RuntimeError(
                "Did not find the expected raw data files. Expected to find either \n"
                f"(i) 2 files: {audio_archive_path} and {transcript_archive_path} \n"
                "or \n"
                f"(ii) 1 file: {str(self.raw_data_dir / 'data.tar.gz')}"
            )

        # convert audio files from .sph to .wav
        sph_src_dir = os.path.join(self.extracted_path, "fisher_spa/data/speech")
        wav_tgt_dir = os.path.join(self.processed_path, "wavs/original_duration")
        if not os.path.exists(wav_tgt_dir):
            os.makedirs(wav_tgt_dir)

        logging.info("Converting files from .sph to .wav")
        sph_list = glob.glob(sph_src_dir + "/*.sph")

        for sph_path in tqdm(sph_list):
            file_id = os.path.basename(sph_path).split(".sph")[0]
            wav_path = os.path.join(wav_tgt_dir, file_id + ".wav")
            cmd = [self.path_to_sph2pipe, "-f", "wav", "-p", sph_path, wav_path]
            subprocess.run(cmd)
        logging.info("Finished converting files from .sph to .wav")

    def read_manifest(self) -> List[tuple[str]]:
        transcript_src_dir = os.path.join(self.extracted_path, "fisher_spa_tr/data/transcripts/")

        logging.info(f"Attempting to read transcription files in dir {transcript_src_dir}")
        dataset_entries = []

        for transcript_file in tqdm(glob.glob(transcript_src_dir + "/*.tdf")):
            with open(transcript_file, "r") as f_in:
                f_in.readline()  # skip column headings
                f_in.readline()  # skip comments with ;;
                f_in.readline()  # skip comments with ;;
                for line_i, line in enumerate(f_in):
                    line = line.strip()
                    line = line.split("\t")
                    line = [line_i] + line
                    dataset_entries.append(tuple(line))

        return dataset_entries

    def process_dataset_entry(self, data_entry: tuple[str]):

        wav_src_dir = os.path.join(self.processed_path, "wavs/original_duration")
        wav_tgt_dir = os.path.join(self.processed_path, "wavs/trimmed_and_segmented")
        manifest_dir = os.path.join(self.processed_path, "manifests/")

        os.makedirs(wav_tgt_dir, exist_ok=True)
        os.makedirs(manifest_dir, exist_ok=True)

        (
            line_i,
            file_id,
            channel,
            start,
            end,
            speaker,
            speaker_type,
            speaker_dialect,
            transcript,
            section,
            turn,
            segment,
            *other_info,
        ) = data_entry

        file_id = file_id.split(".sph")[0]

        src_wav_file = os.path.join(wav_src_dir, f"{file_id}.wav")
        tgt_wav_file = os.path.join(
            wav_tgt_dir, f"{file_id}_line{line_i}_channel{channel}_{section}_{turn}_{segment}.wav",
        )

        if len(transcript) == 0:
            logging.info(f"Empty transcript. Skipping trying to make wav file {tgt_wav_file}")
            return []

        if float(end) - float(start) < 0.2:
            logging.info(f"start time: {start}, end time: {end}")
            logging.info(f"=> (end time) - (start time) is too small. Skipping trying to make wav file {tgt_wav_file}")
            return []

        # make trimmed wave file
        transformer = Transformer()
        transformer.trim(float(start), float(end))
        transformer.rate(samplerate=16000, quality="v")
        # pick out 1 speaker and make mono
        # Note that mapping in remix dictionary is
        # (output channel):(input channel), with indexing starting from 1
        transformer.remix({1: [int(channel) + 1]}, num_output_channels=1)
        transformer.build(src_wav_file, tgt_wav_file)

        entry = {}
        entry["audio_filepath"] = tgt_wav_file

        # get duration
        duration = subprocess.check_output("soxi -D {0}".format(entry["audio_filepath"]), shell=True)

        if float(duration) == 0:
            logging.info(f"created wave file with duration zero: {tgt_wav_file}")
            logging.info(f"=> will not add this file to manifest")
            return []

        entry["duration"] = float(duration)
        entry["text"] = transcript
        entry["channel"] = channel
        entry["start"] = start
        entry["end"] = end
        entry["speaker"] = speaker
        entry["speaker_type"] = speaker_type
        entry["speaker_dialect"] = speaker_dialect
        entry["section"] = section
        entry["turn"] = turn
        entry["segment"] = segment
        entry["other_info"] = ",".join(other_info)

        return [DataEntry(data=entry)]
