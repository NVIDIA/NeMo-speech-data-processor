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

import os
import pathlib
from pathlib import Path

from pydub import AudioSegment

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class CreateInitialManifestBabel(BaseParallelProcessor):
    """Processor to create initial manifest for the Babel dataset.

    Dataset is available for 25 underserved languages on https://catalog.ldc.upenn.edu

    Segments the raw audio based on transcriptions files
    (each segment contains an utterance from the transcription file for which start and end timestamps are procided)
    and creates manifest for the resampled data.

    .. note::
        The dataset should be downloaded manually from LDC.

    Args:
        raw_data_dir (str): the directory where the downloaded data is saved.
        data_type (str): "conversational" or "scripted".
        data_split (str): "training", "untranscribed-training", "sub-train", "dev" or "eval".
        resampled_audio_dir (str): the directory where the resampled audio
            files will be stored.
        audio_format (str): format in which new audio files will be stored.
        target_samplerate (int): sample rate (Hz) to use for resampling.
            Defaults to 16000.
        target_nchannels (int): number of channels to create during resampling process.
            Defaults to 1.

    Returns:
        This processor generates an initial manifest file with the following fields::

            {
                "outputFn": <initial audio filename>,
                "sessID": <session ID of the recording>,
                "date": <date of the recording>,
                "time": <time of the recording>,
                "spkrCode": <speaker ID>,
                "lineType": <type of the line (inline or outline)>,
                "dialect": <dialect of the speaker>,
                "gen": <gender of the speaker>,
                "envType": <environment (i.e. home, office etc.)>,
                "age": <age of the speaker>,
                "network": <name of the telecommunications network>,
                "phoneModel": <model of the phone>,
                "sampleCount": <count of the sample>,
                "sampleRate": <original sample rate of the recording>,
                "audio_filepath": <path to the processed audio file>,
                "duration": <duration of the audio in seconds>,
            }
    """

    def __init__(
        self,
        raw_data_dir: str,
        data_type: str,
        data_split: str,
        resampled_audio_dir: str,
        audio_format: str = 'flac',
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.data_type = data_type
        self.resampled_audio_dir = resampled_audio_dir
        self.data_split = data_split
        self.audio_format = audio_format
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

        self.audios_dir = Path(self.raw_data_dir, self.data_type, self.data_split, 'audio')
        self.transcriptions_dir = Path(self.raw_data_dir, self.data_type, self.data_split, 'transcription')
        self.demographics_file = Path(self.raw_data_dir, self.data_type, 'reference_materials', 'demographics.tsv')
        self.un_demographics_file = Path(
            self.raw_data_dir, self.data_type, 'reference_materials', 'demographics.untranscribed-training.tsv'
        )

        if not os.path.exists(self.resampled_audio_dir):
            os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def prepare(self):
        self.demographics = {}

        with open(self.demographics_file, "rt", encoding="utf8") as fin:
            header = fin.readline()
            titles = [t.strip() for t in header.split('\t')]
            for line in fin:
                data_entry = dict(zip(titles, line.strip('\n').split('\t')))
                self.demographics[data_entry["outputFn"]] = data_entry

        if self.un_demographics_file.exists():
            with open(self.un_demographics_file, "rt", encoding="utf8") as fin:
                header = fin.readline()
                titles = [t.strip() for t in header.split('\t')]
                for line in fin:
                    data_entry = dict(zip(titles, line.strip('\n').split('\t')))
                    self.demographics[data_entry["outputFn"]] = data_entry

    def read_manifest(self):
        return self.audios_dir.glob("*.sph")

    def process_dataset_entry(self, data_entry: pathlib.PosixPath):
        transcription_path = Path(self.transcriptions_dir, data_entry.stem).with_suffix('.txt')

        tgt_audio_filepath = (
            Path(self.resampled_audio_dir, data_entry.stem).with_suffix(f".{self.audio_format}").as_posix()
        )

        try:
            audio = AudioSegment.from_file(data_entry)

            if not transcription_path.exists():
                if audio.frame_rate != self.target_samplerate:
                    audio = audio.set_frame_rate(self.target_samplerate)
                audio.export(tgt_audio_filepath, format=self.audio_format)

                modified_entry = self.demographics[data_entry.name].copy()
                modified_entry['audio_filepath'] = tgt_audio_filepath
                modified_entry['duration'] = round(audio.duration_seconds, 2)
                return [DataEntry(data=modified_entry)]

            if audio.frame_rate != self.target_samplerate:
                audio = audio.set_frame_rate(self.target_samplerate)

            if audio.channels != self.target_nchannels:
                audio = audio.set_channels(self.target_nchannels)

            with open(transcription_path, 'rt') as f:
                data = f.readlines()
                timestamps = data[::2]
                texts = data[1::2]

            data_entries = []

            for idx in range(len(timestamps) - 1):
                text = texts[idx].strip('\n')
                if text == "<no-speech>":
                    continue
                new_audio_filepath = tgt_audio_filepath.replace(f'.{self.audio_format}', f'_{idx}.{self.audio_format}')

                start = float(timestamps[idx].strip('[]\n'))
                end = float(timestamps[idx + 1].strip('[]\n'))

                audio_segment = audio[start * 1000 : end * 1000]

                audio_segment.export(new_audio_filepath, format=self.audio_format)

                modified_entry = self.demographics[data_entry.name].copy()
                modified_entry['audio_filepath'] = new_audio_filepath
                modified_entry['text'] = text
                modified_entry['duration'] = round(end - start, 2)

                data_entries.append(DataEntry(data=modified_entry))

        except Exception as e:
            logger.warning(str(e) + " file: " + transcription_path)

        return data_entries