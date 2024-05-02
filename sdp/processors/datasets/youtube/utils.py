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
import re
from dataclasses import dataclass

import pysrt
import re
from pydub import AudioSegment

from sdp.processors.base_processor import DataEntry


@dataclass
class RawSegment:
    segment_id: int = None
    start_time: float = None
    end_time: float = None
    duration: str = None
    duration_match: bool = None
    orig_text: str = None
    audio_lang: str = None
    text_lang: str = None
    source_audio: str = None

    def to_dataentry(self):
        return DataEntry(data=self.__dict__)


class AggregatedSegment(RawSegment):
    def __init__(
        self,
        segment: dict,
        segment_id: int,
        sample_id: str,
        output_audio_dir: str,
        audio_lang: str,
        text_lang: str,
        source_audio: str,
    ):
        super().__init__(**segment.__dict__)
        self.segment_id = f"{sample_id}_{str(segment_id).zfill(4)}"
        self.audio_lang = audio_lang
        self.text_lang = text_lang
        self.source_audio = source_audio
        self.audio_filepath = (
            os.path.join(output_audio_dir, f'{self.segment_id}.wav') if output_audio_dir is not None else None
        )

    def aggregate(self, segment):
        self.end_time = segment.end_time
        self.duration = self.end_time - self.start_time
        self.orig_text = re.sub("\s+", " ", f"{self.orig_text} {segment.orig_text}".strip())


@dataclass
class Sample:
    sample_id: str = None
    srt_filepath: str = None
    orig_audio_filepath: str = None
    audio_filepath: str = None
    segments: list[RawSegment | AggregatedSegment] = None

    def to_dataentry(self):
        data = self.__dict__
        data['segments'] = (
            [segment.data.__dict__ for segment in data['segments']] if data['segments'] is not None else []
        )
        return DataEntry(data=data)


def get_audio_segment(audio, start_time: float, end_time: float, output_audio_filepath: str = None):
    start_time = start_time * 1000
    end_time = end_time * 1000
    audio_segment = audio[start_time:end_time]

    if output_audio_filepath:
        audio_segment.export(output_audio_filepath, format="wav")
    return audio_segment


def get_audio_segment_duration(audio, start_time, end_time):
    audio_segment = get_audio_segment(audio, start_time, end_time)
    return audio_segment.duration_seconds


def parse_srt(srt_filepath, verify_duration: bool = True, wav_filepath: str = None):
    subs = pysrt.open(srt_filepath)
    srt_segments = []

    if verify_duration and wav_filepath:
        audio = AudioSegment.from_wav(wav_filepath)
    else:
        audio = None

    epsilon = 1e-2

    for sub in subs:
        segment = RawSegment(
            segment_id=sub.index,
            start_time=sub.start.ordinal / 1000,
            end_time=sub.end.ordinal / 1000,
            orig_text=sub.text_without_tags,
        )

        duration_by_timestemps = segment.end_time - segment.start_time

        if audio:
            segment.duration = get_audio_segment_duration(audio, segment.start_time, segment.end_time)
            segment.duration_match = abs(segment.duration - duration_by_timestemps) < epsilon
        else:
            segment.duration = duration_by_timestemps

        srt_segments.append(segment)

    return srt_segments


@dataclass
class Word:
    sample_id: str = None
    text: str = None
    start_time: float = None
    duration: float = None

    def __init__(self, ctm_str):
        ctm_args = ctm_str.split()
        self.sample_id = ctm_args[0]
        self.start_time = float(ctm_args[2])
        self.duration = float(ctm_args[3])
        self.text = ctm_args[4]

@dataclass
class Sentence:
    words: list[Word] = None
    sample_id: str = None
    text: str = None
    start_time: float = None
    duration: float = None

    def add_word(self, word):
        if self.words is None:
            self.words = []

        self.words.append(word)

    def process(self):
        self.sample_id = self.words[0].sample_id
        self.text = ' '.join([word.text for word in self.words])
        self.text = self.text[0].upper() + self.text[1 : ]
        self.start_time = self.words[0].start_time
        self.duration = round(self.words[-1].start_time + self.words[-1].duration - self.start_time, 2)
    
    def to_dict(self):
        sample = self.__dict__
        del sample['words']
        return sample

    def from_dict(sentence_dict: dict):
        sentence_obj = Sentence() 
        sentence_obj.__dict__.update(sentence_dict)
        return sentence_obj

def read_ctm(ctm_filepath):
    with open(ctm_filepath, 'r') as ctm:
        lines = ctm.readlines()
        words = [Word(line) for line in lines]
        return words

@dataclass
class Sample:
    sample_id: str = None
    text: str = None
    start_time: float = None
    duration: float = None

    def add_segment(self, segment):
        if self.sample_id is None:
            self.text = ""
            self.sample_id = segment.sample_id
            self.start_time = segment.start_time
            
        self.text = re.sub("\s+", " ", self.text + " " + segment.text).strip()
        self.duration = round(segment.start_time + segment.duration - self.start_time, 2)
    
    def to_dict(self):
        sample = self.__dict__
        return sample
    
    def from_dict(sample_dict: dict):
        sample_obj = Sample() 
        sample_obj.__dict__.update(sample_dict)
        return sample_obj