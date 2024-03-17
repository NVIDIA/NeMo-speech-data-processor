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

import pysrt
from pydub import AudioSegment
from dataclasses import dataclass
import re
import os 
from sdp.processors.base_processor import DataEntry


@dataclass
class RawSegment:
    segment_id: int = None
    start_time: float = None
    end_time: float = None
    duration: str = None
    duration_match: bool = None
    orig_text: str = None

    def to_dataentry(self):
        return DataEntry(data = self.__dict__)


class AggregatedSegment(RawSegment):
    def __init__(self, segment: dict, segment_id: int, sample_id: str, output_audio_dir: str):
        super().__init__(**segment.__dict__)
        self.segment_id = f"{sample_id}_{str(segment_id).zfill(4)}"
        self.audio_filepath = os.path.join(output_audio_dir, f'{self.segment_id}.wav') if output_audio_dir is not None else None
    
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
         data['segments'] = [segment.data.__dict__ for segment in  data['segments']] if data['segments'] is not None else []
         return DataEntry(data = data)
    

def get_audio_segment(audio, start_time: float, end_time: float, output_audio_filepath: str = None):
    start_time = start_time * 1000
    end_time = end_time * 1000
    audio_segment = audio[start_time : end_time]
    
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
        segment = RawSegment(segment_id = sub.index,
                             start_time = sub.start.ordinal / 1000,
                             end_time = sub.end.ordinal / 1000,
                             orig_text = sub.text_without_tags)
        
        duration_by_timestemps = segment.end_time - segment.start_time

        if audio:
            segment.duration = get_audio_segment_duration(audio, segment.start_time, segment.end_time)
            segment.duration_match = abs(segment.duration - duration_by_timestemps) < epsilon   
        else: 
            segment.duration = duration_by_timestemps

        srt_segments.append(segment)
    
    return srt_segments