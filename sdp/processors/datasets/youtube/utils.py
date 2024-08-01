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

from dataclasses import dataclass

import pysrt
import os
import re

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
            os.path.join(output_audio_dir, f"{self.segment_id}.wav")
            if output_audio_dir is not None
            else None
        )

    def aggregate(self, segment):
        self.end_time = segment.end_time
        self.duration = self.end_time - self.start_time
        self.orig_text = re.sub(
            "\s+", " ", f"{self.orig_text} {segment.orig_text}".strip()
        )


def get_audio_segment(audio, start_time: float, end_time: float, output_audio_filepath: str = None):
    start_time = start_time * 1000
    end_time = end_time * 1000
    audio_segment = audio[start_time:end_time]

    if output_audio_filepath:
        audio_segment.export(output_audio_filepath, format="wav")
    return audio_segment


def parse_captions(srt_filepath: str):
    """
    Creates a list of segments from .vtt or .srt captions file.
    Each segment contains segment_id, start_time, end_time and text.
    
    Args:
        srt_filepath (str): path to srt file.
    """
    subs = pysrt.open(srt_filepath)
    
    srt_segments = []
    for sub_index, sub in enumerate(subs):              # for each entry in captions file 
        index = sub_index if sub_index else sub_index   # if captions file did not contain segments indices
        segment = {
            "segment_id": index,
            "start_time": sub.start.ordinal / 1000,
            "end_time": sub.end.ordinal / 1000,
            "text": sub.text_without_tags
        }
        srt_segments.append(segment)

    return srt_segments