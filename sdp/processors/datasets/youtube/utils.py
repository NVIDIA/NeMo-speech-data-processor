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
from typing import Optional

from sdp.processors.base_processor import DataEntry

def get_audio_segment(audio, start_time: float, end_time: float, output_audio_filepath: Optional[str]):
    """
    Extracts a segment from audio.
    
    Args:
        audio: input audio
        start_time (float): segment start time in seconds.
        end_time (float): segment end time in seconds.
        audio_filepath (Optional[str]): filepath to store the segment.
        
    Returns:
        audio_segment: audio segment
    
    IndexError: Raised if segment boundaries are out of range.
    """
    start_time = start_time * 1000
    end_time = end_time * 1000
    
    if start_time >= len(audio) or end_time >= len(audio):
        raise IndexError("Segment boundaries are out of range.")
    
    audio_segment = audio[start_time:end_time]
    if output_audio_filepath:
        audio_segment.export(output_audio_filepath, format="wav")
    
    return audio_segment


def parse_captions(captions_filepath: str):
    """
    Creates a list of segments from .vtt or .srt captions file.
    Each segment contains segment_id, start_time, end_time and text.
    
    Args:
        captions_filepath (str): path to srt file.
    """
    subs = pysrt.open(captions_filepath)
    
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