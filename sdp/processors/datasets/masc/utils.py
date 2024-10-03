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

import webvtt # pip install webvtt-py
from typing import Optional
from sdp.processors.datasets.commoncrawl.harv_utils import parse_hours
from datetime import datetime

def save_audio_segment(audio, start_time: float, end_time: float, output_audio_filepath: Optional[str]):
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
    Creates a list of segments from .vtt caption files.
    Each segment has a structure:
    {
        "segment_id": int,       # Unique identifier for the segment
        "start_time": float,     # Start time of the segment (in seconds)
        "end_time": float,       # End time of the segment (in seconds)
        "text": str              # Text content of the segment
    }
    
    Args:
        captions_filepath (str): path to .vtt file.
    """
    srt_segments = []
    initial_timestamp = datetime.strptime('00:00:00.000', '%H:%M:%S.%f')
    for index, caption in enumerate(webvtt.read(captions_filepath)):
        text = ' '.join([text.strip() for text in caption.text.split('\n')])
        start_time = parse_hours(caption.start) - initial_timestamp
        end_time = parse_hours(caption.end) - initial_timestamp
        
        segment = {
            "segment_id": index,
            "start_time": start_time.total_seconds(),
            "end_time": end_time.total_seconds(),
            "text": text
        }
        srt_segments.append(segment)
        
    return srt_segments