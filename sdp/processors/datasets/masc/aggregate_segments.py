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
import logging
from pydub import AudioSegment
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.processors.datasets.masc.utils import save_audio_segment

class AggregateSegments(BaseParallelProcessor):
    """
    Aggregates short segments into segments with duration not longer than `max_duration`.
    The algorithm works by iterating from left to right, merging consecutive segments into the current segment until the total duration reaches `max_duration`.
    
    output_audio_dir (str): Directory where aggregated audio segments will be stored, if `save_aggregated_audio_segments` is True.
        If `save_aggregated_audio_segments` is False, this path is used to create the audio file paths in the manifest.
    input_segments_key (str): The field name that contains list of segments in the input manifest. Defaults to "segments".
    input_audio_filepath_key (str): The field name that contains paths to the audio files in the input manifest.
        Defaults to "audio_filepath".
    output_text_key (str): Field name where to store aggregated segment text. Defaults to "text".
    output_duration_key (str): Field name where aggregated segment durations will be stored. Defaults to "duration".
    output_audio_filepath_key (str): Field name where aggregated segment audio file paths will be stored.
        Defaults to "audio_filepath".
    max_duration (float): Maximum duration of aggregated segment. Default to 20.0s.
    save_aggregated_audio_segments (bool): Flag indicating whether to crop audio files according to the aggregated segments.
        Defaults to True.
    verbose (bool): Set to True to enable more detailed logging. Defaults to False.
    """
    def __init__(
        self,
        output_audio_dir: str,
        input_segments_key: str = "segments",
        input_audio_filepath_key: str = "audio_filepath",
        output_text_key: str = "text",
        output_duration_key: str = "duration",
        output_audio_filepath_key: str = "audio_filepath",
        max_duration: float = 20.0,
        save_aggregated_audio_segments: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_duration = max_duration
        self.input_audio_filepath_key = input_audio_filepath_key
        self.output_splitted_audio_filepath_key = output_audio_filepath_key
        self.save_aggregated_audio_segments = save_aggregated_audio_segments
        self.output_audio_dir = output_audio_dir
        self.input_segments_key = input_segments_key
        self.verbose = verbose
        self.output_text_key = output_text_key
        self.output_duration_key = output_duration_key

    def prepare(self):
        if self.save_aggregated_audio_segments and self.output_audio_dir:
            os.makedirs(os.path.join(self.output_audio_dir), exist_ok=True)

    def process_dataset_entry(self, data_entry: dict):
        if self.input_segments_key not in data_entry:
            if self.verbose:
                logging.info(f"No segments in the sample {data_entry[self.input_audio_filepath_key]}.")
            return []

        segments = data_entry[self.input_segments_key]
        if len(segments) == 0:
            return []
        
        audio = AudioSegment.from_wav(data_entry[self.input_audio_filepath_key])
        
        audio_basename = os.path.basename(data_entry[self.input_audio_filepath_key]).split(".")[0]
        agg_segments = []
        aggregated_segment = {**segments[0]}
        for segment in segments[1:]:
            # checking if adding another segment will cause the total duration to exceed max_duration
            if (segment["end_time"] > audio.duration_seconds or segment["start_time"] > audio.duration_seconds):
                continue
            
            start_time = min(segment["start_time"], aggregated_segment["start_time"])
            end_time = max(segment["end_time"], aggregated_segment["end_time"])
            if end_time - start_time >= self.max_duration:
                agg_segments.append(aggregated_segment)
                aggregated_segment = {**segment}
            else:
                # updating aggregated segment text with correct order of segments.
                if aggregated_segment["start_time"] < segment["start_time"]:
                    aggregated_segment["text"] += f" {segment['text']}".strip()
                else:
                    aggregated_segment["text"] = f"{segment['text']} {aggregated_segment['text']}"
                    
                aggregated_segment["start"] = start_time    # updating aggregated segment start time
                aggregated_segment["end_time"] = end_time   # updating aggregated segment end time
        else:
            # adding the last aggregated segment 
            if aggregated_segment not in agg_segments:
                agg_segments.append(aggregated_segment)
            
        valid_segments = []
        for aggregated_segment in agg_segments:
            aggregated_segment.update(data_entry)
            
            start_time = aggregated_segment.pop("start_time")
            end_time = aggregated_segment.pop("end_time")
            
            aggregated_segment[self.output_duration_key] = end_time - start_time
            aggregated_segment[self.output_splitted_audio_filepath_key] = os.path.join(self.output_audio_dir, f"{audio_basename}_{start_time}_{end_time}.wav")
            
            if self.save_aggregated_audio_segments:
                try:
                    save_audio_segment(
                        audio=audio,
                        start_time=start_time,
                        end_time=end_time,
                        output_audio_filepath=aggregated_segment[self.output_splitted_audio_filepath_key]
                    )
                    valid_segments.append(aggregated_segment)
                except IndexError as e:
                    if self.verbose:
                        logging.warning(f"Invalid segment boundaries in {audio_basename}. Skipping...")
                
        return [DataEntry(data=segment) for segment in valid_segments]
        