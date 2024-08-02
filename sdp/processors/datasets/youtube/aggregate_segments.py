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
from sdp.processors.datasets.youtube.utils import get_audio_segment

class AggregateSegments(BaseParallelProcessor):
    def __init__(
        self,
        input_segments_key: str = "segments",
        input_audio_filepath_key: str = "audio_filepath",
        output_text_key: str = "text",
        output_duration_key: str = "duration",
        output_audio_filepath_key: str = "audio_filepath",
        output_audio_dir: str = None,
        max_duration: float = 20.0,
        crop_audio_segments: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_duration = max_duration
        self.input_audio_filepath_key = input_audio_filepath_key
        self.output_splitted_audio_filepath_key = output_audio_filepath_key
        self.crop_audio_segments = crop_audio_segments
        self.output_audio_dir = output_audio_dir
        self.input_segments_key = input_segments_key
        self.verbose = verbose
        self.output_text_key = output_text_key
        self.output_duration_key = output_duration_key

    def prepare(self):
        if self.crop_audio_segments and self.output_audio_dir:
            os.makedirs(os.path.join(self.output_audio_dir), exist_ok=True)

    def process_dataset_entry(self, data_entry: dict):
        if "segments" not in data_entry:
            if self.verbose:
                logging.info(f"No segements in a sample {data_entry[self.input_audio_filepath_key]}.")
            return []

        segments = data_entry[self.input_segments_key]
        if len(segments) == 0:
            return []
        
        audio_basename = os.path.basename(data_entry[self.input_audio_filepath_key]).split(".")[0]
        agg_segments = []
        aggregated_segment = {**segments[0]}
        for segment in segments[1:]:
            # checking if after adding segement it's duration will exceed `max_duration`
            if segment["end_time"] - aggregated_segment["start_time"] >= self.max_duration:
                agg_segments.append(aggregated_segment)
                aggregated_segment = {**segment}
            else:
                aggregated_segment["end_time"] = segment["end_time"]           # updating aggregated segment end time
                aggregated_segment["text"] += f" {segment['text']}".strip()    # updating aggregated segment text
        else:
            # adding the last aggregated segment 
            if aggregated_segment not in agg_segments:
                agg_segments.append(aggregated_segment)
            
        for aggregated_segment in agg_segments:
            aggregated_segment.update(data_entry)
            
            start_time = aggregated_segment.pop("start_time")
            end_time = aggregated_segment.pop("end_time")
            
            aggregated_segment[self.output_duration_key] = end_time - start_time
            aggregated_segment[self.output_splitted_audio_filepath_key] = os.path.join(self.output_audio_dir, f"{audio_basename}_{start_time}_{end_time}.wav")
            
            if self.crop_audio_segments:
                audio = AudioSegment.from_wav(data_entry[self.input_audio_filepath_key])
                get_audio_segment(
                        audio=audio,
                        start_time=start_time,
                        end_time=end_time,
                        output_audio_filepath=aggregated_segment[self.output_splitted_audio_filepath_key]
                )
        return [DataEntry(data=segment) for segment in agg_segments]
        