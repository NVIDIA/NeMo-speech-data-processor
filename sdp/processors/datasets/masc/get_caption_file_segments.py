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
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.processors.datasets.masc.utils import parse_captions

class GetCaptionFileSegments(BaseParallelProcessor):
    """
    This class extracts subtitle information from .vtt (WebVTT) files.
    Each segment represents a single subtitle line.

    Args:
        input_caption_file_key (str): The field name in the input manifest containing path to the caption file.
        output_segments_key (str): The field name to store segment information. Defaults to "segments".
        verbose (bool): Set true for outputing logging information.
        
    Returns:
        This processor adds an output_segments field to the input manifest with a list of segments.
        Each segment has a structure:
            {
                "segment_id":   <index of subtitle line>,
                "start_time":   <segment start time>,
                "end_time":     <segment end time>
                "text":         <segment text>
            }
    """
    def __init__(
        self,
        input_caption_file_key: str,
        output_segments_key: str = "segments",
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.caption_file_key = input_caption_file_key
        self.output_segments_key = output_segments_key
        self.verbose = verbose

    def process_dataset_entry(self, data_entry):
        caption_file = data_entry[self.caption_file_key]
        
        if not os.path.exists(caption_file):
            if self.verbose:
                logging.info(f"File {caption_file} does not exist.")
            return []

        data_entry[self.output_segments_key] = parse_captions(caption_file)

        return [DataEntry(data=data_entry)]
