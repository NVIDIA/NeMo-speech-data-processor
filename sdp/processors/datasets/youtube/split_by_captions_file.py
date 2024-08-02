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
from sdp.processors.datasets.youtube.utils import parse_captions

class GetCaptionFileSegments(BaseParallelProcessor):
    """
    This class parses segment information from caption files.
    It supports caption files in .vtt (WebVTT) and .srt (SubRip Subtitle) formats.

    Args:
        input_audio_key (str): 
        splited_audio_dir (str): The directory to store the split audio files.
        source_audio_key (str): The field in the dataset containing the path to the source audio files.
        target_audio_key (str): The field to store the paths of the split audio files.
        duration_key (str): The field to store the duration of each split audio segment.
        text_key (str): The field to store the transcriptions corresponding to each split audio segment.
        caption_file_key (str): The field in the dataset containing the path to the VTT (WebVTT) files for segmentation.
        additional_fields (List[str], optional): List of additional fields to copy from the original data entry to the split entries.
            Defaults to an empty list.
        duration_threshold (float, optional): The duration threshold in seconds for each split audio segment. Defaults to 10.0.
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
            if not self.verbose:
                logging.info(f"File {caption_file} does not exist.")
            return []

        data_entry[self.output_segments_key] = parse_captions(caption_file)

        return [DataEntry(data=data_entry)]
