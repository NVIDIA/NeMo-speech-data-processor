# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import soundfile as sf

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class GetAudioDuration(BaseParallelProcessor):
    """
    Processor to count audio duration using audio file path from input_field

    Args:
        audio_filepath_field (str): where to get path to wav file.
        duration_field (str): where to put to audio duration.
    Returns:
        All the same fields as in the input manifest plus output_field
    """

    def __init__(
        self,
        audio_filepath_field: str,
        duration_field: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_filepath_field = audio_filepath_field
        self.duration_field = duration_field

    def process_dataset_entry(self, data_entry):
        audio_filepath = data_entry[self.audio_filepath_field]
        data, samplerate = sf.read(audio_filepath)
        data_entry[self.duration_field] = data.shape[0] / samplerate
        return [DataEntry(data=data_entry)]
