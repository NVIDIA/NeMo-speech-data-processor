# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional
from sox import Transformer

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

from sdp.utils.common import ffmpeg_convert


class FfmpegConvert(BaseParallelProcessor):
    """
    Processor for converting video or audio files to audio using FFmpeg and updating the dataset with the path to the resampled audio.
    If ``id_key`` is not None, the output file path will be ``<resampled_audio_dir>/<id_key>.wav``.
    If ``id_key`` is None, the output file path will be ``<resampled_audio_dir>/<input file name without extension>.wav``.

    .. note:: ``id_key`` can be used to create subdirectories inside ``resampled_audio_dir`` (by using forward slashes ``/``).
        e.g. if ``id_key`` takes the form ``dir_name1/dir_name2/filename``, the output file path will be

        ``<resampled_audio_dir>/dir_name1/dirname2/filename.wav``.

    Args:
        converted_audio_dir (str): The directory to store the resampled audio files.
        input_file_key (str): The field in the dataset representing the path to the input video or audio files.
        output_file_key (str): The field in the dataset representing the path to the resampled audio files with ``output_format``. If ``id_key`` is None, the output file path will be ``<resampled_audio_dir>/<input file name without extension>.wav``.
        id_key (str): (Optional) The field in the dataset representing the unique ID or identifier for each entry. If ``id_key`` is not None, the output file path will be ``<resampled_audio_dir>/<id_key>.wav``. Defaults to None.
        output_format (str): (Optional) Format of the output audio files. Defaults to `wav`.
        target_samplerate (int): (Optional) The target sampling rate for the resampled audio. Defaults to 16000.
        target_nchannels (int): (Optional) The target number of channels for the resampled audio. Defaults to 1.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        converted_audio_dir: str,
        input_file_key: str,
        output_file_key: str,
        id_key: str = None,
        output_format: str = "wav",
        base_dir: str = None,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.converted_audio_dir = converted_audio_dir
        self.input_file_key = input_file_key
        self.output_file_key = output_file_key
        self.output_format = output_format
        self.id_key = id_key
        self.base_dir = base_dir
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def prepare(self):
        assert self.output_format == "wav", "Currently only wav format is supported"
        os.makedirs(self.converted_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        input_file = data_entry[self.input_file_key]
        if self.id_key:
            key = data_entry[self.id_key]
            os.makedirs(os.path.join(self.converted_audio_dir, *key.split("/")[:-1]), exist_ok=True)
        else:
            key = os.path.splitext(input_file)[0].split("/")[-1]

        if self.base_dir:
            new_dir = os.path.dirname(os.path.relpath(input_file, self.base_dir))
            os.makedirs(os.path.join(self.converted_audio_dir, new_dir), exist_ok=True)

            key = os.path.join(new_dir, key)

        audio_file = os.path.join(self.converted_audio_dir, key) + "." + self.output_format

        if not os.path.isfile(audio_file):
            ffmpeg_convert(input_file, audio_file, self.target_samplerate, self.target_nchannels)

        data_entry[self.output_file_key] = audio_file
        return [DataEntry(data=data_entry)]


class SoxConvert(BaseParallelProcessor):
    """Processor for Sox to convert audio files to specified format.

    Args:
        output_manifest_file (str): Path to the output manifest file.
        input_audio_file_key (str): Key in the manifest file that contains the path to the input audio file.
        output_audio_file_key (str): Key in the manifest file that contains the path to the output audio file.
        converted_audio_dir (str): Path to the directory where the converted audio files will be stored.
        output_format (str): Format of the output audio file.
        rate (int): Sample rate of the output audio file.
        channels (int): Number of channels of the output audio file.
        workspace_dir (str, Optional): Path to the workspace directory. Defaults to None.
    """

    def __init__(
        self,
        converted_audio_dir: str,
        input_audio_file_key: str = "audio_filepath",
        output_audio_file_key: str = "audio_filepath",
        output_format: str = "wav",
        rate: int = 16000,
        channels: int = 1,
        workspace_dir: Optional[str] = None,
        **kwargs,
    ):
        # Extract workspace_dir from kwargs to avoid passing it to BaseProcessor
        if "workspace_dir" in kwargs:
            workspace_dir = kwargs.pop("workspace_dir")
            
        super().__init__(**kwargs)
        self.input_audio_file_key = input_audio_file_key
        self.output_audio_file_key = output_audio_file_key
        self.converted_audio_dir = converted_audio_dir
        self.output_format = output_format
        self.workspace_dir = workspace_dir

        # Store the new parameters for later use:
        self.rate = rate
        self.channels = channels

    def prepare(self):
        # Debug print for workspace_dir
        logger.info(f"SoxConvert workspace_dir: {self.workspace_dir}")
        os.makedirs(self.converted_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        audio_path = data_entry[self.input_audio_file_key]
        
        # If workspace_dir is provided, join it with audio_path to get absolute path
        if self.workspace_dir is not None:
            full_audio_path = os.path.join(self.workspace_dir, audio_path)
        else:
            full_audio_path = audio_path
            
        # Debug print first file path
        if not hasattr(self, '_debug_printed'):
            logger.info(f"First audio_path from manifest: {audio_path}")
            logger.info(f"First full_audio_path: {full_audio_path}")
            logger.info(f"Path exists: {os.path.exists(full_audio_path)}")
            self._debug_printed = True

        key = os.path.splitext(audio_path)[0].split("/")[-1]
        converted_file = os.path.join(self.converted_audio_dir, key) + f".{self.output_format}"

        if not os.path.isfile(converted_file):
            transformer = Transformer()

            transformer.rate(self.rate)
            transformer.channels(self.channels)

            transformer.build(full_audio_path, converted_file)

        data_entry[self.output_audio_file_key] = converted_file
        return [DataEntry(data=data_entry)]