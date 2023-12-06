import glob
import os
import urllib.request
from pathlib import Path
import typing as tp
import fnmatch
import json

import pandas as pd
from sox import Transformer

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry, BaseProcessor
from sdp.utils.common import download_file, extract_archive

class Flac2Wav(BaseParallelProcessor):
    """
    Processor for converting flac files to wav
    """
    def __init__(
        self,
        resampled_audio_dir: str,
        input_field: str,
        output_field: str,
        # key_field: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        # self.key_field = key_field
        self.resampled_audio_dir = resampled_audio_dir

    def prepare(self):
        os.makedirs(os.path.split(self.output_manifest_file)[0], exist_ok=True)
        os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        flac_file = data_entry[self.input_field]
        # if self.key_field:
        #     key = data_entry[self.key_field]
        #     os.makedirs(os.path.join(self.resampled_audio_dir, key.split("/")[0]), exist_ok=True)
        # else:
        key = os.path.splitext(flac_file)[0].split("/")[-1].split(".")[0]
        key = "_".join(key.split("-"))
        wav_file = os.path.join(self.resampled_audio_dir, key) + ".wav"

        if not os.path.isfile(wav_file):
            # ffmpeg_convert(flac_file, wav_file, self.target_samplerate, self.target_nchannels)
            Transformer().build(flac_file, wav_file)


        data_entry[self.output_field] = wav_file
        # if self.key_field:
        #     data_entry[self.key_field] = key
        return [DataEntry(data=data_entry)]

