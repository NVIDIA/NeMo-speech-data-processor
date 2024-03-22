# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import json
import random

import torch
import torchaudio
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    SeamlessM4TModel,
    SeamlessM4Tv2ForSpeechToText,
    SeamlessM4Tv2Model,
)

from sdp.processors.base_processor import BaseProcessor

# class ASRSeamless(BaseProcessor):
#     def __init__(
#         self,
#         device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
#         input_field: str = "audio_filepath",
#         output_field: str = "model_transcribed_text",
#         limit: int = 100,
#         **kwargs,
#     ):
#         self.device = device
#         self.input_field = input_field
#         self.output_field = output_field
#         self.limit = limit
#         super().__init__(**kwargs)

#     def process(self):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         print("Using device:", device)

#         # "facebook/hf-seamless-m4t-large"
#         # "facebook/seamless-m4t-v2-large"
#         processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
#         model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

#         entries = []

#         with open(self.input_manifest_file, 'r') as f:
#             lines = f.readlines()

#             total_lines = len(lines)
#             files_to_process_count = int((self.limit / 100) * total_lines)
#             selected_indices = random.sample(range(total_lines), files_to_process_count)

#             for idx in tqdm(selected_indices, desc="Processing Audio Files"):
#                 line = lines[idx]
#                 entry = json.loads(line)
#                 # print(entry)
#                 audio_file_path = entry[self.input_field]

#                 waveform, orig_sampling_rate = torchaudio.load(audio_file_path)
#                 waveform = waveform.to(device)
#                 if orig_sampling_rate != 16000:
#                     # print("Not in the right sample rate. Resampling to 16kHz.")
#                     resampler = torchaudio.transforms.Resample(orig_freq=orig_sampling_rate, new_freq=16000).to(device)
#                     waveform = resampler(waveform)

#                 audio_inputs = processor(
#                     audios=waveform.squeeze().cpu().numpy(), src_lang="hye", sampling_rate=16000, return_tensors="pt"
#                 ).to(device)

#                 outputs = model.generate(**audio_inputs, tgt_lang="hye")

#                 transcribed_text = processor.batch_decode(outputs, skip_special_tokens=True)

#                 entry[self.output_field] = str(transcribed_text)
#                 entries.append(entry)
#             with open(self.output_manifest_file, 'w', encoding='utf-8') as f_out:
#                 for entry in entries:
#                     json.dump(entry, f_out, ensure_ascii=False)
#                     f_out.write("\n")


class ASRSeamless(BaseProcessor):
    def __init__(
        self,
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
        input_field: str = "audio_filepath",
        output_field: str = "model_transcribed_text",
        limit: int = 100,
        **kwargs,
    ):
        self.device = device
        self.input_field = input_field
        self.output_field = output_field
        self.limit = limit
        super().__init__(**kwargs)
        self.failed_files = []  # Initialize array to store failed file names

    def process(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

        entries = []

        with open(self.input_manifest_file, 'r') as f:
            lines = f.readlines()

            total_lines = len(lines)
            files_to_process_count = int((self.limit / 100) * total_lines)
            selected_indices = random.sample(range(total_lines), files_to_process_count)

            for idx in tqdm(selected_indices, desc="Processing Audio Files"):
                line = lines[idx]
                entry = json.loads(line)
                audio_file_path = entry[self.input_field]

                try:
                    waveform, orig_sampling_rate = torchaudio.load(audio_file_path)
                except Exception as e:
                    print(f"Failed to load {audio_file_path}: {e}")
                    self.failed_files.append(audio_file_path)
                    continue

                waveform = waveform.to(device)
                if orig_sampling_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=orig_sampling_rate, new_freq=16000).to(device)
                    waveform = resampler(waveform)

                audio_inputs = processor(
                    audios=waveform.squeeze().cpu().numpy(), src_lang="hye", sampling_rate=16000, return_tensors="pt"
                ).to(device)

                outputs = model.generate(**audio_inputs, tgt_lang="hye")

                transcribed_text = processor.batch_decode(outputs, skip_special_tokens=True)

                entry[self.output_field] = str(transcribed_text)
                entries.append(entry)

            with open(self.output_manifest_file, 'w', encoding='utf-8') as f_out:
                for entry in entries:
                    json.dump(entry, f_out, ensure_ascii=False)
                    f_out.write("\n")

        if self.failed_files:
            print(f"Failed to process the following files: {self.failed_files}")
