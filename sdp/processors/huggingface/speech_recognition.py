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
import json
from pathlib import Path

from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest


import torch
import torchaudio
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    SeamlessM4Tv2ForSpeechToText,
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

                entry[self.output_field] = str(transcribed_text[0]).strip()
                entries.append(entry)

            with open(self.output_manifest_file, 'w', encoding='utf-8') as f_out:
                for entry in entries:
                    json.dump(entry, f_out, ensure_ascii=False)
                    f_out.write("\n")

        if self.failed_files:
            print(f"Failed to process the following files: {self.failed_files}")


class ASRTransformers(BaseProcessor):
    """
    Processor to transcribe using ASR Transformers model from HuggingFace.

    Args:
        pretrained_model (str): name of pretrained model on HuggingFace.
        output_text_key (str): Key to save transcription result.
        input_audio_key (str): Key to read audio file. Defaults to "audio_filepath".
        input_duration_key (str): Audio duration key. Defaults to "duration".
        device (str): Inference device.
        batch_size (int): Inference batch size. Defaults to 1.
        torch_dtype (str): Tensor data type. Default to "float32"
    """

    def __init__(
        self,
        pretrained_model: str,
        output_text_key: str,
        input_audio_key: str = "audio_filepath",
        input_duration_key: str = "duration",
        device: str = None,
        batch_size: int = 1,
        torch_dtype: str = "float32",
        generate_task: str = "transcribe",
        generate_language: str = "english",
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except:
            raise ImportError("Need to install transformers: pip install accelerate transformers")

        logger.warning("This is an example processor, for demonstration only. Do not use it for production purposes.")
        self.pretrained_model = pretrained_model
        self.input_audio_key = input_audio_key
        self.output_text_key = output_text_key
        self.input_duration_key = input_duration_key
        self.device = device
        self.batch_size = batch_size
        self.generate_task = generate_task
        self.generate_language = generate_language
        if torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            raise NotImplementedError(torch_dtype + " is not implemented!")

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.pretrained_model, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.pretrained_model)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=self.batch_size,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def process(self):
        json_list = load_manifest(Path(self.input_manifest_file))
        json_list_sorted = sorted(json_list, key=lambda d: d[self.input_duration_key], reverse=True)

        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)

        with Path(self.output_manifest_file).open('w') as f:
            start_index = 0
            for _ in tqdm(range(len(json_list_sorted) // self.batch_size)):
                batch = json_list_sorted[start_index : start_index + self.batch_size]
                start_index += self.batch_size
                audio_files = [item[self.input_audio_key] for item in batch]
                results = self.pipe(
                    audio_files, generate_kwargs={"language": self.generate_language, "task": self.generate_task}
                )

                for i, item in enumerate(batch):
                    item[self.output_text_key] = results[i]["text"]
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')