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

import torch
import json
from pathlib import Path

from tqdm import tqdm

from datasets import load_dataset, Audio, Dataset
from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest
from typing import Optional

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
        chunk_length_s (int): Length of the chunks (in seconds) into which the input audio should be divided.
            Note: Some models perform the chunking on their own (for instance, Whisper chunks into 30s segments also by maintaining the context of the previous chunks).
        torch_dtype (str): Tensor data type. Default to "float32"
        max_new_tokens (Optional[int]): The maximum number of new tokens to generate.
            If not specified, there is no hard limit on the number of tokens generated, other than model-specific constraints.
    """

    def __init__(
        self,
        pretrained_model: str,
        output_text_key: str,
        input_audio_key: str = "audio_filepath",
        input_duration_key: str = "duration",
        device: str = None,
        batch_size: int = 1,
        beam_size: int = 1,
        chunk_length_s: int = 0,
        torch_dtype: str = "float32",
        generate_task: str = "transcribe",
        generate_language: str = "english",
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
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
        self.beam_size = beam_size
        self.chunk_length_s = chunk_length_s
        self.generate_task = generate_task
        self.generate_language = generate_language
        self.max_new_tokens = max_new_tokens
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
            self.pretrained_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="flash_attention_2", device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model, torch_dtype=torch.bfloat16)


    def process(self):
        json_list = load_manifest(Path(self.input_manifest_file))
        json_list_sorted = sorted(json_list, key=lambda d: d[self.input_duration_key], reverse=True)

        #dataset = load_dataset("json", data_files=self.input_manifest_file)
        dataset = Dataset.from_list(json_list_sorted)
        dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))


        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)

        transcriptions = []

        start_index = 0
        for _ in tqdm(range((len(json_list_sorted) // self.batch_size) + 1)):
            audio = dataset[start_index : start_index + self.batch_size]["audio_filepath"]
            start_index += self.batch_size

            audio = [x["array"] for x in audio]

            inputs = self.processor(audio, return_tensors="pt", torch_dtype=torch.bfloat16, truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000)

            if inputs.input_features.shape[-1] < 3000:
                # we in-fact have short-form -> pre-process accordingly
                inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000)
            
            inputs = inputs.to(self.device).to(torch.bfloat16)

            generated_ids = self.model.generate(language=self.generate_language,
                                                task=self.generate_task,
                                                num_beams=self.beam_size,
                                                **inputs)

            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            transcriptions.extend(transcription)


        with Path(self.output_manifest_file).open("w") as f:
            for item, transcription in zip(json_list_sorted, transcriptions):
                item[self.output_text_key] = transcription
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
