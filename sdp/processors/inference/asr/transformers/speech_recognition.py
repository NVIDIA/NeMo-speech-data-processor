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
from pathlib import Path

from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest
from typing import Optional

class ASRTransformers(BaseProcessor):
    """This processor transcribes audio files using HuggingFace ASR Transformer models.

    It processes audio files from the manifest and adds transcriptions using the specified
    pre-trained model from HuggingFace.

    Args:
        pretrained_model (str): Name of pretrained model on HuggingFace.
        output_text_key (str): Key to save transcription result in the manifest.
        input_audio_key (str): Key to read audio file paths from the manifest. Default: "audio_filepath".
        input_duration_key (str): Key for audio duration in the manifest. Default: "duration".
        device (str): Inference device (e.g., "cuda", "cpu"). Default: None.
        batch_size (int): Inference batch size. Default: 1.
        chunk_length_s (int): Length of audio chunks in seconds. Default: 0.
        torch_dtype (str): Tensor data type for model inference. Default: "float32".
        generate_task (str): Task type for generation. Default: "transcribe".
        generate_language (str): Language for generation. Default: "english".
        max_new_tokens (int, Optional): Maximum number of new tokens to generate. Default: None.

    Returns:
        A manifest with transcribed text added to each entry under the specified output_text_key.

    """

    def __init__(
        self,
        pretrained_model: str,
        output_text_key: str,
        input_audio_key: str = "audio_filepath",
        input_duration_key: str = "duration",
        device: str = None,
        batch_size: int = 1,
        chunk_length_s: int = 0,
        torch_dtype: str = "float32",
        generate_task: str = "transcribe",
        generate_language: str = "english",
        max_new_tokens: Optional[int] = None,
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
            self.pretrained_model, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        # Check if using Whisper/Seamless or NVIDIA model based on the model name
        self.is_whisper_or_seamless = any(x in self.pretrained_model.lower() for x in ['whisper', 'seamless'])
        
        # Only set language in generation config for Whisper/Seamless models
        if self.is_whisper_or_seamless and self.generate_language:
            self.model.generation_config.language = self.generate_language

        processor = AutoProcessor.from_pretrained(self.pretrained_model)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=self.max_new_tokens,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            return_timestamps=self.is_whisper_or_seamless,  # Only set return_timestamps for Whisper/Seamless models
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def process(self):
        json_list = load_manifest(Path(self.input_manifest_file))
        json_list_sorted = sorted(json_list, key=lambda d: d[self.input_duration_key], reverse=True)

        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)

        with Path(self.output_manifest_file).open("w") as f:
            start_index = 0
            for _ in tqdm(range(len(json_list_sorted) // self.batch_size)):
                batch = json_list_sorted[start_index : start_index + self.batch_size]
                start_index += self.batch_size
                audio_files = [item[self.input_audio_key] for item in batch]
                
                # Only pass generate_kwargs for Whisper/Seamless models
                if self.is_whisper_or_seamless and self.generate_language and self.generate_task:
                    results = self.pipe(
                        audio_files, generate_kwargs={"language": self.generate_language, "task": self.generate_task}
                    )
                else:
                    results = self.pipe(audio_files)

                for i, item in enumerate(batch):
                    item[self.output_text_key] = results[i]["text"]
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
