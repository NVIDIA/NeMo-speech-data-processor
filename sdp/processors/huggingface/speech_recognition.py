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


class ASRWhisper(BaseProcessor):
    """
    Simple example to transcribe using ASR Whisper model from HuggingFace.
    There are many ways to improve it: make batch inference, split long files, return predicted language, etc.

    Args:
        pretrained_model (str): name of pretrained model on HuggingFace.
        output_text_field (str): field to save transcription result.
        pad_or_trim_length (int): Audio duration to pad or trim (number of samples). Counted as sample_rate * n_seconds i.e.: 16000*30=480000
        device (str): Inference device.
    """

    def __init__(
        self,
        pretrained_model: str,
        output_text_key: str,
        pad_or_trim_length: int = None,
        device: str = None,
        output_lang_key: str = "lid",
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            import torch
            import whisper
        except:
            raise ImportError("Need to install whisper: pip install -U openai-whisper")

        logger.warning("This is an example processor, for demonstration only. Do not use it for production purposes.")
        self.whisper = whisper
        self.pretrained_model = pretrained_model
        self.output_text_key = output_text_key
        self.device = device
        self.output_lang_key = output_lang_key
        self.pad_or_trim_length = pad_or_trim_length
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        self.model = whisper.load_model(self.pretrained_model)
        self.model.to(self.device)

    def process(self):
        json_list = load_manifest(Path(self.input_manifest_file))

        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)

        with Path(self.output_manifest_file).open('w') as f:
            for item in tqdm(json_list):
                pred_text, pred_lang = self.whisper_infer(item["audio_filepath"])

                item[self.output_text_key] = pred_text
                item[self.output_lang_key] = pred_lang
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def whisper_infer(self, audio_path):
        audio = self.whisper.load_audio(audio_path)

        audio = self.whisper.pad_or_trim(audio, length=self.pad_or_trim_length)
        mel = self.whisper.log_mel_spectrogram(audio)
        mel = mel.to(self.device)

        _, probs = self.model.detect_language(mel)
        lang = max(probs, key=probs.get)

        options = self.whisper.DecodingOptions(fp16=False)
        result = self.whisper.decode(self.model, mel, options)
        return result.text, lang


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
