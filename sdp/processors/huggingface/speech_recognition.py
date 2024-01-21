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
import os
from pathlib import Path

import librosa
from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor


class ASRWhisper(BaseProcessor):
    """
    Simple example to transcribe using ASR Whisper model from HuggingFace.
    There are many ways to improve it: make batch inference, split long files, return predicted language, etc.

    Args:
        pretrained_model (str): name of pretrained model on HuggingFace.
        output_text_field (str): field to save transcription result.
        device (str): Inference device.
        batch_size (int): Inference batch size. Defaults to 1.
    """

    def __init__(
        self,
        pretrained_model: str,
        output_text_field: str,
        device: str = None,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        import torch
        import whisper  # pip install -U openai-whisper

        self.log_in_loop = 0
        self.whisper = whisper
        self.pretrained_model = pretrained_model
        self.output_text_field = output_text_field
        self.device = device
        self.batch_size = batch_size
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        self.model = whisper.load_model(self.pretrained_model)

    def process(self):
        json_list = self.read_input_manifest()
        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)

        with Path(self.output_manifest_file).open('w', encoding='utf-8') as f:
            for item in tqdm(json_list):
                pred_text, pred_lang = self.whisper_infer(item["audio_filepath"])

                item[self.output_text_field] = pred_text
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def whisper_infer(self, audio_path):
        try:
            audio = self.whisper.load_audio(audio_path)
        except FileNotFoundError as fe:  # whisper bug: https://github.com/openai/whisper/discussions/109
            if os.path.exists(audio_path):
                if self.log_in_loop < 1:
                    self.log_in_loop += 1  # do not log more than once in a loop
                    logger.warning(f'While reading audio whisper results in {fe.__str__()}')
                    logger.info('Trying to read the audio with librosa')
            else:
                raise fe

            samplerate = os.environ.get('RESAMPLED_RATE', 16000)
            audio, sr = librosa.load(audio_path, sr=samplerate)

        audio = self.whisper.pad_or_trim(audio)
        mel = self.whisper.log_mel_spectrogram(audio)
        mel = mel.to(self.device)

        _, probs = self.model.detect_language(mel)
        lang = max(probs, key=probs.get)

        options = self.whisper.DecodingOptions()
        result = self.whisper.decode(self.model, mel, options)
        return result.text, lang


class ASRTransformer(BaseProcessor):
    """
    Processor to transcribe using ASR Transformer model from HuggingFace.

    Args:
        pretrained_model (str): name of pretrained model on HuggingFace.
        output_text_field (str): field to save transcription result.
        device (str): Inference device.
        batch_size (int): Inference batch size. Used only batch_size = 1 TODO: support batch_size > 1
        torch_dtype (str): Tensor data type. Default to "float32"
    """

    def __init__(
        self,
        pretrained_model: str,
        output_text_field: str,
        device: str = None,
        batch_size: int = 1,  # TODO: support batch_size > 1
        torch_dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(**kwargs)
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.pretrained_model = pretrained_model
        self.output_text_field = output_text_field
        self.device = device
        self.batch_size = batch_size
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
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def process(self):
        json_list = self.read_input_manifest()
        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)

        with Path(self.output_manifest_file).open('w', encoding='utf-8') as f:
            for item in tqdm(json_list):
                pred_text = self.pipe(item["audio_filepath"])["text"]

                item[self.output_text_field] = pred_text
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
