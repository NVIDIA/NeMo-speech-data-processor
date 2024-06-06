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
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

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
        desired_sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except:
            raise ImportError("Need to install transformers: pip install accelerate transformers senterprice")

        logger.warning("This is an example processor, for demonstration only. Do not use it for production purposes.")
        self.pretrained_model = pretrained_model
        self.input_audio_key = input_audio_key
        self.output_text_key = output_text_key
        self.input_duration_key = input_duration_key
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.generate_task = generate_task
        self.generate_language = generate_language
        self.desired_sample_rate = desired_sample_rate
        self.torch_dtype = getattr(torch, torch_dtype)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.pretrained_model, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.pretrained_model)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=self.batch_size,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        self.failed_files = []

    def process(self):
        json_list = load_manifest(Path(self.input_manifest_file))
        json_list_sorted = sorted(json_list, key=lambda d: d[self.input_duration_key], reverse=True)

        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)

        with Path(self.output_manifest_file).open('w') as f:
            start_index = 0
            while start_index < len(json_list_sorted):
                batch = json_list_sorted[start_index : start_index + self.batch_size]
                start_index += self.batch_size

                if self.batch_size == 1:
                    audio_file = batch[0][self.input_audio_key]
                    try:
                        waveform, orig_sampling_rate = torchaudio.load(audio_file)
                        if orig_sampling_rate != self.desired_sample_rate:
                            resampler = torchaudio.transforms.Resample(
                                orig_freq=orig_sampling_rate, new_freq=self.desired_sample_rate
                            )
                            waveform = resampler(waveform)
                        result = self.pipe(
                            waveform.squeeze(),
                            generate_kwargs={"language": self.generate_language, "task": self.generate_task},
                        )
                        batch[0][self.output_text_key] = result["text"]
                        f.write(json.dumps(batch[0], ensure_ascii=False) + '\n')
                    except Exception as e:
                        logger.error(f"Failed to process file {audio_file}: {str(e)}")
                        self.failed_files.append(audio_file)
                else:
                    logger.warning("Batch size greater than 1: Damaged files will not be handled individually.")
                    audio_files = [item[self.input_audio_key] for item in batch]
                    results = self.pipe(
                        audio_files, generate_kwargs={"language": self.generate_language, "task": self.generate_task}
                    )

                    for i, item in enumerate(batch):
                        item[self.output_text_data] = results[i]["text"]
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

        if self.failed_files:
            logger.error(f"Failed to process the following files: {self.failed_files}")


class ASRSeamless(BaseProcessor):
    """
    An audio speech recognition (ASR) processor class utilizing the Seamless model from Facebook's Hugging Face repository to transcribe audio files.

     Args:
        device (str): Computing device for processing, either CUDA-enabled GPU or CPU. Defaults to GPU if available.
        input_key (str): Key in the input manifest file indicating the path to the audio file.
        output_key (str): Key where the transcribed text by the model will be stored.
        limit (int): Maximum percentage of files to process, helping manage resource use on large datasets.
        src_lang (str): Source language code used for processing audio input.
        tgt_lang (str): Target language code for the output transcription.
    """

    def __init__(
        self,
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
        input_key: str = "audio_filepath",
        output_key: str = "model_transcribed_text",
        limit: int = 100,
        tgt_lang="hye",
        src_lang="hye",
        **kwargs,
    ):
        self.device = device
        self.input_key = input_key
        self.output_key = output_key
        self.limit = limit
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

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
                audio_file_path = entry[self.input_key]

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
                    audios=waveform.squeeze().cpu().numpy(),
                    src_lang=self.src_lang,
                    sampling_rate=16000,
                    return_tensors="pt",
                ).to(device)

                outputs = model.generate(**audio_inputs, tgt_lang=self.tgt_lang)

                transcribed_text = processor.batch_decode(outputs, skip_special_tokens=True)

                entry[self.output_key] = str(transcribed_text[0]).strip()
                entries.append(entry)

            with open(self.output_manifest_file, 'w', encoding='utf-8') as f_out:
                for entry in entries:
                    json.dump(entry, f_out, ensure_ascii=False)
                    f_out.write("\n")

        if self.failed_files:
            print(f"Failed to process the following files: {self.failed_files}")
