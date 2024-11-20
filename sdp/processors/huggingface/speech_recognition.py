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
import os
import math
import json
from pathlib import Path

import tempfile
from tqdm import tqdm

from datasets import load_dataset, Audio, Dataset
from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest
from typing import Optional, Union, List

from multiprocessing import Process, Queue

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
        accelerator: str = None,
        devices: Union[int, List] = -1,
        attn_implementation: str = "flash_attention_2",
        batch_size: int = 1,
        beam_size: int = 1,
        chunk_length_s: int = 0,
        torch_dtype: str = "float32",
        generate_task: str = "transcribe",
        generate_language: str = "english",
        max_new_tokens: Optional[int] = None,
        out_dir: Optional[str] = None,
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
        self.accelerator = accelerator
        self.devices = devices
        self.attn_implementation = attn_implementation
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.chunk_length_s = chunk_length_s
        self.generate_task = generate_task
        self.generate_language = generate_language
        self.max_new_tokens = max_new_tokens
        self.out_dir = out_dir

        if not self.out_dir:
            self.out_dir = tempfile.mkdtemp()

        if torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            raise NotImplementedError(torch_dtype + " is not implemented!")

        if self.accelerator is None:
            if torch.cuda.is_available():
                torch.multiprocessing.set_start_method('spawn')
                self.accelerator = "cuda"
            else:
                self.accelerator = "cpu"

        if self.accelerator == "cpu":
            self.map_devices = ["cpu"]
        elif self.devices == -1:
            self.map_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            self.map_devices = [f"cuda:{i}" for i in self.devices]

        self.devices_model = {}
        self.get_model_on_devices(AutoModelForSpeechSeq2Seq)

        # processor does not require gpu so it is instantiated here
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model)


    def get_model_on_devices(self, model_class):

        for device in self.map_devices:
            self.devices_model[device] = model_class.from_pretrained(
                self.pretrained_model, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True,
                use_safetensors=True, 
                attn_implementation=self.attn_implementation, 
                device_map=device
                )
            
    def write_results(self, entries, results, device):
        result_file = Path(self.out_dir, f"results_{device}").with_suffix('.json')

        with open(result_file, "w") as f:
            for entry, result in zip(entries, results):
                entry[self.output_text_key] = result
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

    def combine_results(self):
        print('COMBINING RESULTS')
        cmd = f"cat {self.out_dir}/results_*.json > {self.output_manifest_file}"
        os.system(cmd)
        

    def infer(self, dataset, entries, device):
        all_transcriptions = []

        audios = dataset["audio_filepath"]

        for start_idx in tqdm(range(0, len(audios) + 1, self.batch_size), desc=f'Processing on {device}'):
            batch = [x["array"] for x in audios[start_idx:start_idx+self.batch_size]]
            max_audio_dur = entries[start_idx:start_idx+self.batch_size][-1]['duration']

            if max_audio_dur < 30:
                # we in-fact have short-form -> pre-process accordingly
                inputs = self.processor(batch, 
                                        return_tensors="pt", 
                                        return_attention_mask=True,
                                        sampling_rate=16_000)
            else:
                inputs = self.processor(batch, 
                                        return_tensors="pt", 
                                        truncation=False, 
                                        padding="longest", 
                                        return_attention_mask=True, 
                                        sampling_rate=16_000)
            
            
            inputs = inputs.to(device).to(self.torch_dtype)

            generated_ids = self.devices_model[device].generate(language=self.generate_language,
                                                                task=self.generate_task,
                                                                num_beams=self.beam_size,
                                                                **inputs)

            transcription = self.processor.batch_decode(generated_ids, 
                                                        skip_special_tokens=True)
            
            all_transcriptions.extend(transcription)

        self.write_results(entries, all_transcriptions, device)

    def parallel_infer(self, dataset, entries):

        block_size = math.ceil(len(dataset) / len(self.devices_model))

        processes = []

        for idx, device in enumerate(self.devices_model):
            dataset_block = dataset[idx * block_size : idx * block_size + block_size]
            entries_block = entries[idx * block_size : idx * block_size + block_size]

            process = Process(target=self.infer, args=(dataset_block, entries_block, device))
            process.start()
            print(f'STARTED PROCESS ON {device} with {len(dataset_block)} entries.')
            processes.append(process)

        for process in processes:
            process.join()

    def process(self):
        json_list = load_manifest(Path(self.input_manifest_file))
        json_list_sorted = sorted(json_list, key=lambda d: d[self.input_duration_key], reverse=True)

        dataset = Dataset.from_list(json_list_sorted)
        dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

        if len(self.devices_model) == 1:
            self.infer(dataset, json_list_sorted, self.map_devices[0])

        else:
            self.parallel_infer(dataset, json_list_sorted)
            
        self.combine_results()
