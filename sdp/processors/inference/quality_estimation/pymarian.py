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
import json
from tqdm import tqdm
import termplotlib as tpl
import numpy as np

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor

class CometoidWMTQualityEstimation(BaseParallelProcessor):
    """
    A processor for estimating translation quality using pretrained COMET-like models 
    based on MarianNMT and the pymarian Evaluator.

    This processor evaluates the quality of source-target text pairs (bitext) using 
    COMETOID-style quality estimation and appends the resulting score to each dataset entry.

    Args:
        source_text_field (str): The key in the data entry containing the source (original) text.
        target_text_field (str): The key in the data entry containing the target (translated) text.
        model_name_or_path (str): Hugging Face model name or path to local model checkpoint.
        vocab_path (str, optional): Path to the vocabulary file. If None and model is from HF, it will be downloaded.
        save_model_to (str, optional): Directory to download and cache the model and vocab.
        mini_batch (int): Mini-batch size for evaluation.
        maxi_batch (int): Maxi-batch size for evaluation.
        output_field (str): The name of the field where the quality score will be saved in the output manifest.
        device_type (str): Device type to use: 'cpu' or 'gpu'.
        num_devices (int): Number of CPU threads or GPU devices to use. Use -1 to use all available.
        chunksize (int): Number of lines to process in each chunk.

    Returns:
        A manifest file where each entry has an added key (`output_field`) with the computed score.
    
    .. note::
        This processor uses MarianNMT models fine-tuned for quality estimation. See https://marian-nmt.github.io/.
    """

    # Mapping of supported model aliases to Hugging Face repo paths
    MODEL_NAME_TO_HF_PATH = {
        "cometoid-wmt23": "marian-nmt/cometoid22-wmt23",
        "cometoid-wmt23-mqm": "marian-nmt/cometoid22-wmt23",
    }

    # Marian evaluation arguments depending on device
    MARIAN_GPU_ARGS = "-w 8000 -d {device_indicies}"
    MARIAN_CPU_ARGS = "-w 2000 --cpu-threads {num_threads}"

    def __init__(self,
                 source_text_field: str,
                 target_text_field: str,
                 model_name_or_path: str,
                 vocab_path: str = None,
                 save_model_to: str = None,
                 mini_batch: int = 16,
                 maxi_batch: int = 96,
                 output_field: str = 'cometoid_score',
                 device_type: str = 'cpu',
                 num_devices: int = -1,
                 chunksize = 5000,
                 **kwargs,
    ):
        super().__init__(max_workers = num_devices, chunksize = chunksize, in_memory_chunksize = chunksize, **kwargs)
        self.source_text_field = source_text_field
        self.target_text_field = target_text_field
        self.model_name_or_path = model_name_or_path
        self.vocab_path = vocab_path
        self.save_model_to = save_model_to
        self.device_type = device_type
        self.mini_batch = mini_batch
        self.maxi_batch = maxi_batch
        self.output_field = output_field
        self.model = None

    def load_model(self):
        from pymarian import Evaluator
        from huggingface_hub import hf_hub_download

        """
        Load the model and vocabulary from Hugging Face if necessary.
        Assemble command-line arguments for launching pymarian Evaluator.
        Depending on the device (CPU/GPU), configure parallelism parameters.
        """
        repo_id = None
        if self.model_name_or_path in self.MODEL_NAME_TO_HF_PATH:
            repo_id = self.MODEL_NAME_TO_HF_PATH[self.model_name_or_path]
            self.model_name_or_path = hf_hub_download(repo_id, filename="checkpoints/marian.model.bin", local_dir = self.save_model_to)
       
        if not os.path.exists(self.model_name_or_path):
            raise ValueError(f'`model_name_or_path`: model name is not valid or model path does not exist ({self.model_name_or_path}).')
        
        if not self.vocab_path and repo_id is not None:
            self.vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.spm", local_dir = self.save_model_to)
        
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f'`vocab_path`: path does not exist ({self.vocab_path}).')
    
        marian_args = f"-m {self.model_name_or_path} -v {self.vocab_path} {self.vocab_path} --like comet-qe"

        if self.device_type == "cpu":
            max_available_cpus = os.cpu_count()
            if self.max_workers == -1 or self.max_workers > max_available_cpus:
                self.max_workers = max_available_cpus

            cpu_args = self.MARIAN_CPU_ARGS.format(num_threads = self.max_workers)
            marian_args += f' {cpu_args}'
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    max_available_gpus = torch.cuda.device_count()
                if self.max_workers == -1 or self.max_workers > max_available_gpus:
                    self.max_workers = max_available_cpus
            except Exception:
                pass

            device_indicies = ' '.join([str(i) for i in range(self.max_workers)])
            gpu_args = self.MARIAN_GPU_ARGS.format(device_indicies = device_indicies)
            marian_args += f' {gpu_args}'
        
        marian_args += f' --mini-batch {self.mini_batch} --maxi-batch {self.maxi_batch}'
        
        self.model = Evaluator(marian_args)

    def process_dataset_entry(self):
        pass

    def process(self):
        """
        Process the entire manifest in chunks.
        For each pair of texts (source–target), compute the translation quality score.
        Save the resulting scores in output_manifest_file.
        """
        self.load_model()
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        metrics = []

        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for manifest_chunk in self._chunk_manifest():
                entries = []
                bitext_pairs = []
                for data_entry in manifest_chunk:
                    src = str(data_entry[self.source_text_field]).replace('\t', ' ')
                    tgt = str(data_entry[self.target_text_field]).replace('\t', ' ')
                    bitext_pairs.append(f'{src}\t{tgt}')
                    entries.append(data_entry)
                    
                scores = self.model.evaluate(bitext_pairs)
                for entry, score in tqdm(zip(entries, scores)):
                    metrics.append(score)
                    entry[self.output_field] = score
                    json.dump(entry, fout, ensure_ascii=False)
                    self.number_of_entries += 1
                    fout.write("\n")
        
        self.finalize(metrics)
    
    def finalize(self, metrics):
        """
        Print statistics about the quality scores: histogram, min, max, mean, median.
        Use termplotlib to render the histogram directly in the terminal.
        """
        logger.info("Total number of entries after processing: %d", self.number_of_entries)
        logger.info("Histogram of scores:")

        bins = np.arange(0, 1.1, 0.1)
        hist, bin_edges = np.histogram(metrics, bins=bins)

        labels = []
        for i in range(len(bin_edges) - 1):
            left = f"{bin_edges[i]:.1f}"
            right = f"{bin_edges[i+1]:.1f}"
            if i < len(bin_edges) - 2:
                labels.append(f"[{left}–{right})")
            else:
                labels.append(f"[{left}–{right}]")
        
        fig = tpl.figure()
        fig.barh(hist, labels)
        fig.show()

        logger.info(f"Min score: {np.min(metrics):.4f}")
        logger.info(f"Max score: {np.max(metrics):.4f}")
        logger.info(f"Mean score: {np.mean(metrics):.4f}")
        logger.info(f"Median score: {np.median(metrics):.4f}")















