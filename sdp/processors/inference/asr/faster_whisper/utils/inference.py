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
import time
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import librosa

import hydra
from hydra.core.config_store import ConfigStore

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import traceback

from .logger import logger
from .dataloader import WhisperDataloader
from .distribute import init_distributed
from .utils import serialize, join_rank_manifests

from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.vad import VadOptions
from faster_whisper.audio import decode_audio
from omegaconf import OmegaConf, MISSING


@dataclass
class InferenceConfig:
    language: Optional[str] = None
    task: str = "transcribe"
    log_progress: bool = False
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1
    length_penalty: float = 1
    repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    temperature: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    compression_ratio_threshold: Optional[float] = 2.4
    log_prob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    condition_on_previous_text: bool = True
    prompt_reset_on_temperature: float = 0.5
    initial_prompt: Optional[Any] = None
    prefix: Optional[str] = None
    suppress_blank: bool = True
    suppress_tokens: Optional[List[int]] = field(default_factory=lambda: [-1])
    without_timestamps: bool = True
    max_initial_timestamp: float = 1.0
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'“¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、"
    multilingual: bool = False
    vad_filter: bool = True
    vad_parameters: Optional[VadOptions] = None
    max_new_tokens: Optional[int] = None
    chunk_length: Optional[int] = None
    clip_timestamps: Optional[List[dict]] = None
    hallucination_silence_threshold: Optional[float] = None
    batch_size: int = 8
    hotwords: Optional[str] = None
    language_detection_threshold: Optional[float] = 0.5
    language_detection_segments: int = 1

@dataclass
class ModelConfig:
    model_size_or_path: str = MISSING
    device: str = "auto"
    device_index: Optional[List[int]] = field(default_factory=lambda: [0])
    compute_type: str = "default"
    cpu_threads: int = 0
    num_workers: int = 1
    download_root: Optional[str] = None
    local_files_only: bool = False
    files: Optional[dict] = None


@dataclass
class DatasetConfig:
    manifest_filepath: str = MISSING
    output_dir: str = MISSING
    skip_corrupted: bool = False
    save_timestamps_separately: bool = True
    offset: bool = False


@dataclass
class WhisperInferenceConfig:
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    inference: InferenceConfig = field(default_factory=lambda: InferenceConfig())
    language_detection_only: bool = False


class SpeechInferenceModule(pl.LightningModule):
        def __init__(self, 
                     model, 
                     config: WhisperInferenceConfig, 
                     output_manifest_file):
            
            super(SpeechInferenceModule, self).__init__()
            self.model = model
            self.config = config
            self.output_manifest_file = output_manifest_file

        def write_batch(self, batch, results):
            for i_sample in range(len(batch['audio_filepath'])):
                sample = dict()
                for field in batch:
                    if isinstance(batch[field], torch.Tensor):
                        sample[field] = batch[field][i_sample].item()
                    else:
                        sample[field] = batch[field][i_sample]
                
                sample.update(results[i_sample])
                self.output_manifest_file.write(json.dumps(sample) + '\n')
                self.output_manifest_file.flush()    
       
        def write_words(self, filename: str, words: List[Dict]):
            output_manifest_filepath = os.path.join(self.config.dataset.output_dir, 'words', f'{filename}.json')
            with open(output_manifest_filepath, 'w', encoding = 'utf8') as output_manifest:
                for word in words:
                    line = json.dumps(word)
                    output_manifest.writelines(f'{line}\n')
                return output_manifest_filepath
       
        def write_timestamps(self, filename: str, segments: List[Dict]):
            output_segments_filepath = os.path.join(self.config.dataset.output_dir, 'segments', f'{filename}.json')
            sample_words = []
            with open(output_segments_filepath, 'w', encoding = 'utf8') as output_manifest:
                for segment in segments:
                    words = segment.pop('words')
                    if self.config.inference.word_timestamps:
                        for word in words:
                            word['segment_id'] = segment['id']
                            sample_words.append(word)
                    
                    line = json.dumps(segment)
                    output_manifest.writelines(f'{line}\n')
            
            output_words_filepath = None
            if self.config.inference.word_timestamps:
                output_words_filepath = self.write_words(filename, sample_words)
            
            return dict(segments = output_segments_filepath, words = output_words_filepath)
            
        
        def get_audio_segment(self, audio_filepath: str, offset: float, duration: float):
            audio, sr = librosa.load(audio_filepath, sr=None)
            start_sample = int(offset * sr)
            end_sample = int((offset + duration) * sr)
            audio_segment = audio[start_sample : end_sample]
            return audio_segment 

        def predict_step(self, batch, batch_idx):
            cfg_dict = OmegaConf.to_container(self.config.inference, resolve=True)

            start_time = time.time()
            rank = self.config.model.device_index
            
            logger.info(f"Processing batch {batch_idx} on Rank {rank}.")

            results = []

            for i, audio_filepath in enumerate(batch['audio_filepath']):
                if self.config.language_detection_only:
                    try:
                        audio = decode_audio(audio_filepath)
                        language, language_probability, all_language_probs = self.model.model.detect_language(audio = audio,
                                                vad_filter = self.config.inference.vad_filter,
                                                vad_parameters = self.config.inference.vad_parameters,
                                                language_detection_segments = self.config.inference.language_detection_segments,
                                                language_detection_threshold = self.config.inference.language_detection_threshold)
                    except Exception:
                        if self.config.dataset.skip_corrupted:
                            logger.warning(f"Sample can't be processed: {audio_filepath}. Skipping.")
                            continue
                        else:
                            traceback.print_exc()
                            exit(1)
                    
                    result = dict(language = language, language_probability = language_probability)
                else:
                    try:
                        if self.config.dataset.offset:
                            cfg_dict['audio'] = self.get_audio_segment(audio_filepath, batch['offset'][i], batch['duration'][i])
                        else:
                            cfg_dict['audio'] = audio_filepath
                    
                        segments, info = self.model.transcribe(**cfg_dict)
                    
                    except Exception:
                        if self.config.dataset.skip_corrupted:
                            logger.warning(f"Sample can't be transcribed: {audio_filepath}. Skipping.")
                            continue
                        else:
                            traceback.print_exc()
                            exit(1)
                    
                    result = serialize(info)
                    result.pop('all_language_probs', None)
                    result.pop('transcription_options', None)
                    result.pop('vad_options', None)
            
                    _segments = []
                    for segment in segments:
                        _segments.append(serialize(segment))
                    segments = _segments

                    if self.config.dataset.save_timestamps_separately:
                        audio_filename = os.path.splitext(os.path.basename(audio_filepath))[0]
                        timestamps_filepaths = self.write_timestamps(audio_filename, segments)
                        result.update(timestamps_filepaths)
                    else:
                        result['segments'] = segments
                        
                    
                    pred_text = ' '.join(str(segment['text']) for segment in segments).strip()
                    result['pred_text'] = pred_text

                results.append(result)

            self.write_batch(batch, results)
            logger.info(f"Batch {batch_idx} processed in {time.time() - start_time:.2f} seconds.")

def run_task(cfg, distributed: bool = False):
    cfg.model.device = "cuda" if torch.cuda.is_available() and cfg.model.device in ['auto', 'cuda'] else "cpu"
    num_replicas = len(cfg.model.device_index)

    if distributed:
        cfg.model.device_index = int(os.getenv('RANK', 0))
    
    start_time = time.time()
    model = WhisperModel(**cfg.model)
    batched_model = BatchedInferencePipeline(model=model)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds. Moved to {cfg.model.device} {cfg.model.device_index}")
   
    dataloader = WhisperDataloader(manifest_filepath = cfg.dataset.manifest_filepath, 
                                   batch_size = cfg.inference.batch_size,
                                   num_replicas = num_replicas,
                                   rank = cfg.model.device_index[0], #remove [0]
                                  )
    
    trainer = pl.Trainer(
        devices=1, #torch.cuda.device_count(),
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )
    
    start_time = time.time()
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)

    if cfg.dataset.save_timestamps_separately:
        os.makedirs(os.path.join(cfg.dataset.output_dir, f'segments'), exist_ok=True)
        if cfg.inference.word_timestamps:
            os.makedirs(os.path.join(cfg.dataset.output_dir, f'words'), exist_ok=True)

    output_manifest_filepath = os.path.join(cfg.dataset.output_dir, f"prediction.json")
    if distributed:
        output_manifest_filepath = output_manifest_filepath.replace('.json', f'_{str(cfg.model.device_index).zfill(3)}.json')
    
    output_manifest_file = open(output_manifest_filepath, "w", encoding='utf8')

    inference_module = SpeechInferenceModule(model=batched_model, 
                                             config=cfg,
                                             output_manifest_file=output_manifest_file)
    
    trainer.predict(inference_module, dataloaders=dataloader)
    output_manifest_file.close()

    logger.info(f"Total inference time: {time.time() - start_time:.2f} seconds.")
    return output_manifest_filepath

@hydra.main(config_path=None, config_name="whisper_config", version_base=None)
def main(cfg):
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")
    if cfg.model.device in ['auto', 'cuda'] and (cfg.model.device_index == [-1] or len(cfg.model.device_index) > 1):
        cfg.model.device_index = validate_device_ids(cfg.model.device_index)
        rank_output_filepaths = init_distributed(cfg.model.device_index, run_task, cfg, distributed = True)
    else:
        rank_output_filepaths = [run_task(cfg, distributed = False)]
    
    return join_rank_manifests(rank_output_filepaths)
    
cs = ConfigStore.instance()
cs.store(name="whisper_config", node=WhisperInferenceConfig)

if __name__ == "__main__":
    main()