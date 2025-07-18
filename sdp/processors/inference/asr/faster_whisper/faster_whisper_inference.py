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
from copy import deepcopy
from tqdm import tqdm
import librosa
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Optional, Any, Dict
from omegaconf import OmegaConf, MISSING

from sdp.logging import logger
from multiprocessing import Pool
import traceback

from sdp.processors.base_processor import BaseProcessor

"""
This module implements `FasterWhisperInference`, a multiprocessing-compatible audio transcription
processor using the FasterWhisper library.

It reads an input manifest, runs inference on available devices (GPU/CPU), and outputs predictions,
including optional timestamp and word-level information.

Classes:
    - InferenceConfig: Configuration for whisper decoding and inference behavior.
    - ModelConfig: Configuration for the Whisper model loading.
    - DatasetConfig: Configuration for dataset input/output handling.
    - WhisperInferenceConfig: Combined config container.
    - FasterWhisperInference: Main processor class for transcribing input audio files in parallel.
"""

def serialize(obj):
    """
    Recursively serializes a dataclass, list, or dict to a JSON-compatible structure.
    
    Args:
        obj (Any): Object to serialize (dataclass, list, or dict).

    Returns:
        JSON-serializable version of the object.
    """
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, list):
        return [serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    return obj

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

    try:
        from faster_whisper.vad import VadOptions
        vad_parameters: Optional[VadOptions] = None
    except ModuleNotFoundError:
        pass
    
    max_new_tokens: Optional[int] = None
    chunk_length: Optional[int] = None
    clip_timestamps: Optional[Any] = "0"
    hallucination_silence_threshold: Optional[float] = None
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


class FasterWhisperInference(BaseProcessor):
    """
    Processor that performs parallel audio transcription using the FasterWhisper model.

    This class reads a manifest of audio files, transcribes them using multiprocessing
    (each device or CPU thread handles a portion), and writes results in a NeMo-compatible manifest.

    Args:
        input_manifest_file (str): Path to the input manifest.
        output_manifest_file (Optional[str]): Path to the output manifest (default: `<output_dir>/predictions_all.json`).
        model_size_or_path (str): Whisper model path or model name (e.g., 'base', 'medium').
        device (str): Device type to use ('auto', 'cuda', or 'cpu').
        num_devices (int): Number of workers/devices to use (-1 = all available).
        model_download_root (Optional[str]): Directory where model checkpoints will be downloaded.
        output_dir (Optional[str]): Directory to store output predictions and timestamps.
        skip_corrupted_audios (bool): Whether to skip audio files that raise exceptions.
        save_timestamps_separately (bool): If True, saves segment/word timestamps as separate files.
        slice_by_offset (bool): If True, slices audio using offset/duration before inference.
        inference (Optional[Dict]): Additional inference parameters for Whisper.
        language_detection_only (bool): If True, only perform language detection.
        in_memory_chunksize (int): Number of samples to load per worker at once.
        audio_filepath_field (str): Name of the field in manifest pointing to audio path.
     
    Returns:
        A final merged manifest file where each line corresponds to the transcription result of an input audio sample.
        The manifest is assembled from multiple per-worker (rank) manifest files, each produced by a separate device or process.

        Each entry contains the following fields:

        - ``language`` (str, optional): Detected language (if language detection is enabled).
        - ``language_probability`` (float, optional): Confidence score of detected language.
        - ``pred_text`` (str): Final transcribed text obtained by concatenating all segment texts.

        One of the following timestamp representations will also be included, depending on the value of `save_timestamps_separately`:

        - If ``save_timestamps_separately=False``:
            - ``segments`` (List[Dict]): List of segment dictionaries with start/end timestamps and transcribed text.

        - If ``save_timestamps_separately=True``:
            - ``segments`` (str): Path to a JSON file containing segment-level timestamps.
            - ``words`` (str, optional): Path to a JSON file containing word-level timestamps (if `word_timestamps=True`).

        The final combined manifest is written to ``output_manifest_file``, which defaults to ``<output_dir>/predictions_all.json``.
    
    .. note::
        Make sure to install the following packages before using this processor:
        
            pip install pytorch-lightning nvidia-cublas-cu12 nvidia-cudnn-cu12==9.* faster_whisper

        Additionally, ensure that the dynamic libraries for cuBLAS and cuDNN are discoverable at runtime:

            export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`

        This is required for CUDA backend components to function correctly when using FasterWhisper with GPU acceleration.

        For detailed configuration options and advanced usage of FasterWhisper, refer to the official repository:
        https://github.com/SYSTRAN/faster-whisper

    """
    def __init__(self, 
                 input_manifest_file: str,
                 output_manifest_file: Optional[str] = None,
                 model_size_or_path: str = "base",
                 device: str = "auto",
                 num_devices: int = -1,
                 compute_type: str = "default",
                 model_download_root: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 skip_corrupted_audios: bool = False,
                 save_timestamps_separately: bool = True,
                 slice_by_offset: bool = False,
                 inference: Optional[Dict] = {},
                 language_detection_only: bool = False,
                 in_memory_chunksize: int = 100000,
                 audio_filepath_field: str = 'audio_filepath',
                 ):
        
        super().__init__(input_manifest_file = input_manifest_file,
                         output_manifest_file = output_manifest_file,
                        )

        #DatasetConfig setup
        if not self.output_manifest_file and not output_dir:
            raise ValueError("Either `output_manifest_file` or `output_dir` must be provided.")
        
        if not output_dir:
            output_dir = os.path.splitext(self.output_manifest_file)[0]
        
        if not self.output_manifest_file:
            self.output_manifest_file = os.path.join(output_dir, 'predictions_all.json')
        
        dataset_cfg = DatasetConfig(manifest_filepath = self.input_manifest_file, 
                                    output_dir = output_dir, 
                                    skip_corrupted = skip_corrupted_audios,
                                    save_timestamps_separately = save_timestamps_separately,
                                    offset = slice_by_offset)

        #InferenceConfig setup
        inference_cfg = OmegaConf.structured(InferenceConfig(**inference))

        #ModelConfig setup
        device, device_ids = self.setup_devices(device, num_devices)
        self.device_ids = device_ids
        model_cfg = ModelConfig(model_size_or_path = model_size_or_path,
                                device = device, compute_type = compute_type,
                                download_root = model_download_root)
        
        #GeneralConfig setup
        self.config = WhisperInferenceConfig(model=model_cfg, 
                                             dataset=dataset_cfg, 
                                             inference=inference_cfg, 
                                            )
        
        #Additional args
        self.audio_filepath_field = audio_filepath_field
        self.language_detection_only = language_detection_only
        self.in_memory_chunksize = in_memory_chunksize

    @staticmethod 
    def setup_devices(device: str = "auto", num_devices: int = -1):
        """
        Determines device type and number of workers to use for inference.

        Returns:
            Tuple[str, List[int]]: Selected device type and list of device indices.
        """
        try:
            import torch
            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False

        if device in ["cuda", "auto"] and TORCH_AVAILABLE:
            cuda_available_workers = torch.cuda.device_count()
            if cuda_available_workers == 0:
                if device == "cuda":
                    raise RuntimeError("GPU was requested, but no CUDA devices are available.")
                else:
                    logger.warning("No GPU found in auto mode — switching to CPU.")
                    device = "cpu"
            else:
                logger.info("CUDA devices found. GPU will be used as workers.")
                device = "cuda"
        elif device == "cpu":
            logger.info("CPU will be used as workers.")
        else:
            raise ValueError(f"Invalid device type: {device}")
        
        if device == "cuda":
            max_available_workers = cuda_available_workers
        else:
            max_available_workers = os.cpu_count()

        if num_devices < -1 or num_devices == 0: 
            raise ValueError(f"Invalid number of workers: {num_devices}.")
        elif num_devices == -1: 
            workers = max_available_workers
            logger.info(f"Using {workers} {device.upper()} worker(s).")
        elif num_devices > max_available_workers:
            workers = max_available_workers
            logger.warning(f"Requested {num_devices} {device.upper()} workers, but only {max_available_workers} {device.upper()} available — using {workers}.")
        else:
            workers = num_devices
            logger.info(f"Using {workers} {device.upper()} worker(s).")
        
        device_ids = list(range(workers))
        return device, device_ids
    
    def prepare(self):
        """
        Creates output directories required for storing prediction and timestamp files.
        """
        os.makedirs(self.config.dataset.output_dir, exist_ok = True)
        if self.config.dataset.save_timestamps_separately:
            os.makedirs(os.path.join(self.config.dataset.output_dir, "segments"), exist_ok = True)
            if self.config.inference.word_timestamps:
                os.makedirs(os.path.join(self.config.dataset.output_dir, "words"), exist_ok = True)
    
    def _chunk_manifest(self):
        """Splits the manifest into smaller chunks defined by ``in_memory_chunksize``."""
        manifest_chunk = []
        for idx, data_entry in enumerate(self.read_manifest(), 1):
            manifest_chunk.append(data_entry)
            if idx % self.in_memory_chunksize == 0:
                yield manifest_chunk
                manifest_chunk = []
        if manifest_chunk:
            yield manifest_chunk

    def read_manifest(self):
        """Reading the input manifest file."""
        if not self.input_manifest_file:
            raise NotImplementedError("Override this method if no input manifest file is used")
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            for line in fin:
                yield json.loads(line)
    
    def get_entries_for_device(self, device_id: int):
        """
        Yields manifest entries assigned to a given device.

        Uses round-robin assignment of sorted entries by duration.
        """
        for chunk in self._chunk_manifest():
            chunk.sort(key=lambda entry: entry["duration"])
            batch = chunk[device_id::len(self.device_ids)]
            for entry in batch:
                yield entry
    
    def get_audio_segment(self, audio_filepath: str, offset: float, duration: float):
        """Loads a segment of audio based on offset and duration."""
        audio, sr = librosa.load(audio_filepath, sr=None)
        start_sample = int(offset * sr)
        end_sample = int((offset + duration) * sr)
        audio_segment = audio[start_sample : end_sample]
        return audio_segment 

    def write_timestamps(self, filename: str, segments: List[Dict]):
        """Saves timestamp information (segments and optionally word-level) to separate files."""

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
        
        def write_words(words: List[Dict]):
            output_manifest_filepath = os.path.join(self.config.dataset.output_dir, 'words', f'{filename}.json')
            with open(output_manifest_filepath, 'w', encoding = 'utf8') as output_manifest:
                for word in words:
                    line = json.dumps(word)
                    output_manifest.writelines(f'{line}\n')
                return output_manifest_filepath

        output_words_filepath = None
        if self.config.inference.word_timestamps:
            output_words_filepath = write_words(output_words_filepath, sample_words)
        
        return dict(segments = output_segments_filepath, words = output_words_filepath)

    def transcribe(self, device_id: int):
        """""
        Transcribes all samples assigned to a given device.

        Loads the Whisper model, reads samples, performs language detection or full transcription,
        and writes predictions to a device-specific output file.
        """

        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio

        model_cfg = deepcopy(self.config.model)
        model_cfg.device_index = [device_id] if model_cfg.device == "cuda" else [0]
        model = WhisperModel(**asdict(model_cfg))

        inference_cfg = OmegaConf.to_container(self.config.inference, resolve=True)

        output_manifest_file = os.path.join(self.config.dataset.output_dir, f'prediction_{device_id}.json')
        
        with open(output_manifest_file, 'w', encoding='utf8') as fout:
            for entry in tqdm(self.get_entries_for_device(device_id), desc = f"Transcribing ({self.config.model.device.upper()} {device_id})"):
                audio_filepath = entry[self.audio_filepath_field]

                if self.language_detection_only:
                    try:
                        audio = decode_audio(audio_filepath)
                        features = model.feature_extractor(audio)
                        language, language_probability, all_language_probs = model.detect_language(features = features,
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
                            audio = self.get_audio_segment(audio_filepath, entry['offset'], entry['duration'])
                        else:
                            audio = audio_filepath
                    
                        segments, info = model.transcribe(audio = audio, **inference_cfg)
                    
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
                
                entry.update(result)
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fout.flush()
        
        return output_manifest_file
    
    def process(self):
        """
        Main entry point for the processor.

        Prepares directories, distributes transcription tasks across devices, and aggregates results
        into the final output manifest file.
        """
        self.prepare()

        with Pool(processes=len(self.device_ids)) as pool:
            output_rank_manifests = pool.map(self.transcribe, self.device_ids)
        
        with open(self.output_manifest_file, 'w', encoding='utf8') as output_manifest:
            for rank_manifest_filepath in tqdm(output_rank_manifests, desc = "Writing output manifest.."):
                with open(rank_manifest_filepath, 'r', encoding='utf8') as rank_manifest:
                    for line in rank_manifest:
                        output_manifest.writelines(line)