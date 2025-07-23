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

import json
import os
from dataclasses import dataclass, field, is_dataclass
from typing import List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel, EncDecRNNTModel
from nemo.collections.asr.models.aed_multitask_models import parse_multitask_prompt
from nemo.collections.asr.modules.conformer_encoder import ConformerChangeConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding, MultiTaskDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    prepare_audio_data,
    restore_transcription_order,
    setup_model,
    write_transcription,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.timers import SimpleTimer

"""
Transcribe audio file on a single CPU/GPU. Useful for transcription of moderate amounts of audio data.

# Arguments
  model_path: path to .nemo ASR checkpoint
  pretrained_name: name of pretrained ASR model (from NGC registry)
  audio_dir: path to directory with audio files
  dataset_manifest: path to dataset JSON manifest file (in NeMo formats
  compute_langs: Bool to request language ID information (if the model supports it)
  timestamps: Bool to request greedy time stamp information (if the model supports it) by default None 

  (Optionally: You can limit the type of timestamp computations using below overrides)
  ctc_decoding.ctc_timestamp_type="all"  # (default all, can be [all, char, word, segment])
  rnnt_decoding.rnnt_timestamp_type="all"  # (default all, can be [all, char, word, segment])

  output_filename: Output filename where the transcriptions will be written
  batch_size: batch size during inference
  presort_manifest: sorts the provided manifest by audio length for faster inference (default: True)

  cuda: Optional int to enable or disable execution of model on certain CUDA device.
  allow_mps: Bool to allow using MPS (Apple Silicon M-series GPU) device if available
  amp: Bool to decide if Automatic Mixed Precision should be used during inference
  audio_type: Str filetype of the audio. Supported = wav, flac, mp3

  overwrite_transcripts: Bool which when set allows repeated transcriptions to overwrite previous results.

  ctc_decoding: Decoding sub-config for CTC. Refer to documentation for specific values.
  rnnt_decoding: Decoding sub-config for RNNT. Refer to documentation for specific values.

  calculate_wer: Bool to decide whether to calculate wer/cer at end of this script
  clean_groundtruth_text: Bool to clean groundtruth text
  langid: Str used for convert_num_to_words during groundtruth cleaning
  use_cer: Bool to use Character Error Rate (CER)  or Word Error Rate (WER)

  calculate_rtfx: Bool to calculate the RTFx throughput to transcribe the input dataset.

# Usage
ASR model can be specified by either "model_path" or "pretrained_name".
Data for transcription can be defined with either "audio_dir" or "dataset_manifest".
append_pred - optional. Allows you to add more than one prediction to an existing .json
pred_name_postfix - optional. The name you want to be written for the current model
Results are returned in a JSON manifest file.

python transcribe_speech.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    clean_groundtruth_text=True \
    langid='en' \
    batch_size=32 \
    timestamps=False \
    compute_langs=False \
    cuda=0 \
    amp=True \
    append_pred=False \
    pred_name_postfix="<remove or use another model name for output filename>"
"""


@dataclass
class ModelChangeConfig:
    """
    Sub-config for changes specific to the Conformer Encoder
    """

    conformer: ConformerChangeConfig = field(default_factory=ConformerChangeConfig)


@dataclass
class TranscriptionConfig:
    """
    Transcription Configuration for audio to text transcription.
    """

    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    channel_selector: Optional[Union[int, str]] = (
        None  # Used to select a single channel from multichannel audio, or use average across channels
    )
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    eval_config_yaml: Optional[str] = None  # Path to a yaml file of config of evaluation
    presort_manifest: bool = True  # Significant inference speedup on short-form data due to padding reduction

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Set to True to output greedy timestamp information (only supported models) and returns full alignment hypotheses
    timestamps: Optional[bool] = None

    # Set to True to return hypotheses instead of text from the transcribe function
    return_hypotheses: bool = False

    # Set to True to output language ID information
    compute_langs: bool = False

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    amp_dtype: str = "float16"  # can be set to "float16" or "bfloat16" when using amp
    compute_dtype: Optional[str] = (
        None  # "float32", "bfloat16" or "float16"; if None (default): bfloat16 if available else float32
    )
    matmul_precision: str = "high"  # Literal["highest", "high", "medium"]
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = field(default_factory=CTCDecodingConfig)

    # Decoding strategy for RNNT models
    # enable CUDA graphs for transcription
    rnnt_decoding: RNNTDecodingConfig = field(default_factory=lambda: RNNTDecodingConfig(fused_batch_size=-1))

    # Decoding strategy for AED models
    multitask_decoding: MultiTaskDecodingConfig = field(default_factory=MultiTaskDecodingConfig)
    # Prompt slots for prompted models, e.g. Canary-1B. Examples of acceptable prompt inputs:
    # Implicit single-turn assuming default role='user' (works with Canary-1B)
    #  +prompt.source_lang=en +prompt.target_lang=es +prompt.task=asr +prompt.pnc=yes
    # Explicit single-turn prompt:
    #  +prompt.role=user +prompt.slots.source_lang=en +prompt.slots.target_lang=es
    # +prompt.slots.task=s2t_translation +prompt.slots.pnc=yes
    # Explicit multi-turn prompt:
    #  +prompt.turns='[{role:user,slots:{source_lang:en,target_lang:es,task:asr,pnc:yes}}]'
    prompt: dict = field(default_factory=dict)

    # decoder type: ctc or rnnt, can be used to switch between CTC and RNNT decoder for Hybrid RNNT/CTC models
    decoder_type: Optional[str] = None
    # att_context_size can be set for cache-aware streaming models with multiple look-aheads
    att_context_size: Optional[list] = None

    # Use this for model-specific changes before transcription
    model_change: ModelChangeConfig = field(default_factory=ModelChangeConfig)

    # Config for word / character error rate calculation
    calculate_wer: bool = True
    clean_groundtruth_text: bool = False
    langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    use_cer: bool = False

    # can be set to True to return list of transcriptions instead of the config
    # if True, will also skip writing anything to the output file
    return_transcriptions: bool = False

    # key for groundtruth text in manifest
    gt_text_attr_name: str = "text"
    gt_lang_attr_name: str = "lang"

    extract_nbest: bool = False  # Extract n-best hypotheses from the model

    calculate_rtfx: bool = False
    warmup_steps: int = 0  # by default - no warmup
    run_steps: int = 1  # by default - single run


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> Union[TranscriptionConfig, List[Hypothesis]]:
    """
    Transcribes the input audio and can be used to infer with Encoder-Decoder models.
    """
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # Load augmentor from exteranl yaml file which contains eval info, could be extend to other feature such VAD, P&C
    augmentor = None
    if cfg.eval_config_yaml:
        eval_config = OmegaConf.load(cfg.eval_config_yaml)
        augmentor = eval_config.test_ds.get("augmentor")
        logging.info(f"Will apply on-the-fly augmentation on samples during transcription: {augmentor} ")

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.warning(
                "MPS device (Apple Silicon M-series GPU) support is experimental."
                " Env variable `PYTORCH_ENABLE_MPS_FALLBACK=1` should be set in most cases to avoid failures."
            )
            device = [0]
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    logging.info(f"Inference will be done on device: {map_location}")

    asr_model, model_name = setup_model(cfg, map_location)

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    if (cfg.compute_dtype is not None and cfg.compute_dtype != "float32") and cfg.amp:
        raise ValueError("amp=true is mutually exclusive with a compute_dtype other than float32")

    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16

    compute_dtype: torch.dtype
    if cfg.compute_dtype is None:
        can_use_bfloat16 = (not cfg.amp) and map_location.type == "cuda" and torch.cuda.is_bf16_supported()
        if can_use_bfloat16:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float32
    else:
        assert cfg.compute_dtype in {"float32", "bfloat16", "float16"}
        compute_dtype = getattr(torch, cfg.compute_dtype)

    asr_model.to(compute_dtype)

    # we will adjust this flag if the model does not support it
    compute_langs = cfg.compute_langs

    if cfg.timestamps:
        cfg.return_hypotheses = True

    # Check whether model and decoder type match
    if isinstance(asr_model, EncDecCTCModel):
        if cfg.decoder_type and cfg.decoder_type != 'ctc':
            raise ValueError('CTC model only support ctc decoding!')
    elif isinstance(asr_model, EncDecHybridRNNTCTCModel):
        if cfg.decoder_type and cfg.decoder_type not in ['ctc', 'rnnt']:
            raise ValueError('Hybrid model only support ctc or rnnt decoding!')
    elif isinstance(asr_model, EncDecRNNTModel):
        if cfg.decoder_type and cfg.decoder_type != 'rnnt':
            raise ValueError('RNNT model only support rnnt decoding!')

    if cfg.decoder_type and hasattr(asr_model.encoder, 'set_default_att_context_size'):
        asr_model.encoder.set_default_att_context_size(cfg.att_context_size)

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy') and hasattr(asr_model, 'decoding'):
        if isinstance(asr_model.decoding, MultiTaskDecoding):
            cfg.multitask_decoding.compute_langs = cfg.compute_langs
            if cfg.extract_nbest:
                cfg.multitask_decoding.beam.return_best_hypothesis = False
                cfg.return_hypotheses = True
            asr_model.change_decoding_strategy(cfg.multitask_decoding)
        elif cfg.decoder_type is not None:
            # TODO: Support compute_langs in CTC eventually
            if cfg.compute_langs and cfg.decoder_type == 'ctc':
                raise ValueError("CTC models do not support `compute_langs` at the moment")

            decoding_cfg = cfg.rnnt_decoding if cfg.decoder_type == 'rnnt' else cfg.ctc_decoding
            if cfg.extract_nbest:
                decoding_cfg.beam.return_best_hypothesis = False
                cfg.return_hypotheses = True
            if 'compute_langs' in decoding_cfg:
                decoding_cfg.compute_langs = cfg.compute_langs
            if hasattr(asr_model, 'cur_decoder'):
                asr_model.change_decoding_strategy(decoding_cfg, decoder_type=cfg.decoder_type)
            else:
                asr_model.change_decoding_strategy(decoding_cfg)

        # Check if ctc or rnnt model
        elif hasattr(asr_model, 'joint'):  # RNNT model
            if cfg.extract_nbest:
                cfg.rnnt_decoding.beam.return_best_hypothesis = False
                cfg.return_hypotheses = True
            cfg.rnnt_decoding.fused_batch_size = -1
            cfg.rnnt_decoding.compute_langs = cfg.compute_langs

            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        else:
            if cfg.compute_langs:
                raise ValueError("CTC models do not support `compute_langs` at the moment.")
            if cfg.extract_nbest:
                cfg.ctc_decoding.beam.return_best_hypothesis = False
                cfg.return_hypotheses = True

            asr_model.change_decoding_strategy(cfg.ctc_decoding)

    # Setup decoding config based on model type and decoder_type
    with open_dict(cfg):
        if isinstance(asr_model, EncDecCTCModel) or (
            isinstance(asr_model, EncDecHybridRNNTCTCModel) and cfg.decoder_type == "ctc"
        ):
            cfg.decoding = cfg.ctc_decoding
        elif isinstance(asr_model.decoding, MultiTaskDecoding):
            cfg.decoding = cfg.multitask_decoding
        else:
            cfg.decoding = cfg.rnnt_decoding

    filepaths, sorted_manifest_path = prepare_audio_data(cfg)

    remove_path_after_done = sorted_manifest_path if sorted_manifest_path is not None else None

    filepaths = sorted_manifest_path if sorted_manifest_path is not None else filepaths

    # Compute output filename
    cfg = compute_output_filename(cfg, model_name)

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.return_transcriptions and not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )
        return cfg

    # transcribe audio

    if cfg.calculate_rtfx:
        total_duration = 0.0

        with open(cfg.dataset_manifest, "rt") as fh:
            for line in fh:
                item = json.loads(line)
                if "duration" not in item:
                    raise ValueError(
                        f"Requested calculate_rtfx=True, but line {line} in manifest {cfg.dataset_manifest} \
                            lacks a 'duration' field."
                    )
                total_duration += item["duration"]

        if cfg.warmup_steps == 0:
            logging.warning(
                "RTFx measurement enabled, but warmup_steps=0. "
                "At least one warmup step is recommended to measure RTFx"
            )

    timer = SimpleTimer()
    model_measurements = []
    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=amp_dtype, enabled=cfg.amp):
        with torch.no_grad():
            override_cfg = asr_model.get_transcribe_config()
            override_cfg.batch_size = cfg.batch_size
            override_cfg.num_workers = cfg.num_workers
            override_cfg.return_hypotheses = cfg.return_hypotheses
            override_cfg.channel_selector = cfg.channel_selector
            override_cfg.augmentor = augmentor
            override_cfg.text_field = cfg.gt_text_attr_name
            override_cfg.lang_field = cfg.gt_lang_attr_name
            override_cfg.timestamps = cfg.timestamps
            if hasattr(override_cfg, "prompt"):
                override_cfg.prompt = parse_multitask_prompt(OmegaConf.to_container(cfg.prompt))

            device = next(asr_model.parameters()).device
            for run_step in range(cfg.warmup_steps + cfg.run_steps):
                if run_step < cfg.warmup_steps:
                    logging.info(f"Running warmup step {run_step}")
                # reset timer
                timer.reset()
                timer.start(device=device)
                # call transcribe
                transcriptions = asr_model.transcribe(
                    audio=filepaths,
                    override_config=override_cfg,
                    timestamps=cfg.timestamps,
                )
                # stop timer, log time
                timer.stop(device=device)
                logging.info(f"Model time for iteration {run_step}: {timer.total_sec():.3f}")
                if run_step >= cfg.warmup_steps:
                    model_measurements.append(timer.total_sec())

    model_measurements_np = np.asarray(model_measurements)
    logging.info(
        f"Model time avg: {model_measurements_np.mean():.3f}"
        + (f" (std: {model_measurements_np.std():.3f})" if cfg.run_steps > 1 else "")
    )

    if cfg.dataset_manifest is not None:
        logging.info(f"Finished transcribing from manifest file: {cfg.dataset_manifest}")
        if cfg.presort_manifest:
            transcriptions = restore_transcription_order(cfg.dataset_manifest, transcriptions)
    else:
        logging.info(f"Finished transcribing {len(filepaths)} files !")
    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

    # if transcriptions form a tuple of (best_hypotheses, all_hypotheses)
    if type(transcriptions) == tuple and len(transcriptions) == 2:
        if cfg.extract_nbest:
            # extract all hypotheses if exists
            transcriptions = transcriptions[1]
        else:
            # extract just best hypothesis
            transcriptions = transcriptions[0]

    if cfg.return_transcriptions:
        return transcriptions

    # write audio transcriptions
    output_filename, pred_text_attr_name = write_transcription(
        transcriptions,
        cfg,
        model_name,
        filepaths=filepaths,
        compute_langs=compute_langs,
        timestamps=cfg.timestamps,
    )
    logging.info(f"Finished writing predictions to {output_filename}!")

    # clean-up
    if cfg.presort_manifest is not None:
        if remove_path_after_done is not None:
            os.unlink(remove_path_after_done)

    if cfg.calculate_wer:
        output_manifest_w_wer, total_res, _ = cal_write_wer(
            pred_manifest=output_filename,
            gt_text_attr_name=cfg.gt_text_attr_name,
            pred_text_attr_name=pred_text_attr_name,
            clean_groundtruth_text=cfg.clean_groundtruth_text,
            langid=cfg.langid,
            use_cer=cfg.use_cer,
            output_filename=None,
        )
        if output_manifest_w_wer:
            logging.info(f"Writing prediction and error rate of each sample to {output_manifest_w_wer}!")
            logging.info(f"{total_res}")

    if cfg.calculate_rtfx:
        rtfx_measurements = total_duration / model_measurements_np
        logging.info(
            f"Model RTFx on the dataset: {rtfx_measurements.mean():.3f}"
            + (f" (std: {rtfx_measurements.std():.3f})" if cfg.run_steps > 1 else "")
        )

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter