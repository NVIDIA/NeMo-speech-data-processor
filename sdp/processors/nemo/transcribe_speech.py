# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import glob
import json
import os
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.utils.transcribe_utils import transcribe_partial_audio
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from omegaconf import OmegaConf

"""
Transcribe audio file on a single CPU/GPU. Useful for transcription of moderate amounts of audio data.

# Arguments
  model_path: path to .nemo ASR checkpoint
  pretrained_name: name of pretrained ASR model (from NGC registry)
  audio_dir: path to directory with audio files
  dataset_manifest: path to dataset JSON manifest file (in NeMo format)

  compute_langs: Bool to request language ID information (if the model supports it)

  output_filename: Output filename where the transcriptions will be written
  batch_size: batch size during inference

  cuda: Optional int to enable or disable execution of model on certain CUDA device.
  amp: Bool to decide if Automatic Mixed Precision should be used during inference
  audio_type: Str filetype of the audio. Supported = wav, flac, mp3

  overwrite_transcripts: Bool which when set allowes repeated transcriptions to overwrite previous results.

  rnnt_decoding: Decoding sub-config for RNNT. Refer to documentation for specific values.

# Usage
ASR model can be specified by either "model_path" or "pretrained_name".
Data for transcription can be defined with either "audio_dir" or "dataset_manifest".
append_pred - optional. Allows you to add more than one prediction to an existing .json
pred_name_postfix - optional. The name you want to be written for the current model
Results are returned in a JSON manifest file.

python transcribe_speech.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="" \
    dataset_manifest="" \
    output_filename="" \
    batch_size=32 \
    compute_langs=False \
    cuda=0 \
    amp=True \
    append_pred=False \
    pred_name_postfix=""
"""


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.

    # Set to True to output language ID information
    compute_langs: bool = False

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()

    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig(fused_batch_size=-1)


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> TranscriptionConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
        else:
            device = 1
            accelerator = 'cpu'
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'

    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')

    # setup model
    if cfg.model_path is not None:
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        asr_model = imported_class.restore_from(
            restore_path=cfg.model_path, map_location=map_location
        )  # type: ASRModel
        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(
            model_name=cfg.pretrained_name, map_location=map_location
        )  # type: ASRModel
        model_name = cfg.pretrained_name

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()
    partial_audio = False

    # collect additional transcription information
    return_hypotheses = True

    # we will adjust this flag is the model does not support it
    compute_langs = cfg.compute_langs

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        # Check if ctc or rnnt model
        if hasattr(asr_model, 'joint'):  # RNNT model
            cfg.rnnt_decoding.fused_batch_size = -1
            cfg.rnnt_decoding.compute_langs = cfg.compute_langs
            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        else:
            asr_model.change_decoding_strategy(cfg.ctc_decoding)

    if cfg.audio_dir is not None and not cfg.append_pred:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
    else:
        # get filenames from manifest
        filepaths = []
        if os.stat(cfg.dataset_manifest).st_size == 0:
            logging.error(f"The input dataset_manifest {cfg.dataset_manifest} is empty. Exiting!")
            return None

        manifest_dir = Path(cfg.dataset_manifest).parent
        with open(cfg.dataset_manifest, 'r') as f:
            has_two_fields = []
            for line in f:
                item = json.loads(line)
                if "offset" in item and "duration" in item:
                    has_two_fields.append(True)
                else:
                    has_two_fields.append(False)
                audio_file = Path(item['audio_filepath'])
                if not audio_file.is_file() and not audio_file.is_absolute():
                    audio_file = manifest_dir / audio_file
                filepaths.append(str(audio_file.absolute()))
        partial_audio = all(has_two_fields)

    logging.info(f"\nTranscribing {len(filepaths)} files...\n")

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # Compute output filename
    if cfg.output_filename is None:
        # create default output filename
        if cfg.audio_dir is not None:
            cfg.output_filename = os.path.dirname(os.path.join(cfg.audio_dir, '.')) + '.json'
        elif cfg.pred_name_postfix is not None:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{cfg.pred_name_postfix}.json')
        else:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{model_name}.json')

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )

        return cfg

    # transcribe audio

    with autocast():
        with torch.no_grad():
            if partial_audio:
                if isinstance(asr_model, EncDecCTCModel):
                    transcriptions = transcribe_partial_audio(
                        asr_model=asr_model,
                        path2manifest=cfg.dataset_manifest,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        return_hypotheses=return_hypotheses,
                    )
                else:
                    logging.warning(
                        "RNNT models do not support transcribe partial audio for now. Transcribing full audio."
                    )
                    transcriptions = asr_model.transcribe(
                        paths2audio_files=filepaths,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        return_hypotheses=return_hypotheses,
                    )
            else:
                transcriptions = asr_model.transcribe(
                    paths2audio_files=filepaths,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    return_hypotheses=return_hypotheses,
                )

    logging.info(f"Finished transcribing {len(filepaths)} files !")

    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

    # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
    if type(transcriptions) == tuple and len(transcriptions) == 2:
        transcriptions = transcriptions[0]

    # write audio transcriptions

    if cfg.append_pred:
        logging.info(f'Transcripts will be written in "{cfg.output_filename}" file')
        if cfg.pred_name_postfix is not None:
            pred_by_model_name = cfg.pred_name_postfix
        else:
            pred_by_model_name = model_name
        pred_text_attr_name = 'pred_text_' + pred_by_model_name
    else:
        pred_text_attr_name = 'pred_text'
    with open(cfg.output_filename, 'w', encoding='utf-8') as f:
        if cfg.audio_dir is not None:
            for idx, transcription in enumerate(transcriptions):
                item = {'audio_filepath': filepaths[idx], pred_text_attr_name: transcription.text}
                if compute_langs:
                    item['pred_lang'] = transcription.langs
                    item['pred_lang_chars'] = transcription.langs_chars
                f.write(json.dumps(item) + "\n")
        else:
            with open(cfg.dataset_manifest, 'r') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    item[pred_text_attr_name] = transcriptions[idx].text

                    if compute_langs:
                        item['pred_lang'] = transcriptions[idx].langs
                        item['pred_lang_chars'] = transcriptions[idx].langs_chars
                    f.write(json.dumps(item) + "\n")

    logging.info("Finished writing predictions !")
    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
