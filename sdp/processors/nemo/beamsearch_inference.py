# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#

import contextlib
import Levenshtein
import json
import os
import re
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import editdistance
import numpy as np
import torch
from omegaconf import MISSING, OmegaConf
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.submodules import ctc_beam_decoding
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.utils.transcribe_utils import PunctuationCapitalization, TextProcessingConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging

from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor, DataEntry


def read_manifest(input_manifest_file, encoding):
    """Reading the input manifest file.

    .. note::
        This function should be overridden in the "initial" class creating
        manifest to read from the original source of data.
    """
    if input_manifest_file is None:
        raise NotImplementedError("Override this method if the processor creates initial manifest")

    with open(input_manifest_file, "rt", encoding=encoding) as fin:
        for line in fin:
            yield json.loads(line)
            
@dataclass
class EvalBeamSearchNGramConfig:
    """
    Evaluate an ASR model with beam search decoding and n-gram KenLM language model.
    """
    # # The path of the '.nemo' file of the ASR model or the name of a pretrained model (ngc / huggingface)
    model_path: str = MISSING

    # File paths
    dataset_manifest: str = MISSING  # The manifest file of the evaluation set
    preds_output_folder: Optional[str] = None  # The optional folder where the predictions are stored
    cache_file: Optional[str] = None  # The cache file for storing the logprobs of the model

    # Parameters for inference
    batch_size: int = 16  # The batch size to calculate log probabilities
    beam_batch_size: int = 1  # The batch size to be used for beam search decoding
    
    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # Beam Search hyperparameters
    ctc_decoding: CTCDecodingConfig = field(default_factory=lambda: CTCDecodingConfig(
        strategy="flashlight", # gready, beam = pyctcdecode, flashlight
        beam = ctc_beam_decoding.BeamCTCInferConfig(
            nemo_kenlm_path="/mnt/md1/YTDS/ES/lm/lm.kenlm",
            beam_size=16,
            beam_alpha=0.5, # LM weight
            beam_beta=0.5, # length weight
            return_best_hypothesis = False,
            flashlight_cfg=ctc_beam_decoding.FlashlightConfig(
                lexicon_path = "/mnt/md1/YTDS/ES/lm/lm.flashlight_lexicon"),
            pyctcdecode_cfg=ctc_beam_decoding.PyCTCDecodeConfig(),
            ),
        ))
    
    text_processing: Optional[TextProcessingConfig] = field(default_factory=lambda: TextProcessingConfig(
        punctuation_marks = ".,?",
        separate_punctuation = False,
        do_lowercase = False,
        rm_punctuation = False,
    ))


class BeamsearchTopNInference(BaseProcessor):
    """Adds predictions of a text-based punctuation and capitalization (P&C) model.

    Operates on the text in the ``input_text_field``, and saves predictions in
    the ``output_text_field``.

    Args:
        input_audio_key (str): the text field that will be the input to the P&C model.
        output_text_key (str): the text field where the output of the PC model
            will be saved.
        batch_size (int): the batch sized used by the P&C model.
        device (str): the device used by the P&C model. Can be skipped to auto-select.
        pretrained_name (str): the pretrained_name of the P&C model.
        model_path (str): the model path to the P&C model.

    .. note::
        Either ``pretrained_name`` or ``model_path`` have to be specified.

    Returns:
         The same data as in the input manifest with an additional field
         <output_text_field> containing P&C model's predictions.
    """

    def __init__(
        self,
        input_audio_key: str,
        output_text_key: str,
        batch_size: int,
        device: Optional[str] = None,
        pretrained_name: Optional[str] = None,
        model_path: Optional[str] = None,
        in_memory_chunksize: int = 100000,
        cfg: Optional[EvalBeamSearchNGramConfig] = EvalBeamSearchNGramConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pretrained_name = pretrained_name
        self.model_path = model_path
        self.input_audio_key = input_audio_key
        self.output_text_key = output_text_key
        self.device = device
        self.batch_size = batch_size
        self.in_memory_chunksize=in_memory_chunksize
        self.cfg=cfg

        # verify self.pretrained_name/model_path
        if self.pretrained_name is None and self.model_path is None:
            raise ValueError("pretrained_name and model_path cannot both be None")
        if self.pretrained_name is not None and self.model_path is not None:
            raise ValueError("pretrained_name and model_path cannot both be specified")
    
    def _chunk_manifest(self):
        """Splits the manifest into smaller chunks defined by ``in_memory_chunksize``.
        """
        manifest_chunk = []
        for idx, data_entry in enumerate(read_manifest(self.input_manifest_file, encoding="utf8"), 1):
            manifest_chunk.append(data_entry)
            if idx % self.in_memory_chunksize == 0:
                yield manifest_chunk
                manifest_chunk = []
        if len(manifest_chunk) > 0:
            yield manifest_chunk

    def process(self):
        if self.pretrained_name:
            model = EncDecHybridRNNTCTCModel.from_pretrained(self.pretrained_name)
        else:
            model = EncDecHybridRNNTCTCModel.restore_from(self.model_path)

        if self.device is None:
            if torch.cuda.is_available():
                model = model.cuda()
            else:
                model = model.cpu()
        else:
            model = model.to(self.device)

        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)
        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:

            for manifest in self._chunk_manifest():

                audio_file_paths = [x[self.input_audio_key] for x in manifest]


                if isinstance(model, EncDecHybridRNNTCTCModel):
                    model.change_decoding_strategy(decoding_cfg=None, decoder_type="ctc")
                else:
                    model.change_decoding_strategy(None)

                # Override the beam search config with current search candidate configuration
                model.cfg.decoding = CTCDecodingConfig(
                    strategy=self.cfg.ctc_decoding.strategy,
                    preserve_alignments=self.cfg.ctc_decoding.preserve_alignments,
                    compute_timestamps=self.cfg.ctc_decoding.compute_timestamps,
                    word_seperator=self.cfg.ctc_decoding.word_seperator,
                    ctc_timestamp_type=self.cfg.ctc_decoding.ctc_timestamp_type,
                    batch_dim_index=self.cfg.ctc_decoding.batch_dim_index,
                    greedy=self.cfg.ctc_decoding.greedy,
                    confidence_cfg=self.cfg.ctc_decoding.confidence_cfg,
                    temperature=self.cfg.ctc_decoding.temperature,
                    beam = ctc_beam_decoding.BeamCTCInferConfig(beam_size=self.cfg.ctc_decoding.beam.beam_size,
                                                                beam_alpha=self.cfg.ctc_decoding.beam.beam_alpha,
                                                                beam_beta=self.cfg.ctc_decoding.beam.beam_beta,
                                                                word_kenlm_path=self.cfg.ctc_decoding.beam.word_kenlm_path,
                                                                nemo_kenlm_path=self.cfg.ctc_decoding.beam.nemo_kenlm_path,
                                                                preserve_alignments=self.cfg.ctc_decoding.beam.preserve_alignments,
                                                                compute_timestamps=self.cfg.ctc_decoding.beam.compute_timestamps,
                                                                flashlight_cfg=self.cfg.ctc_decoding.beam.flashlight_cfg,
                                                                pyctcdecode_cfg=self.cfg.ctc_decoding.beam.pyctcdecode_cfg,
                                                                return_best_hypothesis=self.cfg.ctc_decoding.beam.return_best_hypothesis),
                    )
                # Update model's decoding strategy
                if isinstance(model, EncDecHybridRNNTCTCModel):
                    model.change_decoding_strategy(model.cfg.decoding, decoder_type='ctc')
                else:
                    model.change_decoding_strategy(model.cfg.decoding)


                with torch.no_grad():
                    if isinstance(model, EncDecHybridRNNTCTCModel):
                        model.cur_decoder = 'ctc'

                    override_cfg = model.get_transcribe_config()
                    override_cfg.batch_size = self.batch_size
                    override_cfg.return_hypotheses = True

                    all_hypotheses = model.transcribe(audio_file_paths, override_config=override_cfg)
                    if type(all_hypotheses) == tuple and len(all_hypotheses) == 2: # if transcriptions form a tuple of (best_hypotheses, all_hypotheses)
                        all_hypotheses = all_hypotheses[1]

                pred_texts = [] 
                for hypotheses in all_hypotheses:
                    pred_text = [hyp.text for hyp in hypotheses]
                    pred_texts.append(pred_text)


                for item, t in zip(manifest, pred_texts):
                    item[self.output_text_key] = t
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')

class RestorePCbyTopN(BaseParallelProcessor):
    """
    Adds predictions of a audio-based punctuation and capitalization (P&C) model.

    Args:
        text_without_pc_key (str): Key to get path to wav file.
        texts_with_pc_key (str): Key to put to audio duration.
        output_text_key (str): Key to put to audio duration.
    Returns:
        All the same fields as in the input manifest plus duration_field
    """

    def __init__(
        self,
        text_without_pc_key: str,
        texts_with_pc_key: str,
        output_text_key: str,
        punctuation: str,
        do_lower: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_without_pc_key = text_without_pc_key
        self.texts_with_pc_key = texts_with_pc_key
        self.output_text_key = output_text_key
        self.punctuation = punctuation
        self.do_lower = do_lower
    
    def prepare(self):
        if self.punctuation:
            self.patterns = re.compile("["+self.punctuation+"]")

    def get_capitalisation_from_target(self, text_input, text_to_fix):
        text_input = text_input.strip()
        text_to_fix = text_to_fix.strip()
        if text_input[0].isupper():
            text_to_fix = text_to_fix[0].upper()+text_to_fix[1:]

        return text_to_fix
        

    def process_dataset_entry(self, data_entry):
        text_without_pc = data_entry[self.text_without_pc_key]
        texts_with_pc = data_entry[self.texts_with_pc_key]
        texts = []
        ldists = []
        for text in texts_with_pc:
            if self.do_lower:
                text = text.lower()
            if self.punctuation:
                text = self.patterns.sub('', text)
            ldist = Levenshtein.distance(text, text_without_pc)
            if ldist == 0:
                data_entry[self.output_text_key] = text
                return [DataEntry(data=data_entry)]
                
            ldists.append(ldist)
            texts.append(text)

        text_with_pc = self.get_capitalisation_from_target(text_without_pc, texts_with_pc[np.argmin(ldists)])
        data_entry[self.output_text_key] = text_with_pc
        return [DataEntry(data=data_entry)]
    
class ConcatManifests(BaseProcessor):
    """Adds predictions of a text-based punctuation and capitalization (P&C) model.

    Operates on the text in the ``input_text_field``, and saves predictions in
    the ``output_text_field``.

    Args:
        input_audio_key (str): the text field that will be the input to the P&C model.

    .. note::
        Either ``pretrained_name`` or ``model_path`` have to be specified.

    Returns:
         The same data as in the input manifest with an additional field
         <output_text_field> containing P&C model's predictions.
    """

    def __init__(
        self,
        input_manifest_files: List[str],
        encoding: str = "utf8",
        ensure_ascii: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_manifest_files = input_manifest_files
        self.encoding = encoding
        self.ensure_ascii = ensure_ascii

    def process(self):
        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)
        with open(self.output_manifest_file, "wt", encoding=self.encoding) as fout:
            for input_manifest_file in self.input_manifest_files:
                for idx, data_entry in enumerate(read_manifest(input_manifest_file, self.encoding)):
                    fout.write(json.dumps(data_entry, ensure_ascii=self.ensure_ascii) + '\n')
