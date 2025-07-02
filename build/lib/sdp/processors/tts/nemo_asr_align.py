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

import omegaconf
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest, save_manifest

class NeMoASRAligner(BaseProcessor):
    """This processor aligns text and audio using NeMo ASR models.

    It uses a pre-trained ASR model to transcribe audio files and generate word-level
    alignments with timestamps. The processor supports both CTC and RNNT decoders and
    can process either full audio files or just specific segments.

    Args:
        model_name (str): Name of pretrained model to use. Defaults to "nvidia/parakeet-tdt_ctc-1.1b"
        model_path (str, Optional): Path to local model file. If provided, overrides model_name
        min_len (float): Minimum length of audio segments to process in seconds. Defaults to 0.1
        max_len (float): Maximum length of audio segments to process in seconds. Defaults to 40
        parakeet (bool): Whether model is a Parakeet model. Affects time stride calculation. Defaults to True
        ctc (bool): Whether to use CTC decoding. Defaults to False
        batch_size (int): Batch size for processing. Defaults to 32
        num_workers (int): Number of workers for data loading. Defaults to 10
        split_batch_size (int): Maximum size for splitting large batches. Defaults to 5000
        timestamp_type (str): Type of timestamp to generate ("word" or "char"). Defaults to "word"
        infer_segment_only (bool): Whether to process only segments instead of full audio. Defaults to False
        device (str): Device to run the model on. Defaults to "cuda"

    Returns:
        The same data as in the input manifest, but with word-level alignments added
        to each segment.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.tts.nemo_asr_align.NeMoASRAligner
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_aligned.json
              parakeet: True
    """
    def __init__(self,
            model_name="nvidia/parakeet-tdt_ctc-1.1b",
            model_path=None,
            min_len: float = 0.1,
            max_len: float = 40,
            parakeet: bool = True,
            ctc: bool = False,
            batch_size: int = 32,
            num_workers: int = 10,
            split_batch_size: int = 5000,
            timestamp_type: str = "word",
            infer_segment_only: bool = False,
            device: str = "cuda",
            **kwargs):
        super().__init__(**kwargs)
        if model_path is not None:
            self.asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)
        else:
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        
        if not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA is not available, using CPU")
        
        self.asr_model.to(device)
        # Configuring attention to work with longer files
        self.asr_model.change_attention_model(
            self_attention_model="rel_pos_local_attn", att_context_size=[128, 128]
        )
        self.asr_model.change_subsampling_conv_chunking_factor(1)
        self.min_len = min_len
        self.max_len = max_len
        self.parakeet = parakeet # if model type is parakeet or not, determines time stride
        self.ctc = ctc # if decoder type is ctc or not, determines timestamp substraction
        self.timestamp_type = timestamp_type
        self.infer_segment_only = infer_segment_only

        cfg =  self.asr_model.cfg.decoding
        with omegaconf.open_dict(cfg):
            cfg['compute_timestamps']=True
            cfg['preserve_alignments']=True
            if ctc:
                cfg.strategy = "greedy_batch"
            else:
                cfg['rnnt_timestamp_type'] = self.timestamp_type 
        self.asr_model.change_decoding_strategy(decoding_cfg=cfg)

        # set batch size
        self.override_cfg = self.asr_model.get_transcribe_config()
        self.override_cfg.batch_size = batch_size
        self.split_batch_size = split_batch_size
        self.override_cfg.num_workers = num_workers
        self.override_cfg.return_hypotheses = True
        self.override_cfg.timestamps = True

    def get_alignments_text(self, hypotheses):
        """Extract word alignments and text from model hypotheses.

        Args:
            hypotheses: The hypothesis object containing timesteps and text predictions.

        Returns:
            tuple: A tuple containing:
                - list: List of dictionaries with word alignments (word, start, end)
                - str: The transcribed text
        """
        timestamp_dict = hypotheses.timestep # extract timesteps from hypothesis of first (and only) audio file

        # For a FastConformer model, you can display the word timestamps as follows:
        # 80ms is duration of a timestep at output of the Conformer
        if self.parakeet:
            time_stride = 8 * self.asr_model.cfg.preprocessor.window_stride
        else:
            time_stride = 4 * self.asr_model.cfg.preprocessor.window_stride

        word_timestamps = timestamp_dict[self.timestamp_type]

        alignments = []
        for stamp in word_timestamps:
            if self.ctc:
                start = stamp['start_offset'] * time_stride
                end = stamp['end_offset'] * time_stride 
            else: # if rnnt or tdt decoder
                start = max(0, stamp['start_offset'] * time_stride - 0.08)
                end = max(0, stamp['end_offset'] * time_stride - 0.08)

            word = stamp['char'] if 'char' in stamp else stamp['word']
            alignments.append({'word': word, 'start': round(start, 3), 'end': round(end, 3)})

        text = hypotheses.text
        text = text.replace("⁇", "")
        return alignments, text
    
    
    def _prepare_metadata_batch(self, metadata_batch):
        """Prepare audio data and segment mapping for a batch of metadata files.

        Args:
            metadata_batch (list): List of metadata dictionaries containing audio information.

        Returns:
            tuple: A tuple containing:
                - list: List of audio segments
                - list: List of tuples mapping segments to their original metadata (metadata_idx, segment_idx)
        """
        all_segments = []
        segment_indices = []

        for metadata_idx, metadata in enumerate(metadata_batch):
            audio, sr = torchaudio.load(metadata['resampled_audio_filepath'])

            for segment_idx, segment in enumerate(metadata['segments']):
                duration = segment['end'] - segment['start']
                if duration >= self.min_len and segment['speaker']!='no-speaker':
                    start = int(segment['start'] * sr)
                    end = int(segment['end'] * sr)
                    audio_segment = audio[:, start:end].squeeze(0)
                    if len(audio_segment) > 0:
                        all_segments.append(audio_segment)
                        segment_indices.append((metadata_idx, segment_idx))

        return all_segments, segment_indices


    def process(self):
        """Process the input manifest file to generate word alignments and transcriptions.

        This method reads the input manifest, processes audio files either in full or by segments,
        generates transcriptions and word alignments using the ASR model, and saves the results
        to the output manifest file.

        The processing can be done in two modes:
        1. Full audio processing (infer_segment_only=False)
        2. Segment-only processing (infer_segment_only=True)

        Results are saved in JSONL format with alignments and transcriptions added to the original metadata.
        """
        manifest = load_manifest(self.input_manifest_file)
        
        results = []
        if not self.infer_segment_only:
            transcribe_manifest = []
            for data in manifest:
                if  (('split_filepaths' in data and data['split_filepaths'] is None) or ('split_filepaths' not in data)) and data['duration'] > self.min_len:
                    transcribe_manifest.append(data)
                else:
                    data['text'] = ''
                    data['alignment'] = []
                    results.append(data)


            files = [x['resampled_audio_filepath'] for x in transcribe_manifest]

            for i in range(0, len(files), self.split_batch_size):
                batch = files[i:i + self.split_batch_size]
                with torch.no_grad():
                    hypotheses_list = self.asr_model.transcribe(batch, override_config=self.override_cfg)
                # if hypotheses form a tuple (from RNNT), extract just "best" hypotheses
                if type(hypotheses_list) == tuple and len(hypotheses_list) == 2:
                    hypotheses_list = hypotheses_list[0]
                    
                metadatas = transcribe_manifest[i:i + self.split_batch_size]
                for idx, metadata in enumerate(metadatas):
                    hypotheses =  hypotheses_list[idx]
                    alignments, text = self.get_alignments_text(hypotheses)
                    metadata['text'] = text
                    metadata['alignment']= alignments
                    results.append(metadata)
        else:
            for i in range(0, len(manifest), self.split_batch_size):
                metadata_batch = manifest[i:i + self.split_batch_size]
                all_segments, segment_indices = self._prepare_metadata_batch(metadata_batch)
                
                try:
                    with torch.no_grad():
                        hypotheses_list = self.asr_model.transcribe(all_segments, override_config=self.override_cfg)
                except Exception as e:
                    files_list = [ item['resampled_audio_filepath'] for item in metadata_batch ]
                    raise ValueError(f"Exception occurred for audio filepath list: {files_list}, Error is : {str(e)}")

                
                if type(hypotheses_list) == tuple and len(hypotheses_list) == 2:
                    hypotheses_list = hypotheses_list[0]

                for (metadata_idx, segment_idx), hypotheses in zip(segment_indices, hypotheses_list):
                    alignments, text = self.get_alignments_text(hypotheses)
                    segment = metadata_batch[metadata_idx]['segments'][segment_idx]
                    segment['text'] = text
                    for word in alignments:
                        word['start'] = round(word['start'] + segment['start'], 3)
                        word['end'] = round(word['end'] + segment['start'], 3)
                    
                    segment['words']= alignments

                results.extend(metadata_batch)

        save_manifest(results, self.output_manifest_file)
