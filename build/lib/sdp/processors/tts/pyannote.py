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

import random
import os
import logging
from time import time
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from whisperx.audio import SAMPLE_RATE
from whisperx.vad import load_vad_model, merge_chunks
import torch
import torchaudio

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest, save_manifest

def has_overlap(turn, overlaps):
    """Check if a given turn overlaps with any segment in the overlaps list.

    Args:
        turn: A segment representing a speech turn
        overlaps: List of overlap segments, sorted by start time

    Returns:
        bool: True if the turn overlaps with any segment, False otherwise
    """
    turn_overlaps = False
    for overlap in overlaps:
        if overlap.start > turn.end:
            # Overlap happens after turn, no need to keep looping since overlaps is sorted
            break
        elif overlap.start >= turn.start and overlap.start <= turn.end:
            # overlap starts during turn
            turn_overlaps = True
            break
        elif (overlap.end <= turn.end) and (overlap.end >= turn.start):
            # overlap ends during turn
            turn_overlaps = True
            break
    return turn_overlaps

class PyAnnoteDiarizationAndOverlapDetection(BaseProcessor):
    """This processor performs speaker diarization and overlap detection using PyAnnote.

    It processes audio files to identify different speakers and detect overlapping speech
    segments using PyAnnote's speaker diarization pipeline and VAD (Voice Activity Detection).
    The processor segments audio into speaker turns and identifies regions with overlapping speech.

    Args:
        hf_token (str): HuggingFace authentication token for accessing pretrained models
        segmentation_batch_size (int, Optional): Batch size for segmentation. Defaults to 128
        embedding_batch_size (int, Optional): Batch size for speaker embeddings. Defaults to 128
        min_length (float, Optional): Minimum length of segments in seconds. Defaults to 0.5
        max_length (float, Optional): Maximum length of segments in seconds. Defaults to 40
        device (str, Optional): Device to run the models on ('cuda' or 'cpu'). Defaults to "cuda"

    Returns:
        The same data as in the input manifest, but with speaker diarization and overlap
        detection information added to each segment.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.tts.pyannote.PyAnnoteDiarizationAndOverlapDetection
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_diarized.json
              hf_token: ${hf_token}
    """
    
    def __init__(self,
                 hf_token: str,
                 segmentation_batch_size: int = 128,
                 embedding_batch_size: int = 128,
                 min_length: float = 0.5,
                 max_length: float = 40,
                 device: str = "cuda",
                 **kwargs
        ):

        super().__init__(**kwargs)


        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=hf_token)
        self.pipeline.segmentation_batch_size = segmentation_batch_size
        self.pipeline.embedding_batch_size = embedding_batch_size

        if not torch.cuda.is_available():
            device = "cpu"
            logging.warning("CUDA is not available, using CPU")
        
        self.pipeline.to(torch.device(device))

        self.min_length = min_length
        self.max_length = max_length

        self.vad_onset = 0.5
        self.vad_offset = 0.363

        default_vad_options = {"vad_onset": self.vad_onset, "vad_offset": self.vad_offset}
        self.vad_model = load_vad_model(
            torch.device(device), use_auth_token=None, **default_vad_options
        )
        random.seed(42)
        
    
    def get_vad_segments(self, audio, merge_max_length=3):
        """Get voice activity detection segments for the given audio.

        Args:
            audio: Audio tensor to process
            merge_max_length (int, Optional): Maximum length for merging segments. Defaults to 3

        Returns:
            list: List of VAD segments with start and end times
        """
        vad_segments = self.vad_model(
                {
                    "waveform": audio,
                    "sample_rate": SAMPLE_RATE,
                }
            )
        
        vad_segments = merge_chunks(vad_segments, merge_max_length, onset=self.vad_onset)
        
        return vad_segments
    
    def add_vad_segments(self, audio, fs, start, end, segments, speaker_id):
        """Add VAD segments for a given audio region to the segments list.

        This method processes an audio region and adds VAD segments to the provided list.
        For segments longer than max_length, it splits them using VAD detection.
        For shorter segments, it adds them directly.

        Args:
            audio: Audio tensor
            fs (int): Sample rate
            start (float): Start time of the region
            end (float): End time of the region
            segments (list): List to append the segments to
            speaker_id (str): Identifier for the speaker
        """
        segment_duration = end - start
        if segment_duration > self.max_length:
            audio_seg = audio[: , int(start * fs): int(end * fs)]
            vad_segments = self.get_vad_segments(audio_seg)
            i = 0
            n = len(vad_segments)

            while i < n:
                random_duration = random.uniform(self.min_length, self.max_length)

                start_seg = vad_segments[i]['start']
                end_seg = vad_segments[i]['end']

                if end_seg - start_seg >= random_duration:
                    segment_data_entry = {}
                    segment_data_entry['speaker'] = speaker_id
                    segment_data_entry['start'] = start + start_seg
                    segment_data_entry['end'] = start + end_seg
                    segments.append(segment_data_entry)
                    i += 1  
                    continue

                # Merge segments until the random duration is reached
                while i < n and (vad_segments[i]['end'] - start_seg) < random_duration:
                    end_seg = vad_segments[i]['end']  # Extend the end time
                    i += 1
                    
                segment_data_entry = {}
                segment_data_entry['speaker'] = speaker_id
                segment_data_entry['start'] = start + start_seg
                segment_data_entry['end'] = start + end_seg
                segments.append(segment_data_entry)
        else:
            segment_data_entry = {}
            segment_data_entry['speaker'] = speaker_id
            segment_data_entry['start'] = start
            segment_data_entry['end'] = end
            segments.append(segment_data_entry)

    def process(self):
        """Process the input manifest file for speaker diarization and overlap detection.

        This method:
        1. Reads the input manifest
        2. Performs speaker diarization on each audio file
        3. Detects overlapping speech segments
        4. Splits long segments using VAD
        5. Identifies non-speaker regions
        6. Saves results to the output manifest

        The output includes:
        - Speaker segments with start/end times and speaker IDs
        - Overlap segments
        - Non-speaker segments
        """
        manifest = load_manifest(self.input_manifest_file)

        results = []
        start_time = time()

        for metadata in manifest:
            file_path = metadata['resampled_audio_filepath']
            logger.info(file_path)
            
            s, fs = torchaudio.load(file_path)
            with ProgressHook() as hook:
                diarization = self.pipeline({'waveform': s, 'sample_rate': fs}, hook=hook)
            overlaps = diarization.get_overlap().segments_list_

            # Due to a bug in PyAnnote-Audio, diarization might have timestamps longer than
            # the audio, so we crop it to the audio length
            # https://github.com/pyannote/pyannote-audio/issues/1611
            diarization.crop(0, len(s)/fs)

            # write in RTTM format
            logger.info("Writing {} turns to RTTM file".format(len(diarization._tracks)))
            rttm_filepath = os.path.splitext(file_path)[0] + ".rttm"
            with open(rttm_filepath, "w") as rttm_file:
                diarization.write_rttm(rttm_file)

            segments = []
            overlap_segments = []

            # iterate over segments
            for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
                if 'audio_item_id' in metadata:
                    speaker_id = metadata['audio_item_id'] + '_' + speaker
                elif 'speaker_id' in metadata:
                    speaker_id = metadata['speaker_id'] + '_' + speaker
                else:
                    raise ValueError('No speaker identifier in sample {}'.format(metadata['resampled_audio_filepath']))
                
                if has_overlap(speech_turn, overlaps):
                    segment_data_entry = {}
                    segment_data_entry['speaker'] = speaker_id
                    segment_data_entry['start'] = speech_turn.start
                    segment_data_entry['end'] = speech_turn.end
                    overlap_segments.append(segment_data_entry)
                else:
                    speech_duration = speech_turn.end - speech_turn.start
                    if speech_duration > self.min_length:
                        self.add_vad_segments(s, fs, speech_turn.start, speech_turn.end, segments, speaker_id)

            audio_duration = metadata['duration']

            ## split non speaker regions into max length required
            non_speaker_segments = []
            last_end_time = 0
            for seg in segments:
                start = seg['start']
                end = seg['end']
                if start > last_end_time:
                    non_speaker_segments.append((last_end_time, start))
                last_end_time = end

            # If there is any remaining audio after the last speaker segment
            if last_end_time < audio_duration:
                non_speaker_segments.append((last_end_time, audio_duration))
            
            for start, end in non_speaker_segments:
                speaker_id = "no-speaker"
                current_start = start
                while current_start < end:
                    current_end = min(current_start + self.max_length, end)
                    segment_data_entry = {}
                    segment_data_entry['speaker'] = speaker_id
                    segment_data_entry['start'] = current_start
                    segment_data_entry['end'] = current_end
                    segments.append(segment_data_entry)
                    current_start = current_end
                
            # Sort all segments by start time
            segments.sort(key=lambda x: x['start'])
            metadata['segments'] = segments
            metadata['overlap_segments'] = overlap_segments
            results.append(metadata)

        logger.info(f'Completed diarization in {(time()-start_time)/3600} hrs')
        save_manifest(results, self.output_manifest_file)

