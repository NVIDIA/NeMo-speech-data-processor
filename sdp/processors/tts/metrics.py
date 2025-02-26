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

import librosa
import math
import ndjson
import numpy as np
from tqdm import tqdm
from sdp.processors.base_processor import BaseProcessor

import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.pipelines import SQUIM_OBJECTIVE


class TorchSquimObjectiveQualityMetricsProcessor(BaseProcessor):
    """A processor class for calculating Squim objective quality metrics.
    
    This class uses a pre-trained Squim model to calculate the objective quality metrics
    of audio files. It loads the model, processes each audio file in the manifest, and
    updates the manifest with the calculated metrics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SQUIM_OBJECTIVE.get_model().cuda()

    def process(self):
        with open(self.input_manifest_file) as f:
            manifest = ndjson.load(f)

        results = []

        for metadata in tqdm(manifest):
            info = torchaudio.info(metadata['resampled_audio_filepath'])
            sr = info.sample_rate

            try: 
                audio, _ = librosa.load(path=metadata['resampled_audio_filepath'], sr=sr)
            except Exception as ex:
                print(f"Failed to load {metadata['resampled_audio_filepath']}, exception={ex}")
                continue

            for segment in metadata["segments"]:
                if ("text" in segment and segment["text"].strip() == "") or (segment["speaker"]=="no-speaker"):
                    continue
                start = segment["start"]
                end = segment["end"]

                start = math.floor(start * sr)
                end = math.floor(end * sr)
                num_samples = end - start

                y = audio[start: end]
                y = torch.from_numpy(y)
                y = torch.unsqueeze(y, dim=0) # needed otherwise throws input size error

                if sr != 16000:
                    y = F.resample(y, sr, 16000)

                try:
                    with torch.no_grad():
                        y_cuda = y.cuda()
                        stoi_hyp, pesq_hyp, si_sdr_hyp = self.model(y_cuda)

                    if 'metrics' in segment:
                        metrics = segment['metrics']
                    else:
                        metrics = {}

                    pesq = pesq_hyp.item()
                    stoi = stoi_hyp.item()
                    si_sdr = si_sdr_hyp.item()

                    metrics['pesq_squim'] = round(pesq, 3)
                    metrics['stoi_squim'] = round(stoi, 3)
                    metrics['sisdr_squim'] = round(si_sdr, 3)
                    segment['metrics'] = metrics
                except Exception as e:
                    torch.cuda.empty_cache()
                    print('Failed to extract Squim metrics {} with frame_offset={} and num_frames={}'.format(
                        metadata['resampled_audio_filepath'],
                        start,
                        num_samples))
                    continue
            results.append(metadata)

        with open(self.output_manifest_file, 'w') as f:
            ndjson.dump(results, f)


class BandwidthEstimationProcessor(BaseProcessor):
    """A processor class for estimating audio bandwidth.
    
    This class analyzes audio files to estimate their effective bandwidth by examining
    the power spectrum and determining the highest frequency with significant energy
    content above a threshold.

    Args:
        n_fft (int, optional): Size of FFT window. Defaults to 512.
        stride_seconds (float, optional): Time between successive FFT windows in seconds. Defaults to 0.01.
        top_db (float, optional): Maximum decibel value for power spectrum normalization. Defaults to 100.0.
        frequency_threshold (float, optional): Threshold in dB below peak for bandwidth estimation. Defaults to -50.0.
    """
    def __init__(
        self,
        n_fft: int = 512,
        stride_seconds: float = 0.01,
        top_db: float = 100.0,
        frequency_threshold: float = -50.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_fft = n_fft
        self.stride_seconds = stride_seconds
        self.top_db = top_db
        self.frequency_threshold = frequency_threshold

    def _estimate_bandwidth(self, audio, sample_rate):
        """Estimates the bandwidth of an audio signal.
        
        This method calculates the power spectrogram of the audio signal and determines
        the bandwidth based on a frequency threshold.
        
        Args:
            audio (np.ndarray): The audio signal to estimate the bandwidth of
            sample_rate (int): The sample rate of the audio signal
            
        Returns:
            int: The estimated bandwidth of the audio signal
        """
        hop_length = int(sample_rate * self.stride_seconds)
        # calculate power spectrogram
        # use Blackman-Harris window to significantly reduce spectral leakage (level of sidelobes)
        spec = librosa.stft(y=audio, n_fft=self.n_fft, hop_length=hop_length, window="blackmanharris")
        power_spec = np.abs(spec) ** 2
        power_spec = np.mean(power_spec, axis=1)
        power_spec = librosa.power_to_db(power_spec, ref=self.n_fft, top_db=self.top_db)

        bandwidth = 0
        peak = np.max(power_spec)
        freq_width = sample_rate / self.n_fft
        for idx in range(len(power_spec) - 1, -1, -1):
            if power_spec[idx] - peak > self.frequency_threshold:
                bandwidth = idx * freq_width
                break

        return bandwidth

    def process(self):
        """Process the audio files in the manifest to estimate bandwidth.
        
        Reads the input manifest, processes each audio segment to estimate its bandwidth,
        and writes the results to the output manifest with bandwidth measurements added
        to the metrics field of each segment.
        """
        with open(self.input_manifest_file) as f:
            manifest = ndjson.load(f)

        results = []

        for metadata in tqdm(manifest):
            audio_filepath = metadata['audio_filepath']
            try: 
                audio, sample_rate = librosa.load(path=audio_filepath, sr=None)
            except Exception as ex:
                print(f"Failed to load {audio_filepath}, exception={ex}")
                continue

            for segment in metadata['segments']:
                if ("text" in segment and segment["text"].strip() == "") or (segment["speaker"]=="no-speaker"):
                    continue
                start = segment['start']
                end = segment['end']

                audio_segment = audio[int(start*sample_rate): int(end*sample_rate)]

                bandwidth = self._estimate_bandwidth(audio=audio_segment, sample_rate=sample_rate)

                if 'metrics' in segment:
                    metrics = segment['metrics']
                else:
                    metrics = {}

                metrics['bandwidth'] = int(bandwidth)
                segment['metrics'] = metrics

            results.append(metadata)

        with open(self.output_manifest_file, 'w') as f:
            ndjson.dump(results, f)


class HallucinationDetectionProcessor(BaseProcessor):
    """A processor class for detecting potential hallucinations in transcribed text.
    
    This class analyzes text segments to detect repeated word patterns that might
    indicate hallucinated content in speech-to-text output.

    Args:
        max_sequence_length (int, optional): Maximum length of word sequences to check for repetition. Defaults to 5.
        threshold (int, optional): Number of repetitions required to flag as hallucination. Defaults to 6.
    """
    def __init__(
        self,
        max_sequence_length: int = 5,
        threshold: int = 6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
        self.threshold = threshold
    
    def detect_hallucination(self, text: str):
        """Detect potential hallucinations in text by identifying repeated word sequences.
        
        Args:
            text (str): The input text to analyze for hallucinations.
            
        Returns:
            bool: True if hallucination is detected, False otherwise.
        """
        words = text.lower().replace(',', '').replace('.', '').replace('?', '').split()

        for seq_len in range(1, self.max_sequence_length + 1):
            counts = {}
            for i in range(len(words) - seq_len + 1):
                sequence = tuple(words[i:i + seq_len])

                if i + seq_len < len(words) and words[i:i + seq_len] == words[i + seq_len:i + 2 * seq_len]:
                    counts[sequence] = counts.get(sequence, 1) + 1
                    if counts[sequence] >= self.threshold:
                        return True
                else:
                    counts.pop(sequence, None)

        return False


    def process(self):
        """Process the text segments in the manifest to detect hallucinations.
        
        Reads the input manifest, analyzes each text segment for hallucinations,
        and writes the results to the output manifest with hallucination detection
        results added to the metrics field of each segment.
        """
        with open(self.input_manifest_file) as f:
            manifest = ndjson.load(f)
        
        results = []
        for metadata in tqdm(manifest):
            for segment in metadata['segments']:

                if ("text" in segment and segment["text"].strip() == "") or (segment["speaker"]=="no-speaker") or (not "text" in segment):
                    continue

                detected = self.detect_hallucination(segment["text"])

                if 'metrics' in segment:
                    metrics = segment['metrics']
                else:
                    metrics = {}

                metrics['hallucination'] = detected
                segment['metrics'] = metrics

            results.append(metadata)

        with open(self.output_manifest_file, 'w') as f:
            ndjson.dump(results, f)