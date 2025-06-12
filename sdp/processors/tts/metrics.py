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
import numpy as np
from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest, save_manifest

import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.pipelines import SQUIM_OBJECTIVE


class TorchSquimObjectiveQualityMetricsProcessor(BaseProcessor):
    """This processor calculates Squim quality metrics for audio files.

    It uses a pre-trained Squim model to calculate audio quality metrics like PESQ, STOI
    and SI-SDR for each audio segment in the manifest:

        PESQ (Perceptual Evaluation of Speech Quality)
        A measure of overall quality for speech (originally designed to detect codec distortions but highly correlated to all kinds of distortion.
        
        STOI (Short-Time Objective Intelligibility)
        A measure of speech intelligibility, basically measures speech envelope integrity. 
        A STOI value of 1.0 means 100% of the speech being evaluated is intelligible on average.

        SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
        A measure of how strong the speech signal is vs. all the distortion present in the audio, in decibels. 
        0 dB means the energies of speech and distortion are the same. A value between 15-20 dB is what is considered "clean enough" speech in general.

    Args:
        device (str, Optional): Device to run the model on. Defaults to "cuda".

    Returns:
        The same data as in the input manifest, but with quality metrics added to each
        segment's metrics field.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.tts.metrics.TorchSquimObjectiveQualityMetricsProcessor
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_squim.json
    """
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)

        if not torch.cuda.is_available():
            device="cpu"
            logger.warning("CUDA is not available, using CPU")
        
        if device == "cuda":
            self.model = SQUIM_OBJECTIVE.get_model().cuda()
        else:
            self.model = SQUIM_OBJECTIVE.get_model()

    def process(self):
        manifest = load_manifest(self.input_manifest_file)

        results = []

        for metadata in tqdm(manifest):
            info = torchaudio.info(metadata['resampled_audio_filepath'])
            sr = info.sample_rate

            try: 
                audio, _ = librosa.load(path=metadata['resampled_audio_filepath'], sr=sr)
            except Exception as ex:
                logger.info(f"Failed to load {metadata['resampled_audio_filepath']}, exception={ex}")
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
                    logger.info('Failed to extract Squim metrics {} with frame_offset={} and num_frames={}'.format(
                        metadata['resampled_audio_filepath'],
                        start,
                        num_samples))
                    continue
            results.append(metadata)

        save_manifest(results, self.output_manifest_file)


class BandwidthEstimationProcessor(BaseProcessor):
    """This processor estimates audio bandwidth by analyzing power spectra.

    It analyzes audio files to estimate their effective bandwidth by examining the power
    spectrum and determining the highest frequency with significant energy content above
    a threshold.

    Args:
        n_fft (int, Optional): Size of FFT window. Defaults to 512
        stride_seconds (float, Optional): Time between successive FFT windows in seconds. Defaults to 0.01
        top_db (float, Optional): Maximum decibel value for power spectrum normalization. Defaults to 100.0
        frequency_threshold (float, Optional): Threshold in dB below peak for bandwidth estimation. Defaults to -50.0

    Returns:
        The same data as in the input manifest, but with bandwidth estimates added
        to each segment.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.tts.metrics.BandwidthEstimationProcessor
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_with_bandwidth.json
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
        manifest = load_manifest(self.input_manifest_file)

        results = []

        for metadata in tqdm(manifest):
            audio_filepath = metadata['audio_filepath']
            try: 
                audio, sample_rate = librosa.load(path=audio_filepath, sr=None)
            except Exception as ex:
                logger.info(f"Failed to load {audio_filepath}, exception={ex}")
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

        save_manifest(results, self.output_manifest_file)

