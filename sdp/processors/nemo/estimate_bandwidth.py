import librosa
import numpy as np
from pathlib import Path

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class EstimateBandwidth(BaseParallelProcessor):
    """
    Adds estimated bandwidth to each utterance in the input manifest file.

    Args:
        audio_dir (str): Root directory where audio files are stored.
        input_audio_key (str): Manifest key with relative audio paths.
        output_bandwidth_key (str): Manifest key to store estimated bandwidth in.
        max_seconds (float): The maximum length of audio to use for bandwidth estimation.
            By default, uses the first 30 seconds.
        sample_rate (int): Sample rate to resample audio to before doing bandwidth estimation.
            Defaults to 44100, upsampling the input audio as needed.
        n_fft (int): Number of FFT bins to use for bandwidth estimation. Defaults to 512.
        hop_length (int): Audio frame hop length to use for bandwidth estimation.
            Defaults to 441, corresponding to 0.01 seconds for 44100 sample rate.
        top_db (float): top_db treshhold to use for bandwidth estimation.
        frequency_threshold (float): Bandwidth estimation finds the highest frequency with mean power spectrum that is
            within 'frequency_threshold' dB of its peak power. Defaults to -50 dB.

    Returns:
        This processor estimates the bandwidth of the audio file in the`input_audio_key` field and saves the estimate
            in the output_bandwidth_key` field.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.EstimateBandwidth
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_bandwidth.json
              audio_dir: ${workspace_dir}/audio_22khz
              max_workers: 8
    """

    def __init__(
        self,
        audio_dir: str,
        input_audio_key: str = "audio_filepath",
        output_bandwidth_key: str = "bandwidth",
        max_seconds: float = 30.0,
        sample_rate: int = 44100,
        n_fft: int = 512,
        hop_length: int = 441,
        top_db: float = 100.0,
        frequency_threshold: float = -50.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_directory = Path(audio_dir)
        self.input_audio_key = input_audio_key
        self.output_bandwidth_key = output_bandwidth_key
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.top_db = top_db
        self.frequency_threshold = frequency_threshold

    def _estimate_bandwidth(self, audio, sample_rate):
        spec = librosa.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop_length, window="blackmanharris")
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

    def process_dataset_entry(self, data_entry):
        audio_filename = data_entry[self.input_audio_key]
        audio_file = self.audio_directory / audio_filename
        audio, sr = librosa.load(path=audio_file, sr=self.sample_rate, duration=self.max_seconds)
        bandwidth = self._estimate_bandwidth(audio=audio, sample_rate=sr)
        data_entry[self.output_bandwidth_key] = int(bandwidth)
        return [DataEntry(data=data_entry)]
