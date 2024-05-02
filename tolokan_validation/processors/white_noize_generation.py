import json
import os
from typing import Dict, List

import numpy as np
from scipy.io.wavfile import write

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class WhiteNoiseManifestGenerator(BaseParallelProcessor):
    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 16000,
        length_seconds: float = 11.3,
        limit: float = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.limit = limit
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.length_seconds = length_seconds

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def generate_white_noise(self, filename: str) -> None:
        # Ensure that the product of sample_rate and length_seconds is an integer
        noise_length = int(self.sample_rate * self.length_seconds)

        # Generate random values between -1 and 1, scaled by the maximum value for 16-bit PCM
        noise = np.random.uniform(-1.0, 1.0, noise_length)
        noise = np.int16(noise * 32767)
        write(filename, self.sample_rate, noise)

    def read_manifest(self):
        with open(self.input_manifest_file, "rt") as fin:
            total_lines = sum(1 for line in fin)

            lines_to_read = max(1, int(total_lines * (self.limit / 100)))

            fin.seek(0)

            # Read the specified percentage of lines
            for _ in range(lines_to_read):
                line = fin.readline()
                yield json.loads(line)

    def process_dataset_entry(self, data_entry):
        filename = f"wn_{os.path.basename(data_entry['audio_path'])}"
        filename = os.path.join(self.output_dir, filename)
        self.generate_white_noise(filename)
        entry = {
            "audio_filepath": filename,
            "duration": self.length_seconds,
            "text": data_entry["text"],
            "GOLDEN:bg_speech": False,
            "GOLDEN:result": "bad",
        }

        return [DataEntry(data=entry)]
