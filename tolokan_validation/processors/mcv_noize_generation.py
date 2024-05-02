import json
import os
from typing import Dict, List

import numpy as np
from scipy.io.wavfile import write

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class MCVNoizeManifestGenerator(BaseParallelProcessor):
    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 16000,
        limit: float = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.limit = limit
        self.output_dir = output_dir
        self.sample_rate = sample_rate

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

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
        filename = f"mcv_{os.path.basename(data_entry['audio_path'])}"
        filename = os.path.join(self.output_dir, filename)
        entry = {
            "audio_path": filename,
            "duration": data_entry["duration"],
            "text": data_entry["text"],
            "GOLDEN:bg_speech": False,
            "GOLDEN:result": "good",
        }

        return [DataEntry(data=entry)]
