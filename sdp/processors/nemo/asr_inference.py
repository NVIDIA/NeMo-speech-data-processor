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

import os
import subprocess
import shutil
from pathlib import Path

from sdp.processors.base_processor import BaseProcessor

# Note that we do not re-use base parallel implementation, since the ASR
# inference is already run in batches.

# TODO: actually, it might still be beneficial to have another level of
#       parallelization, but that needs to be tested.


class ASRInference(BaseProcessor):
    """This processor performs ASR inference on each utterance of the input manifest.

    ASR predictions will be saved in the ``pred_text`` key.

    Args:
        pretrained_model (str): the name of the pretrained NeMo ASR model
            which will be used to do inference.
        batch_size (int): the batch size to use for ASR inference. Defaults to 32.

    Returns:
         The same data as in the input manifest with an additional field
         ``pred_text`` containing ASR model's predictions.
    """

    def __init__(
        self,
        pretrained_model: str,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.script_path = Path(__file__).parents[1] / "nemo" / "transcribe_speech.py"
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size

    def process(self):
        """This will add "pred_text" key into the output manifest."""
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        if self.pretrained_model[-5:] == ".nemo":
            subprocess.run(
                f"python {self.script_path} "
                f"model_path={self.pretrained_model} "
                f"dataset_manifest={self.input_manifest_file} "
                f"output_filename={self.output_manifest_file} "
                f"batch_size={self.batch_size} ",
                shell=True,
                check=True,
            )
        else:
            subprocess.run(
                f"python {self.script_path} "
                f"pretrained_name={self.pretrained_model} "
                f"dataset_manifest={self.input_manifest_file} "
                f"output_filename={self.output_manifest_file} "
                f"batch_size={self.batch_size} ",
                shell=True,
                check=True,
            )


class ASRInferenceParallel(BaseProcessor):
    def __init__(
        self,
        pretrained_model: str,
        batch_size: int = 32,
        devices: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.script_path = Path(__file__).parents[1] / "nemo" / "transcribe_speech_parallel.py"
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.devices = devices
        self.output_manifest_dir = self.output_manifest_file.replace(".json", "")

    def process(self):
        subprocess.run(
            f"python {self.script_path} "
            f"model={self.pretrained_model} "
            f"predict_ds.manifest_filepath={self.input_manifest_file} "
            f"output_path={self.output_manifest_dir} "
            f"predict_ds.batch_size={self.batch_size} "
            f"trainer.devices={self.devices} ",
            shell=True,
            check=True,
        )

        os.rename(os.path.join(self.output_manifest_dir, "predictions_all.json"), self.output_manifest_file)
        shutil.rmtree(self.output_manifest_dir)
