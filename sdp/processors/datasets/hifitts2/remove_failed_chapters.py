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


import json
from pathlib import Path
from tqdm import tqdm

from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest


class RemovedFailedChapters(BaseProcessor):
    """
    Removes all utterances in the input chapter file from the input manifest. This processor is expected to be
    run using the file output by the DownloadHiFiTTS2 containing failed chapter downloads.

    Args:
        error_file (str): Path to file with chapter download errors.

    Returns:
        This outputs a manifest which is the same as its input manifest but with utterances in 'error_file' removed.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.RemovedFailedChapters
              input_manifest_file: ${workspace_dir}/manifest_22khz.json
              output_manifest_file: ${workspace_dir}/manifest_filtered_22khz.json
              error_file: ${workspace_dir}/errors_22khz.json
    """

    def __init__(
        self,
        error_file: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.error_file = Path(error_file)

    def process(self):
        chapter_rows = load_manifest(self.error_file)
        audio_files_to_remove = set()
        for chapter_row in chapter_rows:
            for utt_list in chapter_row["utterances"]:
                audio_files_to_remove.add(utt_list["audio_filepath"])

        rows = load_manifest(Path(self.input_manifest_file))
        with open(self.output_manifest_file, "w", encoding="utf-8") as output_f:
            for row in tqdm(rows):
                if row["audio_filepath"] in audio_files_to_remove:
                    continue

                output_line = f"{json.dumps(row, ensure_ascii=False)}\n"
                output_f.write(output_line)
