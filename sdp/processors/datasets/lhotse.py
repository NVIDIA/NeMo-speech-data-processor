# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# To convert mp3 files to wav using sox, you must have installed sox with mp3 support
# For example sudo apt-get install libsox-fmt-mp3
from sdp.processors.base_processor import BaseProcessor


class LhotseImport(BaseProcessor):
    """Processor to create an initial manifest import from a Lhotse CutSet.

    Lhotse is a library for speech data processing and loading; see:
    - https://github.com/lhotse-speech/lhotse
    - https://lhotse.readthedocs.io

    TODO: list limitations

    Returns:
        This processor generates an initial manifest file with the following fields::

            {
                "audio_filepath": <path to the audio file>,
                "duration": <duration of the audio in seconds>,
                "text": <transcription (with capitalization and punctuation)>,
            }
    """

    def process(self):
        from lhotse import CutSet

        cuts = CutSet.from_file(self.input_manifest_file)
        with open(self.output_manifest_file, "w") as f:
            for cut in cuts:
                self.check_entry(cut)
                data = {
                    "audio_filepath": cut.recording.sources[0].source,
                    "duration": cut.duration,
                }
                for meta in ("text", "speaker", "gender", "language"):
                    if (item := getattr(cut.supervisions[0], meta)) is not None:
                        data[meta] = item
                print(json.dumps(data), file=f)

    def check_entry(self, cut) -> None:
        from lhotse import MonoCut

        assert isinstance(
            cut, MonoCut
        ), f"Currently, only MonoCut import is supported. Received: {cut}"
        assert (
            cut.has_recording
        ), f"Currently, we only support cuts with recordings. Received: {cut}"
        assert (
            cut.recording.num_channels == 1
        ), f"Currently, we only supports recordings with a single channel. Received: {cut}"
        assert (
            len(cut.recording.sources) == 1
        ), f"Currently, we only support recordings with a single AudioSource. Received: {cut}"
        assert (
            cut.recording.sources[0].type == "file"
        ), f"Currently, we only suppport AudioSources of type='file'. Received: {cut}"
        assert (
            len(cut.supervisions) == 1
        ), f"Currently, we only support cuts with a single supervision. Received: {cut}"
