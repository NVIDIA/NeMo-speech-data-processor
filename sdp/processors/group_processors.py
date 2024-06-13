# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from sdp.processors.base_processor import BaseProcessor
from sdp.utils.chunk_processing import ChunkProcessingPipeline


class GroupProcessors(BaseProcessor):
    def __init__(
        self,
        output_manifest_file: str,
        input_manifest_file: str | None = None,
        chunksize: int = 500,
        **processors_cfg,
    ):
        super().__init__(
            output_manifest_file=output_manifest_file,
            input_manifest_file=input_manifest_file,
        )

        self.initial_manifest_file = input_manifest_file
        self.chunksize = chunksize
        self.processors_cfg = processors_cfg["processors"]

    def test(self):
        pass

    def process(self):
        chunked_pipeline = ChunkProcessingPipeline(
            initial_manifest_file=self.initial_manifest_file,
            chunksize=self.chunksize,
            processors_cfgs=self.processors_cfg,
        )

        chunked_pipeline.run()
