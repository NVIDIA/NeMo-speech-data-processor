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
import os
import tempfile
import uuid
from typing import Any

import hydra
from omegaconf import OmegaConf

from sdp.logging import logger
from sdp.utils.common import read_manifest, write_manifest

def get_last_output_manifest_file_in_group(group_processors_cfg):
    return group_processors_cfg[-1].get("output_manifest_file", None)

class ChunkedProcessor:
    def __init__(
        self,
        chunk_input_file: str,
        chunk_output_file: str,
        output_manifest_file: str,
        **processor_kwargs: Any,
    ) -> None:
        self.processor_cfg = processor_kwargs
        self.chunk_input_file = chunk_input_file
        self.chunk_output_file = chunk_output_file
        self.agg_output_manifest_file = output_manifest_file

        self.processor = None

    def build_processor(self):
        if "input_manifest_file" in self.processor_cfg:
            logger.warning(
                f"Processor inside chunked pipeline can't have `input_manifest_file` argument [{self.processor_cfg['_target_']}: {self.processor_cfg['input_manifest_file']}]. It will be chaged to the value of `chunk_input_file` ({self.processor_cfg['chunk_input_file']})."
            )

        self.processor_cfg["input_manifest_file"] = self.chunk_input_file
        self.processor_cfg["output_manifest_file"] = self.chunk_output_file

        self.processor = hydra.utils.instantiate(self.processor_cfg, _recursive_=False)
        self.processor.test()

    def append_chunk_to_agg_output(self):
        samples = [sample for sample in read_manifest(self.chunk_output_file)]

        write_manifest(
            samples=samples,
            manifest_filepath=self.agg_output_manifest_file,
            mode="a",
        )
        logger.info(
            f"Chunk output of processor `{self.processor_cfg['_target_']}` added to {self.agg_output_manifest_file}."
        )

    def process(self):
        logger.info('=> Running processor "%s"', self.processor)
        self.processor.process()


class СhunkRunner:
    def __init__(
        self,
        initial_manifest_chunk_file: str,
        chunk_steps_dir: str,
        processors_cfgs: list[dict],
        aggregation_at_end: bool = True,
    ):
        self.initial_manifest_chunk_file = initial_manifest_chunk_file
        self.chunk_steps_dir = chunk_steps_dir
        self.chunk_processors_cfgs = processors_cfgs.copy()
        self.processors = None
        self.aggregation_at_end = aggregation_at_end

    def prepare(self):
        os.makedirs(self.chunk_steps_dir, exist_ok=True)

    def set_chunk_configs(self):
        if "chunk_input_file" in self.chunk_processors_cfgs[0]:
            logger.warning(
                f"`chunk_input_file` can't be set for the 1st processor in chunked pipeline processing. Value will be set as path to file of manifest chunk ({self.initial_manifest_chunk_file})."
            )

        self.chunk_processors_cfgs[0][
            "chunk_input_file"
        ] = self.initial_manifest_chunk_file
        self.chunk_processors_cfgs[0].setdefault(
            "chunk_output_file", os.path.join(self.chunk_steps_dir, str(uuid.uuid4()))
        )

        for i, processor_cfg in enumerate(self.chunk_processors_cfgs[1:]):
            processor_cfg.setdefault(
                "chunk_input_file",
                self.chunk_processors_cfgs[i]["chunk_output_file"],
            )
            processor_cfg.setdefault(
                "chunk_output_file",
                os.path.join(self.chunk_steps_dir, str(uuid.uuid4())),
            )

        self.chunk_processors_cfgs = OmegaConf.to_container(
            OmegaConf.create(self.chunk_processors_cfgs), resolve=True
        )

        logger.info(
            f"Chunk hydra config:\n{OmegaConf.to_yaml(self.chunk_processors_cfgs)}"
        )

    def build_processors(self):
        self.processors = []
        for processor_cfg in self.chunk_processors_cfgs:
            processor = ChunkedProcessor(**processor_cfg)
            processor.build_processor()
            self.processors.append(processor)

    def run_processors(self):
        for processor in self.processors:
            logger.info('=> Running processor "%s"', processor)
            processor.process()

            if not self.aggregation_at_end:
                processor.append_chunk_to_agg_output()

    def process(self):
        self.prepare()
        self.set_chunk_configs()
        self.build_processors()
        self.run_processors()

        if self.aggregation_at_end:
            logger.info("Appending chunk outputs to `output_manifest_file`..")
            for processor in self.processors:
                processor.append_chunk_to_agg_output()


class ChunkProcessingPipeline:
    def __init__(
        self,
        initial_manifest_file: str,
        last_output_manifest_file: str,
        processors_cfgs: list[dict],
        chunksize: int = 100,
        aggregation_at_end: bool = True,
        light_logging: bool = True,
    ):
        self.initial_manifest_file = initial_manifest_file
        self.last_output_manifest_file = last_output_manifest_file
        self.chunksize = chunksize
        self.processors_cfgs = processors_cfgs
        self.aggregation_at_end = aggregation_at_end

        self.tmp_dir = None

    def prepare(self):
        for processor_cfg in self.processors_cfgs[:-1]:
            if "output_manifest_file" not in processor_cfg:
                processor_cfg["output_manifest_file"] = os.path.join(
                    self.tmp_dir, str(uuid.uuid4())
                )
            os.makedirs(
                os.path.dirname(processor_cfg["output_manifest_file"]), exist_ok=True
            )
            write_manifest(processor_cfg["output_manifest_file"])
        
        if "output_manifest_file" not in self.processors_cfgs[-1]:
            self.processors_cfgs[-1]['output_manifest_file'] = self.last_output_manifest_file
            os.makedirs(
                os.path.dirname(self.last_output_manifest_file), exist_ok=True
            )
            write_manifest(self.last_output_manifest_file)

    def chunk_manifest(self):
        """Splits the manifest into smaller chunks defined by ``chunksize``."""
        manifest_chunk = []
        for idx, data_entry in enumerate(read_manifest(self.initial_manifest_file), 1):
            manifest_chunk.append(data_entry)
            if idx % self.chunksize == 0:
                yield manifest_chunk
                manifest_chunk = []
        if len(manifest_chunk) > 0:
            yield manifest_chunk

    def run(self):
        with tempfile.TemporaryDirectory() as pipeline_tmp_dir:
            self.tmp_dir = pipeline_tmp_dir
            self.prepare()

        chunk_no = 1
        for chunk_samples in self.chunk_manifest():
            logger.info(f"Starting batch #{chunk_no} processing:..".center(50, "-"))

            with tempfile.TemporaryDirectory() as chunk_tmp_dir:
                initial_chunk_file = os.path.join(chunk_tmp_dir, str(uuid.uuid4()))
                write_manifest(
                    manifest_filepath=initial_chunk_file,
                    samples=chunk_samples,
                )

                chunk = СhunkRunner(
                    initial_manifest_chunk_file=initial_chunk_file,
                    chunk_steps_dir=chunk_tmp_dir,
                    processors_cfgs=self.processors_cfgs,
                    aggregation_at_end=self.aggregation_at_end,
                )
                chunk.process()

            logger.info(f"Batch #{chunk_no} processing finished.".center(50, "-"))
            chunk_no += 1
