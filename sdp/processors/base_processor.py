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

import itertools
import json
import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from sdp.logging import logger


@dataclass
class DataEntry:
    """A wrapper for data entry + any additional metrics."""

    data: Optional[Dict]  # can be None to drop the entry
    metrics: Any = None


class BaseProcessor(ABC):
    """Abstract class for SDP processors.

    All processor classes inherit from the ``BaseProcessor`` class.
    This is a simple abstract class which has 2 empty methods: :meth:`process`
    and :meth:`test`.

    These serve to remind us that SDP essentially just runs ``.test()`` on all
    processors (to implement :ref:`run-time tests <sdp-runtime-tests>`),
    and then ``.process()`` on all processors.

    Args:
        output_manifest_file (str): path of where the output manifest file will
            be located. Cannot have the same value as ``input_manifest_file``.
        input_manifest_file (str): path of where the input manifest file is
            located. This arg is optional - some processors may not take in
            an input manifest because they need to create an initial manifest
            from scratch (ie from some transcript file that is in a format
            different to the NeMo manifest format). Cannot have the same value
            as ``input_manifest_file``.
    """

    def __init__(self, output_manifest_file: str, input_manifest_file: Optional[str] = None):

        if output_manifest_file and input_manifest_file and (output_manifest_file == input_manifest_file):
            # we cannot have the same input and output manifest file specified because we need to be able to
            # read from the input_manifest_file and write to the output_manifest_file at the same time
            raise ValueError("A processor's specified input_manifest_file and output_manifest_file cannot be the same")

        self.output_manifest_file = output_manifest_file
        self.input_manifest_file = input_manifest_file

    @abstractmethod
    def process(self):
        """Should be overriden by the child classes to implement some data processing."""
        pass

    def test(self):
        """This method can be used to perform "runtime" tests.

        This can be any kind of self-consistency tests, but are usually
        in the form of checking that provided input test data entries match
        provided output test data entries.

        There are not tests by default.
        """

class BaseParallelProcessor(BaseProcessor):
    """
    A processor that performs per-entry processing in parallel (using Dask or multiprocessing).

    Args:
        input_manifest_file (str): Path to the input manifest file.
        output_manifest_file (str): Path where the output manifest file will be written.
        max_workers (int): Maximum number of workers.
        chunksize (int): Chunk size used for parallel routines.
        in_memory_chunksize (int): Maximum number of entries to load at once.
        test_cases (list[dict]): Optional list of test cases.
        use_dask (bool): If True, use Dask for parallelization; otherwise, use multiprocessing.
        dask_client: (Optional) An existing Dask client.
    """
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the Dask client from state (it is not picklable)
        if 'dask_client' in state:
            state['dask_client'] = None
        return state

    def __init__(
        self,
        input_manifest_file: Optional[str] = None,
        output_manifest_file: Optional[str] = None,
        max_workers: int = -1,
        chunksize: int = 100,
        in_memory_chunksize: int = 100000,
        test_cases: Optional[List[Dict]] = None,
        use_dask: bool = True,
        dask_client=None,
        **kwargs,
    ):
        super().__init__(input_manifest_file=input_manifest_file, output_manifest_file=output_manifest_file, **kwargs)
        if max_workers == -1:
            max_workers = os.cpu_count()
        self.max_workers = max_workers
        self.chunksize = chunksize
        self.in_memory_chunksize = in_memory_chunksize
        self.number_of_entries = 0
        self.total_duration = 0
        self.start_time = time.time()
        self.test_cases = test_cases or []
        self.use_dask = use_dask
        self.dask_client = dask_client

    def process(self):
        os.environ.setdefault("PATH", os.defpath)
        if hasattr(self, 'prepare'):
            self.prepare()
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        metrics = []
        
        #Ability to work sa legacy and as dask
        if self.use_dask:
            self._process_with_dask(metrics)
        else:
            self._process_with_multiprocessing(metrics)
        self.finalize(metrics)

    def _process_with_dask(self, metrics):
        import dask.bag as db
        from dask.distributed import Client

        if self.dask_client is None:
            self.dask_client = Client()
        client = self.dask_client
        from sdp.logging import logger  # Assume a logger exists in your framework.
        logger.info(f"Using Dask client with dashboard at: {client.dashboard_link}")

        # Delegate manifest reading to read_manifest() which returns a Dask bag.
        bag = self.read_manifest()
        total_entries = bag.count().compute()

        if total_entries == 0:
            logger.info("No entries found in the manifest input. Proceeding to create an empty output manifest.")
            results = []
        else:
            processed_bag = bag.map(lambda entry: self.process_dataset_entry(entry)).flatten()
            results = processed_bag.compute()

        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for entry in results:
                metrics.append(entry.metrics)
                if entry.data is not None:
                    json.dump(entry.data, fout, ensure_ascii=False)
                    fout.write("\n")
                    self.number_of_entries += 1
                    self.total_duration += entry.data.get("duration", 0)
        logger.info(f"Processed {total_entries} entries using Dask.")

    def _process_with_multiprocessing(self, metrics):
        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for manifest_chunk in self._chunk_manifest():
                data = itertools.chain(
                    *process_map(
                        self.process_dataset_entry,
                        manifest_chunk,
                        max_workers=self.max_workers,
                        chunksize=self.chunksize,
                    )
                )
                for data_entry in tqdm(data):
                    metrics.append(data_entry.metrics)
                    if data_entry.data is None:
                        continue
                    json.dump(data_entry.data, fout, ensure_ascii=False)
                    fout.write("\n")
                    self.number_of_entries += 1
                    self.total_duration += data_entry.data.get("duration", 0)

    def _chunk_manifest(self):
        """Splits the input manifest into chunks of in_memory_chunksize size.
           Only used in non-Dask (multiprocessing) mode.
        """
        manifest_chunk = []
        # When use_dask is False, read_manifest() returns an iterator.
        for idx, data_entry in enumerate(self.read_manifest(), 1):
            manifest_chunk.append(data_entry)
            if idx % self.in_memory_chunksize == 0:
                yield manifest_chunk
                manifest_chunk = []
        if manifest_chunk:
            yield manifest_chunk

    def read_manifest(self):
        """
        Reads entries from the input manifest.
        
        Behavior depends on the parallelization mode:
         - When use_dask is True:
              If the input_manifest_file exists and is non-empty, returns a Dask bag (reading in 256KB blocks).
              Otherwise, logs the condition and returns an empty Dask bag.
         - When use_dask is False:
              If the input_manifest_file does not exist or is empty, logs the condition and returns an empty iterator.
              Otherwise, opens the file in text mode, strips each line, and yields the parsed JSON from non-empty lines.
        This unified behavior lets the processor run even in manifest-creation mode.
        """
        from sdp.logging import logger  
        if self.use_dask:
            import dask.bag as db
            if self.input_manifest_file and os.path.exists(self.input_manifest_file) and os.path.getsize(self.input_manifest_file) > 0:
                bag = db.read_text(self.input_manifest_file, blocksize=2**18).map(json.loads)
                return bag
            else:
                logger.info("No input manifest file provided or file is empty. Returning an empty Dask bag for manifest creation.")
                return db.from_sequence([])
        else:
            if not self.input_manifest_file or not os.path.exists(self.input_manifest_file):
                logger.info("No input manifest file provided or file does not exist. Continuing with an empty manifest.")
                return iter([])
            else:
                def generator():
                    with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
                        for line in fin:
                            line = line.strip()
                            if line:
                                yield json.loads(line)
                return generator()

    @abstractmethod
    def process_dataset_entry(self, data_entry) -> List[Any]:
        """
        Must be implemented in derived classes.
        For each data entry, return a list of DataEntry objects.
        """
        raise NotImplementedError("Derived classes must implement process_dataset_entry.")

    def finalize(self, metrics: List[Any]):
        from sdp.logging import logger
        logger.info("Total number of entries after processing: %d", self.number_of_entries)
        if self.total_duration:
            logger.info("Total audio duration (hours) after processing: %.2f", self.total_duration / 3600)
        else:
            logger.info("Unable to calculate total audio duration (hours).")
        elapsed = time.time() - self.start_time
        logger.info("Processor completed in (seconds): %.2f", elapsed)

    def test(self):
        for test_case in self.test_cases:
            input_data = test_case["input"].copy() if isinstance(test_case["input"], dict) else test_case["input"]
            generated_outputs = self.process_dataset_entry(input_data)
            expected_outputs = [test_case["output"]] if not isinstance(test_case["output"], list) else test_case["output"]
            for gen_out, exp_out in zip(generated_outputs, expected_outputs):
                gen_data = gen_out.data if hasattr(gen_out, "data") else gen_out
                if gen_data != exp_out:
                    raise RuntimeError(
                        "Runtime test failed.\nTest input: {}\nGenerated output: {}\nExpected output: {}"
                        .format(test_case["input"], gen_data, exp_out)
                    )



# ------------------ Legacy Parallel Processor ------------------ #Just for reference
class LegacyParallelProcessor(BaseProcessor):
    """
    A legacy parallel processor implementation using multiprocessing and process_map.
    
    This class processes the manifest in chunks (using process_map) and is provided for compatibility.
    Child classes must implement process_dataset_entry().
    
    Args:
        max_workers (int): Maximum workers.
        chunksize (int): Number of entries per chunk.
        in_memory_chunksize (int): Maximum entries read in one go.
        test_cases (list[dict]): Optional test cases.
    """
    def __init__(
        self,
        max_workers: int = -1,
        chunksize: int = 100,
        in_memory_chunksize: int = 100000,
        test_cases: Optional[List[Dict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_workers == -1:
            max_workers = multiprocessing.cpu_count()
        self.max_workers = max_workers
        self.chunksize = chunksize
        self.in_memory_chunksize = in_memory_chunksize
        self.number_of_entries = 0
        self.total_duration = 0
        self.start_time = time.time()
        self.test_cases = test_cases or []

    def process(self):
        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for manifest_chunk in self._chunk_manifest():
                data = itertools.chain(
                    *process_map(
                        self.process_dataset_entry,
                        manifest_chunk,
                        max_workers=self.max_workers,
                        chunksize=self.chunksize,
                    )
                )
                for data_entry in tqdm(data):
                    if data_entry.metrics is not None:
                        pass  # optionally accumulate metrics here
                    if data_entry.data is None:
                        continue
                    json.dump(data_entry.data, fout, ensure_ascii=False)
                    self.number_of_entries += 1
                    self.total_duration += data_entry.data.get("duration", 0)
                    fout.write("\n")
        self.finalize(self.test_cases)

    def _chunk_manifest(self):
        manifest_chunk = []
        for idx, data_entry in enumerate(self.read_manifest(), 1):
            manifest_chunk.append(data_entry)
            if idx % self.in_memory_chunksize == 0:
                yield manifest_chunk
                manifest_chunk = []
        if manifest_chunk:
            yield manifest_chunk

    def read_manifest(self):
        if not self.input_manifest_file:
            raise NotImplementedError("Override this method if no input manifest file is used")
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            for line in fin:
                yield json.loads(line)

    @abstractmethod
    def process_dataset_entry(self, data_entry) -> List[DataEntry]:
        raise NotImplementedError("Derived classes must implement `process_dataset_entry`.")

    def finalize(self, metrics):
        logger.info("Total number of entries after processing (legacy): %d", self.number_of_entries)
        if self.total_duration:
            logger.info("Total audio duration (hours) after processing (legacy): %.2f", self.total_duration / 3600)
        else:
            logger.info("Unable to calculate total audio duration (legacy).")
        elapsed = time.time() - self.start_time
        logger.info("Legacy processor completed in (seconds): %.2f", elapsed)