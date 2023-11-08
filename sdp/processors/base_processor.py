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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
            raise ValueError(
                "A processor's specified input_manifest_file and output_manifest_file cannot be the same"
            )

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
    """Processor class which allows operations on each utterance to be parallelized.

    Parallelization is done using ``tqdm.contrib.concurrent.process_map`` inside
    the :meth:`process` method. Actual processing should be defined on a
    per-examples bases inside the :meth:`process_dataset_entry` method.

    See the documentation of all the methods for more details.

    Args:
        max_workers (int): maximum number of workers that will be spawned
            during the parallel processing.
        chunksize (int): the size of the chunks that will be sent to worker processes
            during the parallel processing.
        in_memory_chunksize (int): the maximum number of input data entries that will
            be read, processed and saved at a time.
        test_cases (list[dict]): an optional list of dicts containing test
            cases for checking that the processor makes the changes that we
            are expecting.
            The dicts must have a key ``input``, the value of which is a dictionary
            containing data which is our test's input manifest line, and a key
            ``output``, the value of which is a dictionary containing data which is
            the expected output manifest line.
    """

    def __init__(
        self,
        max_workers: int = -1,
        chunksize: int = 100,
        in_memory_chunksize: int = 1000000,
        test_cases: Optional[List[Dict]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if max_workers == -1:
            max_workers = multiprocessing.cpu_count()
        self.max_workers = max_workers
        self.chunksize = chunksize
        self.in_memory_chunksize = in_memory_chunksize
        self.number_of_entries = 0
        self.total_duration = 0

        self.test_cases = test_cases
        # need to convert to list to avoid errors in iteration over None
        if self.test_cases is None:
            self.test_cases = []

    def process(self):
        """Parallelized implementation of the data processing.

        The execution flow of this method is the following.

        1. :meth:`prepare` is called. It's empty by default but can be used to
           e.g. download the initial data files or compute some aggregates
           required for subsequent processing.
        2. A for-loop begins that loops over all ``manifest_chunk`` lists yielded
           by the :meth:`_chunk_manifest` method. :meth:`_chunk_manifest` reads data
           entries yielded by :meth:`read_manifest` and yields lists containing
           ``in_memory_chunksize`` data entries.

           Inside the for-loop:

           a) :meth:`process_dataset_entry` is called **in parallel** on each element
              of the ``manifest_chunk`` list.
           b) All metrics are aggregated.
           c) All output data-entries are added to the contents of ``output_manifest_file``.

           Note:

           * The default implementation of :meth:`read_manifest` reads an input manifest file
             and returns a list of dictionaries for each line (we assume a standard NeMo format
             of one json per line).
           * :meth:`process_dataset_entry` is called **in parallel** on each element
             of the list created in the previous step. Note that you cannot create
             any new counters or modify the attributes of this class in any way
             inside that function as this will lead to an undefined behavior.
             Each call to the :meth:`process_dataset_entry` returns a list of
             ``DataEntry`` objects that are then aggregated together. ``DataEntry``
             simply defines a ``data`` and ``metrics`` keys.
           * If ``data`` is set to None, the objects are ignored (metrics are still collected).

        3. All ``metrics`` keys that were collected in the for-loop above are passed over to
           :meth:`finalize` for any desired metric aggregation and reporting.

        Here is a diagram outlining the execution flow of this method:

        .. can only be viewed in the online documentation

        .. raw:: html

             <div align="center">
               <img src="https://mermaid.ink/img/pako:eNqFU99r2zAQ_lcOFUYCbfbuhcCS9HFQ6N7mYS7WyRaTJSOdF7zS_32SrDYuDOYn-e6777779SJaJ0lUovM49vD9_KW2EL8wXRZLLZ6ZxgDawp75cMRAT-jRGDJP3rUUgvO7cXlttvvPEQMDce9kLRaq9H39UYsUHsioiKYRPRX0_uIPQMPIc4mDywySFE6GITlbtHAhmAJJYJdNtOt2IN3VGocSPF5BIiPgxG5A1m2UN9fiJzw8HOB8U3Fcq_CEshnQakWB11qKimuv2x5mTUaGnBPjb05Dlv07_elGf1rTN20_2V__T1CakVPkkABOQdB_KFne6bRtBhqcnxfe7E-E_6jyHGUo5yELTgRvGpbQHFYl8g-gVFmTK7sBUh_hg4wy6CahA_ESsLnFlgXIZW7i3PAS2GPLpebt4vkEAX8TuInHKbqKvGjGrvPUIVPCe92GjN_kcd-lvkzMAaR340hy-1b74632x_UIlLZoYqOaQrZZq1tWSIfR4PyeTXk3QKlR2y4mqG25B54NwRGUNqa6U0qtzae1eXGQlbUV92IgP6CW8cBekqMW3NNAtajisyx5upPXCE3b-zzbVlTsJ7oX0xj7SmeN8RAHUSk0IVpJanb-23K0-XZf_wKzfkSg" height=100% />
             </div>
        """
        self.prepare()
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        metrics = []

        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for manifest_chunk in self._chunk_manifest():
                # this will unroll all inner lists
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
                    self.number_of_entries += 1
                    self.total_duration += data_entry.data.get("duration", 0)
                    fout.write("\n")

        self.finalize(metrics)

    def prepare(self):
        """Can be used in derived classes to prepare the processing in any way.

        E.g., download data or compute some aggregates. Will be called before
        starting processing the data.
        """

    def _chunk_manifest(self):
        """Splits the manifest into smaller chunks defined by ``in_memory_chunksize``.
        """
        manifest_chunk = []
        for idx, data_entry in enumerate(self.read_manifest(), 1):
            manifest_chunk.append(data_entry)
            if idx % self.in_memory_chunksize == 0:
                yield manifest_chunk
                manifest_chunk = []
        if len(manifest_chunk) > 0:
            yield manifest_chunk

    def read_manifest(self):
        """Reading the input manifest file.

        .. note::
            This function should be overridden in the "initial" class creating
            manifest to read from the original source of data.
        """
        if self.input_manifest_file is None:
            raise NotImplementedError("Override this method if the processor creates initial manifest")

        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            for line in fin:
                yield json.loads(line)

    @abstractmethod
    def process_dataset_entry(self, data_entry) -> List[DataEntry]:
        """Needs to be implemented in the derived classes.

        Each returned value should be a ``DataEntry`` object that will hold
        a dictionary (or anything else that can be json-serialized) with
        the actual data + any additional metrics required for statistics
        reporting. Those metrics can be used in :meth:`finalize` to
        prepare for final reporting.

        ``DataEntry`` is a simple dataclass defined in the following way::

            @dataclass
            class DataEntry:
                # can be None to drop the entry
                data: Optional[Dict]
                # anything - you'd need to aggregate all
                # values in the finalize method manually
                metrics: Any = None

        .. note::
            This method should always return a list of objects to allow a
            one-to-many mapping. E.g., if you want to cut an utterance into
            multiple smaller parts, you can return a list of all the produced
            utterances and they will be handled correctly.

            The many-to-one mapping is not currently supported by design of
            this method (but can still be done if you don't inherit from
            this class and process the data sequentially).

        Args:
            data_entry: most often, ``data_entry`` will be a dictionary
                containing items which represent the JSON manifest entry.
                Sometimes, such as in :class:`sdp.processors.CreateInitialManifestMLS`,
                it will be a string containing a line for that utterance
                from the original raw MLS transcript. In general it is an element
                of the list returned from the :meth:`read_manifest` method.
        """
        # TODO: it would be more straightforward to use a generator here, but
        #     seems that it's not supported with multiprocessing. Is there a
        #     way to make it work?

    def finalize(self, metrics: List):
        """Can be used to output statistics about the processed data.

        By default outputs new number of entries/hours.

        Args:
            metrics (list): a list containing all ``metrics`` keys from the
                data entries returned from the :meth:`process_dataset_entry`
                method.
        """
        logger.info("Total number of entries after processing: %d", self.number_of_entries)
        if self.total_duration != 0:
            logger.info("Total audio duration (hours) after processing: %.2f", self.total_duration / 3600)


    def test(self):
        """Applies processing to "test_cases" and raises an error in case of mismatch."""
        for test_case in self.test_cases:
            generated_outputs = self.process_dataset_entry(test_case["input"].copy())
            expected_outputs = (
                [test_case["output"]] if not isinstance(test_case["output"], list) else test_case["output"]
            )

            for generated_output, expected_output in zip(generated_outputs, expected_outputs):
                generated_output = generated_output.data

                if generated_output != expected_output:
                    raise RuntimeError(
                        "Runtime test failed.\n"
                        f"Test input: {test_case['input']}\n"
                        f"Generated output: {generated_output}\n"
                        f"Expected output: {expected_output}"
                    )
