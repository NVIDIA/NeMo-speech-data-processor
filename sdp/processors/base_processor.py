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
            be located.
        input_manifest_file (str): path of where the input manifest file is
            located. This arg is optional - some processors may not take in
            an input manifest because they need to create an initial manifest
            from scratch (ie from some transcript file that is in a format
            different to the NeMo manifest format).
    """

    def __init__(self, output_manifest_file: str, input_manifest_file: Optional[str] = None):
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
        chunksize (int): the size of the chunks that will be sent to worker processes.
    """

    def __init__(self, max_workers: int = -1, chunksize: int = 100, **kwargs):
        super().__init__(**kwargs)
        if max_workers == -1:
            max_workers = multiprocessing.cpu_count()
        self.max_workers = max_workers
        self.chunksize = chunksize
        self.number_of_entries = 0
        self.total_duration = 0

    def process(self):
        """Parallelized implementation of the data processing.

        The execution flow of this method is the following.

        1. :meth:`prepare` is called. It's empty by default but can be used to
           e.g. download the initial data files or compute some aggregates
           required for subsequent processing.
        2. A list of data entries is created by calling :meth:`read_manifest`.
           Default implementation reads an input manifest file and returns a
           list of dictionaries for each line (we assume a standard NeMo format
           of one json per line).
        3. :meth:`process_dataset_entry` is called **in parallel** on each element
           of the list created in the previous step. Note that you cannot create
           any new counters or modify the attributes of this class in any way
           inside that function as this will lead to an undefined behavior.
           Each call to the :meth:`process_dataset_entry` returns a list of
           ``DataEntry`` objects that are then aggregated together. ``DataEntry``
           simply defines a ``data`` and ``metrics`` keys.
        4. We loop through all returned data entries and do the following

           a) All ``metrics`` keys are collected in a separate list and passed
              over to the :meth:`finalize` method for any desired metric
              aggregation and reporting.
           b) If ``data`` is set to None, the objects are ignored (metrics are
              still collected).
           c) All non-ignored objects are dumped to the output manifest file
              with a call to ``json.dump``, one object per-line.

        Here is a diagram outlining the execution flow of this method:

        .. can only be viewed in the online documentation

        .. raw:: html

             <div align="center">
               <img src="https://mermaid.ink/img/pako:eNplUl1r6zAM_SvCFy4pbL3vvaVwu-59sL0tl6LESmqIP7DkjWzsv89O0rVjzosiHR8dHetdtV6T2qg-YjjB0-Fv7SAfTs2cqdWjUGAwDrYiuz0yPWDEYaDhIfqWmH1chzmqVts_GQOW5OR1rWaqcv4916pcZxq6jKaAkRb0tok7IBtkXO5BM4KmDtMgUIotOmgIEpMG8VOK1v0atH91g0cNEV9BoyBgEm9RTJvljbX6D7e3O9hfVOyvVURCfbToTEcs11pKocwbksC5PnWFyhB00VvIE7wYnxiWwY3rgbNNqwlnOpATRQLD4B2dhdxdhNx9t2PiOJYRmORITuJYlb85XEydFGDDErGVL4tn6gNcuA-Zm_GFwCf5McJvwL6P1KNQoYim5SlfTY7-At9BEmHQ0YdAenVucH_hv7_W3hmHg3mj40JWXYudX8lwGHD86rb4d7YtN6hd-Qo1Oa1ulKVo0ei8k-8lXatsps0ubnK47EVZrY8MLQ_-OLpWbSQmulEpZNvoYDDvrlWbDgemj0-10vX9" height=100% />
             </div>
        """
        self.prepare()
        dataset_entries = self.read_manifest()

        # this will unroll all inner lists
        data = itertools.chain(
            *process_map(
                self.process_dataset_entry,
                dataset_entries,
                max_workers=self.max_workers,
                chunksize=self.chunksize,
            )
        )
        metrics = []
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for data_entry in tqdm(data):
                metrics.append(data_entry.metrics)
                if data_entry.data is None:
                    continue
                json.dump(data_entry.data, fout)
                self.number_of_entries += 1
                self.total_duration += data_entry.data.get("duration", 0)
                fout.write("\n")

        self.finalize(metrics)

    def prepare(self):
        """Can be used in derived classes to prepare the processing in any way.

        E.g., download data or compute some aggregates. Will be called before
        starting processing the data.
        """

    def read_manifest(self):
        """Reading the input manifest file.

        .. note::
            This function should be overridden in the "initial" class creating
            manifest to read from the original source of data.
        """
        if self.input_manifest_file is None:
            raise NotImplementedError("Override this method if the processor creates initial manifest")

        # TODO: should we not assume that manifest can fully fit in memory?
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            dataset_entries = [json.loads(line) for line in fin.readlines()]

        return dataset_entries

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
