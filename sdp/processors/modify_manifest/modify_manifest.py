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


from abc import abstractmethod
from typing import Dict, List, Optional

from sdp.processors.base_processor import BaseParallelProcessor
from sdp.utils.edit_spaces import add_start_end_spaces, remove_extra_spaces

# TODO: maybe remove additional spaces for simpler logic? Why is it necessary
#       for regular expressions?


class ModifyManifestTextProcessor(BaseParallelProcessor):
    """Base class useful for most "text-only" modifications of the manifest.

    This adds the following functionality on top of the
    :class:`sdp.processors.base_processor.BaseParallelProcessor`

    Args:
        text_key (str): a string indicating which key of the data entries
            should be used to find an utterance transcript. Defaults to "text".
        pred_text_key (str): a string indicating which key of the data entries
            should be used to access the ASR predictions. Defaults to "pred_text".

    .. note::
        This class only supports one-to-one or one-to-none mappings.
        One-to-many is currently not supported.
    """

    def __init__(
        self,
        text_key: str = "text",
        pred_text_key: str = "pred_text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.pred_text_key = pred_text_key

    @abstractmethod
    def _process_dataset_entry(self, data_entry):
        """Main data processing should be implemented here.
        """
        pass

    def process_dataset_entry(self, data_entry):
        """Wrapper for the new :meth:`_process_dataset_entry` abstract method.
        """

        data_entries = self._process_dataset_entry(data_entry)

        return data_entries
