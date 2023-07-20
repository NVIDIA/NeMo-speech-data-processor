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

    * Adds space in the beginning and end of text for easier regex-based
      processing.
    * Automatically handles common test cases by comparing input to output
      values.

    Args:
        text_key (str): a string indicating which key of the data entries
            should be used to find an utterance transcript. Defaults to "text".
        pred_text_key (str): a string indicating which key of the data entries
            should be used to access the ASR predictions. Defaults to "pred_text".
        test_cases (list[dict]): an optional list of dicts containing test
            cases for checking that the processor makes the changes that we
            are expecting.
            The dicts must have a key ``input``, the value of which is a dictionary
            containing data which is our test's input manifest line, and a key
            ``output``, the value of which is a dictionary containing data which is
            the expected output manifest line.

    .. note::
        This class only supports one-to-one or one-to-none mappings.
        One-to-many is currently not supported.
    """

    def __init__(
        self,
        text_key: str = "text",
        pred_text_key: str = "pred_text",
        test_cases: Optional[List[Dict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.pred_text_key = pred_text_key
        self.test_cases = test_cases
        # need to convert to list to avoid errors in iteration over None
        if self.test_cases is None:
            self.test_cases = []

    def test(self):
        """Applies processing to "test_cases" and raises an error in case of mismatch."""
        for test_case in self.test_cases:
            generated_outputs = self.process_dataset_entry(test_case["input"].copy())
            # can only return 1 or zero entries
            if len(generated_outputs) == 1:
                generated_output = generated_outputs[0].data
            else:
                generated_output = None
            if generated_output != test_case["output"]:
                raise RuntimeError(
                    "Runtime test failed.\n"
                    f"Test input: {test_case['input']}\n"
                    f"Generated output: {generated_output}\n"
                    f"Expected output: {test_case['output']}"
                )

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
