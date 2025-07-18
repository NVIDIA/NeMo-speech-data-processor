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

import string

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.metrics_computation import get_cer


class CleanQwenGeneration(BaseParallelProcessor):
    """
    A processor that filters and post-processes model generations, replacing them with
    reference text if they are considered low quality based on character error rate (CER)
    and uppercase letter proportion.

    This processor is typically used after running a generation model (e.g., Qwen) to clean
    up outputs and ensure alignment with reference transcriptions.

    Args:
        cer_threshold (float): Maximum allowable character error rate (CER) between the
            normalized generation and reference text. If exceeded, the generation is
            replaced by the reference.
        upper_case_threshold (float): Threshold for the proportion of uppercase letters in
            the generation. If the ratio exceeds this value, the generation is replaced.
        generation_field (str): Key in the input manifest for the model-generated text.
        text_field (str): Key in the input manifest for the reference (target) text.
        **kwargs: Additional arguments passed to the `BaseParallelProcessor`.

    Returns:
        A manifest where each entry contains the cleaned generation in the specified
        generation field. If a replacement occurred, it is recorded in the metrics.

    Metrics:
        - 1 if the generation was replaced with the reference text.
        - 0 if the generation was left as-is.
    """

    def __init__(
        self,
        cer_threshold=10,
        upper_case_threshold=0.6,
        generation_field='generation',
        text_field='text',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cer_threshold = cer_threshold
        self.upper_case_threshold = upper_case_threshold
        self.generation_field = generation_field
        self.text_field = text_field

    def clean(self, generation):
        """Remove template prompts and special tokens from model generation."""
        if "<|endoftext|>" in generation:
            generation = generation.split("<|endoftext|>")[0]

        if "Input transcript:" in generation:
            generation = generation.replace("Input transcript:", "")

        if "Output:" in generation:
            generation = generation.replace("Output:", "")

        if "Output transcript:" in generation:
            generation = generation.replace("Output transcript:", "")

        if "\n" in generation:
            generation = generation.replace("\n", "")

        return generation

    def maybe_replace_with_text(self, generation, text):
        """
        Determine if generation should be replaced with reference text based on
        CER and uppercase ratio.
        """
        chars = generation.replace(' ', '')
        total_chars = len(chars)

        # Replace if generation is empty
        if not total_chars:
            return text, 1

        # Replace if excessive capitalization
        uppercase_count = sum(1 for char in chars if char.isupper())
        if uppercase_count / total_chars > self.upper_case_threshold:
            return text, 1

        # Normalize both strings for CER comparison
        normalized_text = text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        normalized_generation = generation.lower().translate(str.maketrans('', '', string.punctuation)).strip()

        if not normalized_text:
            return text, 1

        cer = get_cer(normalized_text, normalized_generation)

        if cer > self.cer_threshold:
            return text, 1

        return generation, 0

    def process_dataset_entry(self, data_entry):
        """Process a single entry from the manifest: clean and validate generation."""
        text = data_entry[self.text_field]
        generation = data_entry[self.generation_field]

        generation = self.clean(generation)
        maybe_replaced_generation, replaced = self.maybe_replace_with_text(generation, text)

        data_entry[self.generation_field] = maybe_replaced_generation.strip()

        return [DataEntry(data=data_entry, metrics=replaced)]

    def finalize(self, metrics):
        """Log the total number of replaced generations."""
        logger.info(
            f"Num of utterances that were replaced by text: {sum(metrics)}"
        )
        super().finalize(metrics)