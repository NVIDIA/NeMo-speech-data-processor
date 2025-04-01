from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from pathlib import Path 
import string
from sdp.utils.metrics_computation import get_cer
from sdp.logging import logger


class QwenGenerationFiltering(BaseParallelProcessor):

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
        if "<|endoftext|>" in generation:
            generation = generation.split("<|endoftext|>")[0]

        if "Input transcript:" in generation:
            generation = generation.replace("Input transcript:", "")

        if "Output:" in generation:
            generation = generation.replace("Output:", "")

        if "\n" in generation:
            generation = generation.replace("\n", "")

        return generation

    def maybe_replace_with_text(self, generation, text):
        chars = generation.replace(' ', '')
        total_chars = len(chars)

        if not total_chars:
            return text, 1

        uppercase_count = sum(1 for char in chars if char.isupper())

        if uppercase_count / total_chars > self.upper_case_threshold:
            return text, 1

        normalized_text = text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        normalized_generation = generation.lower().translate(str.maketrans('', '', string.punctuation)).strip()

        if not normalized_text:
            return text, 1

        cer = get_cer(normalized_text, normalized_generation)

        if cer > self.cer_threshold:
            return text, 1

        return generation, 0

    def process_dataset_entry(self, data_entry):
        text = data_entry[self.text_field]
        generation = data_entry[self.generation_field]

        generation = self.clean(generation)
        maybe_replaced_generation, replaced = self.maybe_replace_with_text(generation, text)

        data_entry[self.generation_field] = maybe_replaced_generation.strip()

        return [DataEntry(data=data_entry, metrics=replaced)]

    def finalize(self, metrics):
        logger.info(
            f"Num of utterances that were replaced by text: {sum(metrics)}"
        )
        super().finalize(metrics)