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

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class DetectWhisperHallucinationFeatures(BaseParallelProcessor):
    """
    Computes hallucination-related features for ASR model outputs (e.g., Whisper transcripts).

    This processor analyzes the transcript text and flags common hallucination patterns by computing
    boolean features such as:
    - Repeated or low-diversity n-grams (`hall_repeated_ngrams`)
    - Unusually long or disproportionately long words (`hall_long_word`)
    - Matches with known hallucinated phrases (`hall_frequent_single_word`)

    It appends these features to each entry in the manifest for downstream filtering or analysis.

    Args:
        common_hall_file (str): Path to a file with known hallucinated phrases, one per line.
        unique_words_threshold (float): Maximum allowed share of unique words before marking as repeated. Default is 0.4.
        long_word_threshold (int): Minimum character length for a word to be considered "long". Default is 25.
        long_word_rel_threshold (float): Relative length ratio between the longest and second-longest word. Default is 3.
        char_rate_threshold (float): [Unused in current logic, retained for compatibility]. Default is 4.
        text_field (str): Key in the data entry that contains the transcript. Default is 'text'.
        **kwargs: Additional keyword arguments passed to `BaseParallelProcessor`.

    Returns:
        A manifest with additional boolean fields for hallucination detection.

    Example entry after processing:
        {
            "text": "hello hello hello",
            "duration": 2.0,
            "hall_repeated_ngrams": True,
            "hall_long_word": False,
            "hall_frequent_single_word": False
        }
    """

    def __init__(
        self,
        common_hall_file,
        unique_words_threshold=0.4,
        long_word_threshold=25,
        long_word_rel_threshold=3,
        char_rate_threshold=4,
        text_field='text',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unique_words_threshold = unique_words_threshold
        self.long_word_threshold = long_word_threshold
        self.long_word_rel_threshold = long_word_rel_threshold
        self.char_rate_threshold = char_rate_threshold  # Currently unused
        self.text_field = text_field

        # Load common hallucination phrases into memory
        with open(common_hall_file, 'r') as f:
            self.common_hall_phrases = [line.strip() for line in f]

    def repeated_ngrams(self, words):
        """
        Flags entries with low lexical diversity (i.e., repeated n-grams).

        Returns True if the fraction of unique words is below the threshold.
        """
        unique_words_share = len(set(words)) / len(words)
        return unique_words_share <= self.unique_words_threshold

    def long_word(self, words):
        """
        Detects unusually long words or sharp differences in word lengths.

        Returns True if the longest word is above the absolute threshold or much longer
        than the second-longest word.
        """
        word_lengths = sorted([len(word) for word in words])

        if word_lengths[-1] >= self.long_word_threshold:
            return True

        if len(words) > 1:
            diff = (word_lengths[-1] - word_lengths[-2]) / word_lengths[-2]
            return diff >= self.long_word_rel_threshold

        return False

    def frequent_single_word(self, text):
        """
        Checks if the cleaned transcript matches any known hallucinated phrase.
        """
        cleaned_text = text.strip().replace('.', '').replace('?', '').replace('!', '')
        return cleaned_text in self.common_hall_phrases

    def process_dataset_entry(self, data_entry):
        """
        Processes a single manifest entry and appends hallucination features.
        """
        text = data_entry[self.text_field]
        words = text.split()

        # Compute hallucination indicators
        data_entry['hall_repeated_ngrams'] = self.repeated_ngrams(words)
        data_entry['hall_long_word'] = self.long_word(words)
        data_entry['hall_frequent_single_word'] = self.frequent_single_word(text)

        return [DataEntry(data=data_entry)]
