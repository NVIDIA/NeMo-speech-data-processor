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


class WhisperHallucinationFeatures(BaseParallelProcessor):
    """
    A processor for detecting common hallucination patterns in ASR (automatic speech recognition) model outputs.
    
    This processor calculates simple features from the transcript text to help identify potential hallucinations,
    such as repeated word patterns, overly long words, or unnaturally high character rates.

    The following boolean features are computed and added to each manifest entry:
        - `hall_repeated_ngrams`: True if the fraction of unique words is below a threshold.
        - `hall_long_word`: True if a word is unusually long or significantly longer than the rest.
        - `hall_frequent_single_word`: True if the total character count per second is too low.

    Args:
        unique_words_threshold (float): Maximum share of unique words before flagging repeated n-grams. Default is 0.4.
        long_word_threshold (int): Minimum character length of a word to be considered 'too long'. Default is 25.
        long_word_rel_threshold (float): Relative difference between longest and second-longest word to flag. Default is 3.
        char_rate_threshold (float): Minimum average characters per second for a transcript. Default is 4.
        text_field (str): The key in the data entry containing the transcript. Default is 'text'.
        **kwargs: Additional arguments passed to BaseParallelProcessor.

    Returns:
        A manifest where each entry includes new boolean hallucination-related features.
    
        Example entry after processing::
            
            {
                "text": "<some transcript here>",
                "duration": 2.5,
                "hall_repeated_ngrams": False,
                "hall_long_word": True,
                "hall_frequent_single_word": False
            }
            
    """

    def __init__(
        self,
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
        self.char_rate_threshold = char_rate_threshold
        self.text_field = text_field

    def repeated_ngrams(self, words):
        # Calculate the share of unique words in the transcript
        unique_words_share = len(set(words)) / len(words)
        return unique_words_share <= self.unique_words_threshold

    def long_word(self, words):
        # Sort word lengths in ascending order
        word_lengths = sorted([len(word) for word in words])
        
        # Check if the longest word exceeds the absolute threshold
        if word_lengths[-1] >= self.long_word_threshold:
            return True

        # Check if the longest word is much longer than the second longest
        elif len(words) > 1:
            diff = (word_lengths[-1] - word_lengths[-2]) / word_lengths[-2]
            return diff >= self.long_word_rel_threshold

        return False

    def frequent_single_word(self, words, duration):
        # Calculate average character rate (characters per second)
        chars = sum(len(word) for word in words)
        char_rate = chars / duration
        return char_rate <= self.char_rate_threshold

    def process_dataset_entry(self, data_entry):
        # Extract the text field and tokenize into words
        text = data_entry[self.text_field]
        words = text.split()

        # Compute and assign hallucination features
        data_entry['hall_repeated_ngrams'] = self.repeated_ngrams(words)
        data_entry['hall_long_word'] = self.long_word(words)
        data_entry['hall_frequent_single_word'] = self.frequent_single_word(words, data_entry.get('duration'))

        return [DataEntry(data=data_entry)]
