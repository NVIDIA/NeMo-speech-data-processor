
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from pathlib import Path 

class WhisperHallucinationFeatures(BaseParallelProcessor):
    """
    Processor for counting the number of words in the text_key field saving the number in num_words_key.

    Args:
        text_key (str): The field containing the input text in the dataset.
        num_words_key (str): The field to store the number of words in the dataset.
        alphabet (str): Characters to be used to count words. Any other characters are substituted by whitespace and not take into account.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

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
        unique_words_share = len(set(words)) / len(words)

        if unique_words_share <= self.unique_words_threshold:
            return True

        return False

    def long_word(self, words):
        word_lengths = sorted([len(word) for word in words])

        if word_lengths[-1] >= self.long_word_threshold:
            return True
        
        elif len(words) > 1:

            diff = (word_lengths[-1] - word_lengths[-2]) / word_lengths[-2]

            if diff >= self.long_word_rel_threshold:
                return True 

        return False

    def frequent_single_word(self, words, duration):
        chars = sum([len(word) for word in words])

        char_rate = chars / duration
        if char_rate <= self.char_rate_threshold:
            return True 
        return False


    def process_dataset_entry(self, data_entry):
        text = data_entry[self.text_field]
        words = text.split()

        data_entry['hall_repeated_ngrams'] = self.repeated_ngrams(words)
        data_entry['hall_long_word'] = self.long_word(words)
        data_entry['hall_frequent_single_word'] = self.frequent_single_word(words, data_entry.get('duration'))

        return [DataEntry(data=data_entry)]