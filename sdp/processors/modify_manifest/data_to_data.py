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

import collections
import os
import re
from typing import Dict, List

import soundfile
from sox import Transformer

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.edit_spaces import add_start_end_spaces, remove_extra_spaces
from sdp.utils.get_diff import get_diff_with_subs_grouped


class GetAudioDuration(BaseParallelProcessor):
    """
    Processor to count audio duration using audio file path from audio_filepath_key

    Args:
        audio_filepath_key (str): where to get path to wav file.
        duration_key (str): where to put to audio duration.
    Returns:
        All the same fields as in the input manifest plus duration_key
    """

    def __init__(
        self,
        audio_filepath_key: str,
        duration_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        self.duration_key = duration_key

    def process_dataset_entry(self, data_entry):
        audio_filepath = data_entry[self.audio_filepath_key]
        try:
            data, samplerate = soundfile.read(audio_filepath)
            data_entry[self.duration_key] = data.shape[0] / samplerate
        except Exception as e:
            logger.warning(str(e) + " file: " + audio_filepath)
            data_entry[self.duration_key] = -1.0
        return [DataEntry(data=data_entry)]


class SoxConvert(BaseParallelProcessor):
    """
    Processor for converting audio files from one format to another using Sox,
    and updating the dataset with the path to the converted audio files.

    Args:

        converted_audio_dir (str): Directory to store the converted audio files.
        input_audio_file_key (str): Field in the dataset representing the path to input audio files.
        output_audio_file_key (str): Field to store the path to the converted audio files in the dataset.
        output_format (str): Format of the output audio files (e.g., 'wav', 'mp3').
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.
    """

    def __init__(
        self,
        converted_audio_dir: str,
        input_audio_file_key: str,
        output_audio_file_key: str,
        output_format: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_audio_file_key = input_audio_file_key
        self.output_audio_file_key = output_audio_file_key
        self.converted_audio_dir = converted_audio_dir
        self.output_format = output_format

    def prepare(self):
        os.makedirs(self.converted_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        audio_file = data_entry[self.input_audio_file_key]

        key = os.path.splitext(audio_file)[0].split("/")[-1]
        converted_file = os.path.join(self.converted_audio_dir, key) + f".{self.output_format}"

        if not os.path.isfile(converted_file):
            transformer = Transformer()
            transformer.build(audio_file, converted_file)

        data_entry[self.output_audio_file_key] = converted_file
        return [DataEntry(data=data_entry)]


class InsIfASRInsertion(BaseParallelProcessor):
    """Processor that adds substrings to transcription if they are present in ASR predictions.

    Will insert substrings into ``data[self.text_key]`` if it is
    present at that location in ``data[self.pred_text_key]``.
    It is useful if words are systematically missing from ground truth
    transcriptions.

    Args:
        insert_words (list[str]): list of strings that will be inserted
            into ``data[self.text_key]`` if there is an insertion (containing
            only that string) in ``data[self.pred_text_key]``.
        text_key (str): a string indicating which key of the data entries
            should be used to find the utterance transcript. Defaults to "text".
        pred_text_key (str): a string indicating which key of the data entries
            should be used to access the ASR predictions. Defaults to "pred_text".

            .. note::
                Because this processor looks for an exact match in the insertion,
                we recommend including variations with different spaces in
                ``insert_words``, e.g. ``[' nemo', 'nemo ', ' nemo ']``.

    Returns:
         The same data as in the input manifest with ``<text_key>`` field changed.
    """

    def __init__(
        self,
        insert_words: List[str],
        text_key: str = "text",
        pred_text_key: str = "pred_text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.insert_words = insert_words
        self.text_key = text_key
        self.pred_text_key = pred_text_key

    def process_dataset_entry(self, data_entry) -> List:
        insert_word_counter = collections.defaultdict(int)
        for insert_word in self.insert_words:
            if not insert_word in data_entry[self.pred_text_key]:
                break
            orig_words, pred_words = data_entry[self.text_key], data_entry[self.pred_text_key]
            diff = get_diff_with_subs_grouped(orig_words, pred_words)

            if len(diff) > 0:  # ie if there are differences between text and pred_text
                new_sent = ""

                for diff_entry in diff:
                    if diff_entry[0] == 0:  # no change
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == -1:  # deletion in original string
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == 1:  # insertion in original string
                        if diff_entry[1] == insert_word:
                            new_sent += insert_word
                            insert_word_counter[insert_word] += 1

                    elif isinstance(diff_entry, tuple):  # i.e. diff is a substitution
                        new_sent += diff_entry[0][1]
                    else:
                        raise ValueError(f"unexpected item in diff_entry: {diff_entry}")

                new_sent = " ".join(new_sent.split())  # remove any extra spaces
                data_entry[self.text_key] = new_sent

        return [DataEntry(data=data_entry, metrics=insert_word_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logger.info("Num of words that were inserted")
        for word, count in total_counter.items():
            logger.info(f"{word} {count}")
        super().finalize(metrics)


class SubIfASRSubstitution(BaseParallelProcessor):
    """Processor that substitutes substrings to transcription if they are present in ASR predictions.

    Will convert a substring in ``data[self.text_key]`` to a
    substring in ``data[self.pred_text_key]`` if both are located in the
    same place (ie are part of a 'substitution' operation) and if the substrings
    correspond to key-value pairs in ``sub_words``.
    This is useful if words are systematically incorrect in ground truth
    transcriptions.

    Before starting to look for substitution, this processor adds spaces at the beginning and end of
    ``data[self.text_key]`` and ``data[self.pred_text_key]``, to ensure that an argument like
    ``sub_words = {"nmo ": "nemo "}`` would cause a substitution to be made even if the original
    ``data[self.text_key]`` ends with ``"nmo"`` and ``data[self.pred_text_key]`` ends with ``"nemo"``.

    Args:
        sub_words (dict): dictionary where a key is a string that might be in
            ``data[self.text_key]`` and the value is the string that might
            be in ``data[self.pred_text_key]``. If both are located in the same
            place (i.e. are part of a 'substitution' operation)
            then the key string will be converted to the value string
            in ``data[self.text_key]``.
        text_key (str): a string indicating which key of the data entries
            should be used to find the utterance transcript. Defaults to "text".
        pred_text_key (str): a string indicating which key of the data entries
            should be used to access the ASR predictions. Defaults to "pred_text".

            .. note::
                This processor looks for exact string matches of substitutions,
                so you may need to be careful with spaces in ``sub_words``. E.g.
                it is recommended to do ``sub_words = {"nmo ": "nemo "}``
                instead of ``sub_words = {"nmo" : "nemo"}``.

    Returns:
         The same data as in the input manifest with ``<text_key>`` field changed.
    """

    def __init__(
        self,
        sub_words: Dict,
        text_key: str = "text",
        pred_text_key: str = "pred_text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sub_words = sub_words
        self.text_key = text_key
        self.pred_text_key = pred_text_key

    def process_dataset_entry(self, data_entry) -> List:
        sub_word_counter = collections.defaultdict(int)
        data_entry[self.text_key] = add_start_end_spaces(data_entry[self.text_key])
        data_entry[self.pred_text_key] = add_start_end_spaces(data_entry[self.pred_text_key])
        for original_word, new_word in self.sub_words.items():
            if not original_word in data_entry[self.text_key]:
                break
            orig_words, pred_words = data_entry[self.text_key], data_entry[self.pred_text_key]
            diff = get_diff_with_subs_grouped(orig_words, pred_words)

            if len(diff) > 0:  # ie if there are differences between text and pred_text
                new_sent = ""

                for diff_entry in diff:
                    if diff_entry[0] == 0:  # no change
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == -1:  # deletion in original string
                        new_sent += diff_entry[1]

                    elif diff_entry[0] == 1:  # insertion in original string
                        # don't make changes
                        pass

                    elif isinstance(diff_entry, tuple):  # substitution
                        if diff_entry[0][1] == original_word and diff_entry[1][1] == new_word:
                            # ie. substitution is one we want to use to change the original text
                            new_sent += new_word
                            sub_word_counter[original_word] += 1

                        else:
                            # ie. substitution is one we want to ignore
                            new_sent += diff_entry[0][1]
                    else:
                        raise ValueError(f"unexpected item in diff_entry: {diff_entry}")

                new_sent = add_start_end_spaces(new_sent)
                data_entry[self.text_key] = new_sent

        data_entry[self.text_key] = remove_extra_spaces(data_entry[self.text_key])
        data_entry[self.pred_text_key] = remove_extra_spaces(data_entry[self.pred_text_key])

        return [DataEntry(data=data_entry, metrics=sub_word_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logger.info("Num of words that were substituted")
        for word, count in total_counter.items():
            logger.info(f"{word} {count}")
        super().finalize(metrics)


# TODO: replace with generic regex


class SubMakeLowercase(BaseParallelProcessor):
    """Processor to convert text to lowercase.

    text_key (str): a string indicating which key of the data entries
        should be used to find the utterance transcript. Defaults to "text".

    Returns:
        The same data as in the input manifest with ``<text_key>`` field changed.
    """

    def __init__(
        self,
        text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key

    def process_dataset_entry(self, data_entry) -> List:
        data_entry[self.text_key] = data_entry[self.text_key].lower()
        return [DataEntry(data=data_entry)]

    def finalize(self, metrics):
        logger.info("Made all letters lowercase")
        super().finalize(metrics)


class SubRegex(BaseParallelProcessor):
    """Converts a regex match to a string, as defined by key-value pairs in ``regex_to_sub``.

    Before applying regex changes, we will add a space
    character to the beginning and end of the ``text`` and ``pred_text``
    keys for each data entry. After the the regex changes,
    the extra spaces are removed. This includes the spaces in the beginning
    and end of the text, as well as any double spaces ``"  "``.

    Args:
        regex_params_list (list[dict]): list of dicts.
            Each dict must contain a ``pattern`` and a ``repl`` key,
            and optionally a ``count`` key (by default, ``count`` will be 0).
            This processor will go through the list in order, and apply a ``re.sub`` operation on
            the input text in ``data_entry[self.text_key]``, feeding in the specified ``pattern``, ``repl``
            and ``count`` parameters to ``re.sub``.
        text_key (str): a string indicating which key of the data entries
            should be used to find the utterance transcript. Defaults to "text".

    Returns:
         The same data as in the input manifest with ``<text_key>`` field changed.
    """

    def __init__(
        self,
        regex_params_list: List[Dict],
        text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.regex_params_list = regex_params_list
        self.text_key = text_key

        # verify all dicts in regex_params_list have "pattern" and "repl" keys
        for regex_params_dict in self.regex_params_list:
            if not "pattern" in regex_params_dict.keys():
                raise ValueError(
                    f"Need to have key 'pattern' in all entries of `regex_params_list`: {self.regex_params_list}"
                )
            if not "repl" in regex_params_dict.keys():
                raise ValueError(
                    f"Need to have key 'repl' in all entries of `regex_params_list`: {self.regex_params_list}"
                )

    def process_dataset_entry(self, data_entry) -> List:
        """Replaces each found regex match with a given string."""
        replace_word_counter = collections.defaultdict(int)

        text_in = data_entry[self.text_key]

        text_in = add_start_end_spaces(text_in)
        for regex_params in self.regex_params_list:
            text_out = re.sub(
                pattern=regex_params["pattern"],
                repl=regex_params["repl"],
                string=text_in,
                # note: this count param is the maximum number of pattern occurrences to be replaced.
                count=regex_params.get("count", 0),
            )

            if text_in != text_out:
                replace_word_counter[regex_params["pattern"]] += 1
            text_in = text_out

        text_out = remove_extra_spaces(text_out)

        data_entry[self.text_key] = text_out

        return [DataEntry(data=data_entry, metrics=replace_word_counter)]

    def finalize(self, metrics):
        """Reports how many substitutions were made for each pattern."""
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logger.info("Number of utterances which applied substitutions for the following patterns:")
        total_counter_sorted = dict(sorted(total_counter.items(), key=lambda x: x[1], reverse=True))
        for word, count in total_counter_sorted.items():
            logger.info(f"{word} {count}")
        super().finalize(metrics)
