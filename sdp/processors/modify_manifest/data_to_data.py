# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from typing import Dict, List, Optional

import soundfile
import torchaudio
from docx import Document
from sox import Transformer
from tqdm import tqdm
import json

from sdp.logging import logger
from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)
from sdp.utils.common import ffmpeg_convert
from sdp.utils.edit_spaces import add_start_end_spaces, remove_extra_spaces
from sdp.utils.get_diff import get_diff_with_subs_grouped
from sdp.utils.metrics_computation import (
    get_cer,
    get_charrate,
    get_wer,
    get_wmr,
    get_wordrate,
)


class GetAudioDuration(BaseParallelProcessor):
    """
    Processor that computes the duration of the file in ``audio_filepath_key`` (using soundfile)
    and saves the duration in ``duration_key``. If there is an error computing the duration,
    the value at ``duration_key`` will be updated with the value -1.0.

    Args:
        audio_filepath_key (str): Key to get path to wav file.
        duration_key (str): Key to put to audio duration.
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


class FfmpegConvert(BaseParallelProcessor):
    """
    Processor for converting video or audio files to audio using FFmpeg and updating the dataset with the path to the resampled audio.
    If ``id_key`` is not None, the output file path will be ``<resampled_audio_dir>/<id_key>.wav``.
    If ``id_key`` is None, the output file path will be ``<resampled_audio_dir>/<input file name without extension>.wav``.

    .. note:: ``id_key`` can be used to create subdirectories inside ``resampled_audio_dir`` (by using forward slashes ``/``).
        e.g. if ``id_key`` takes the form ``dir_name1/dir_name2/filename``, the output file path will be

        ``<resampled_audio_dir>/dir_name1/dirname2/filename.wav``.

    Args:
        converted_audio_dir (str): The directory to store the resampled audio files.
        input_file_key (str): The field in the dataset representing the path to the input video or audio files.
        output_file_key (str): The field in the dataset representing the path to the resampled audio files with ``output_format``. If ``id_key`` is None, the output file path will be ``<resampled_audio_dir>/<input file name without extension>.wav``.
        id_key (str): (Optional) The field in the dataset representing the unique ID or identifier for each entry. If ``id_key`` is not None, the output file path will be ``<resampled_audio_dir>/<id_key>.wav``. Defaults to None.
        output_format (str): (Optional) Format of the output audio files. Defaults to `wav`.
        target_samplerate (int): (Optional) The target sampling rate for the resampled audio. Defaults to 16000.
        target_nchannels (int): (Optional) The target number of channels for the resampled audio. Defaults to 1.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        converted_audio_dir: str,
        input_file_key: str,
        output_file_key: str,
        id_key: str = None,
        output_format: str = "wav",
        base_dir: str = None,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.converted_audio_dir = converted_audio_dir
        self.input_file_key = input_file_key
        self.output_file_key = output_file_key
        self.output_format = output_format
        self.id_key = id_key
        self.base_dir = base_dir
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def prepare(self):
        assert self.output_format == "wav", "Currently only wav format is supported"
        os.makedirs(self.converted_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        input_file = data_entry[self.input_file_key]
        if self.id_key:
            key = data_entry[self.id_key]
            os.makedirs(os.path.join(self.converted_audio_dir, *key.split("/")[:-1]), exist_ok=True)
        else:
            key = os.path.splitext(input_file)[0].split("/")[-1]

        if self.base_dir:
            new_dir = os.path.dirname(os.path.relpath(input_file, self.base_dir))
            os.makedirs(os.path.join(self.converted_audio_dir, new_dir), exist_ok=True)

            key = os.path.join(new_dir, key)

        audio_file = os.path.join(self.converted_audio_dir, key) + "." + self.output_format

        if not os.path.isfile(audio_file):
            ffmpeg_convert(input_file, audio_file, self.target_samplerate, self.target_nchannels)

        data_entry[self.output_file_key] = audio_file
        return [DataEntry(data=data_entry)]


class ReadTxtLines(BaseParallelProcessor):
    """
    The text file specified in source_filepath will be read, and each line in it will be added as a line in the output manifest,
    saved in the field text_key.

    Args:
        input_file_key (str): The key in the manifest containing the input txt file path .
        text_key (str): The key to store the read text lines in the manifest.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        input_file_key: str,
        text_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_file_key = input_file_key
        self.text_key = text_key

    def process_dataset_entry(self, data_entry):
        fname = data_entry[self.input_file_key]
        data_list = []
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = data_entry.copy()
                    data[self.text_key] = line
                    data_list.append(DataEntry(data=data))
        return data_list


class SoxConvert(BaseParallelProcessor):
    """Processor for Sox to convert audio files to specified format.

    Args:
        output_manifest_file (str): Path to the output manifest file.
        input_audio_file_key (str): Key in the manifest file that contains the path to the input audio file.
        output_audio_file_key (str): Key in the manifest file that contains the path to the output audio file.
        converted_audio_dir (str): Path to the directory where the converted audio files will be stored.
        output_format (str): Format of the output audio file.
        rate (int): Sample rate of the output audio file.
        channels (int): Number of channels of the output audio file.
        workspace_dir (str, Optional): Path to the workspace directory. Defaults to None.
    """

    def __init__(
        self,
        converted_audio_dir: str,
        input_audio_file_key: str = "audio_filepath",
        output_audio_file_key: str = "audio_filepath",
        output_format: str = "wav",
        rate: int = 16000,
        channels: int = 1,
        workspace_dir: Optional[str] = None,
        **kwargs,
    ):
        # Extract workspace_dir from kwargs to avoid passing it to BaseProcessor
        if "workspace_dir" in kwargs:
            workspace_dir = kwargs.pop("workspace_dir")
            
        super().__init__(**kwargs)
        self.input_audio_file_key = input_audio_file_key
        self.output_audio_file_key = output_audio_file_key
        self.converted_audio_dir = converted_audio_dir
        self.output_format = output_format
        self.workspace_dir = workspace_dir

        # Store the new parameters for later use:
        self.rate = rate
        self.channels = channels

    def prepare(self):
        # Debug print for workspace_dir
        logger.info(f"SoxConvert workspace_dir: {self.workspace_dir}")
        os.makedirs(self.converted_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        audio_path = data_entry[self.input_audio_file_key]
        
        # If workspace_dir is provided, join it with audio_path to get absolute path
        if self.workspace_dir is not None:
            full_audio_path = os.path.join(self.workspace_dir, audio_path)
        else:
            full_audio_path = audio_path
            
        # Debug print first file path
        if not hasattr(self, '_debug_printed'):
            logger.info(f"First audio_path from manifest: {audio_path}")
            logger.info(f"First full_audio_path: {full_audio_path}")
            logger.info(f"Path exists: {os.path.exists(full_audio_path)}")
            self._debug_printed = True

        key = os.path.splitext(audio_path)[0].split("/")[-1]
        converted_file = os.path.join(self.converted_audio_dir, key) + f".{self.output_format}"

        if not os.path.isfile(converted_file):
            transformer = Transformer()

            transformer.rate(self.rate)
            transformer.channels(self.channels)

            transformer.build(full_audio_path, converted_file)

        data_entry[self.output_audio_file_key] = converted_file
        return [DataEntry(data=data_entry)]


class CountNumWords(BaseParallelProcessor):
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
        text_key: str,
        num_words_key: str,
        alphabet: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.num_words_key = num_words_key
        self.pattern = re.compile("[^" + alphabet + "]")

    def process_dataset_entry(self, data_entry):
        text = data_entry[self.text_key]
        cleaned_string = self.pattern.sub("", text).strip()
        cleaned_string = re.sub("\\s+", " ", cleaned_string).strip()
        words = cleaned_string.split()
        num_words = len(words)
        data_entry[self.num_words_key] = num_words
        return [DataEntry(data=data_entry)]


class SplitLineBySentence(BaseParallelProcessor):
    """
    Processor for splitting lines of text into sentences based on a specified pattern.
    One line containing N sentences will be transformed into N lines containing one sentence.

    Args:
        text_key (str): The field containing the text lines in the dataset.
        end_pattern (str): The regular expression pattern to identify sentence boundaries.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.
    """

    def __init__(
        self,
        text_key: str,
        end_pattern: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.pattern = re.compile(end_pattern)

    def process_dataset_entry(self, data_entry):
        line = data_entry[self.text_key]
        data_list = []
        start = 0
        ends = [m.start() for m in self.pattern.finditer(line)]
        if ends:
            for end in ends:
                sent = line[start : end + 1].strip()
                # if sent and sent[0].isupper():
                data = data_entry.copy()
                data[self.text_key] = sent
                data_list.append(DataEntry(data=data))
                start = end + 1
            if start < len(line):
                pass
        else:
            data = data_entry.copy()
            data[self.text_key] = line.strip()
            data_list.append(DataEntry(data=data))
        return data_list


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
            orig_words, pred_words = (
                data_entry[self.text_key],
                data_entry[self.pred_text_key],
            )
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
            orig_words, pred_words = (
                data_entry[self.text_key],
                data_entry[self.pred_text_key],
            )
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


class NormalizeText(BaseParallelProcessor):
    """This processor applies text normalization (TN) to the text. I.e. converts text from written form into its verbalized form.
    E.g., "$123" is converted to "one hundred and twenty-three dollars."

    Args:
        input_text_key (str): the text field that will be the input to the Normalizer. Defaults to: text.
        input_language (str): language specifying the text normalization rules in ISO 639 Set 1 format. E.g., "en", "es", "it", etc.
            Defaults to: English.
        input_case (str): input text capitalization, set to `cased` if text contains capital letters.
            This flag affects normalization rules applied to the text. Note, `lower_cased` won't lower case input.
            Defaults to: cased.
        output_text_key (str): the text field that will be the output from the Normalizer.
            Defaults to: text.

    Returns:
        This processor normalizes the text in the `input_text_key` field and saves the normalized text in `output_text_key` field.

    Raises:
        `NotImplementedError`: when TN is not implemented for the requested language.
    """

    def __init__(
        self,
        input_text_key: str = "text",
        input_language: str = "en",
        input_case: str = "cased",
        output_text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_text_key = input_text_key
        self.output_text_key = output_text_key
        self.input_case = input_case
        self.input_language = input_language

    def prepare(self):
        from nemo_text_processing.text_normalization.normalize import Normalizer
        try:
            self.normalizer = Normalizer(input_case=self.input_case, lang=self.input_language)
        except NotImplementedError as e:
            logger.error("Failed to run text normalization: %s", repr(e))

    def process_dataset_entry(self, data_entry):
        data_entry[self.output_text_key] = self.normalizer.normalize(data_entry[self.input_text_key])
        return [DataEntry(data=data_entry)]


class InverseNormalizeText(BaseParallelProcessor):
    """This processor applies inverse text normalization (ITN) to the text. I.e. transforms spoken forms of numbers, dates, etc into their written equivalents.
    E.g., "one hundred and twenty-three dollars." is converted to "$123".

    Args:
        input_text_key (str): the text field that will be the input to the InverseNormalizer. Defaults to: text.
        input_language (str): language specifying the text normalization rules in ISO 639 Set 1 format. E.g., "en", "es", "it", etc.
            Defaults to: English.
        input_case (str): input text capitalization, set to `cased` if text contains capital letters.
            This flag affects normalization rules applied to the text. Note, `lower_cased` won't lower case input.
            Defaults to: cased.
        output_text_key (str): the text field that will be the output from the InverseNormalizer.
            Defaults to: text.

    Returns:
        This processor inverse normalizes the text in the `input_text_key` field and saves the inverse normalized text in `output_text_key` field.

    Raises:
        `NotImplementedError`: when ITN is not implemented for the requested language.
    """

    def __init__(
        self,
        input_text_key: str = "text",
        input_language: str = "en",
        input_case: str = "cased",
        output_text_key: str = "text",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_text_key = input_text_key
        self.output_text_key = output_text_key
        self.input_case = input_case
        self.input_language = input_language
        self.verbose = verbose

    def prepare(self):
        from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
        try:
            self.inverse_normalizer = InverseNormalizer(input_case=self.input_case, lang=self.input_language)
        except NotImplementedError as e:
            logger.error("Failed to run text inverse normalization: %s", repr(e))

    def process_dataset_entry(self, data_entry):
        data_entry[self.output_text_key] = self.inverse_normalizer.inverse_normalize(
            data_entry[self.input_text_key], verbose=self.verbose
        )
        return [DataEntry(data=data_entry)]


class CopyManifestData(BaseParallelProcessor):
    """This processor copies files specified in the manifest to a new location.

    It is useful for creating a consolidated dataset by gathering files from different sources
    into a single directory.

    Args:
        copy_path (str): The destination directory where files will be copied.
        source_filepath (str): The key in the manifest that contains the path to 
            the file to be copied. Default: "audio_path".

    Returns:
        The same data as in the input manifest, but the files referenced in the manifest
        will have been copied to the specified destination directory.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.modify_manifest.data_to_data.CopyManifestData
              input_manifest_file: ${workspace_dir}/dataset.json
              output_manifest_file: ${workspace_dir}/dataset_copied.json
              copy_path: ${workspace_dir}/consolidated_data
              source_filepath: "audio_filepath"
    """
    def __init__(
        self,
        copy_path: str,
        source_filepath: str = "audio_path",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = source_filepath
        self.copy_path = copy_path

    def prepare(self):
        os.makedirs(self.copy_path, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        fname = data_entry[self.input_field]

        dest_file_path = os.path.join(self.copy_path, os.path.basename(fname))
        shutil.copy(fname, dest_file_path)
        data_entry[self.input_field] = dest_file_path

        return [DataEntry(data=data_entry)]


class ReadDocxLines(BaseParallelProcessor):
    """
    Processor for reading text lines from a docx file and updating the manifest.

    Args:
        source_filepath (str): The field containing the file path in the manifest.
        text_key (str): The field to store the read text lines in the manifest.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        source_filepath: str,
        text_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = source_filepath
        self.output_field = text_key

    def process_dataset_entry(self, data_entry):
        fname = data_entry[self.input_field]

        # Skip hidden files and directories (e.g., .DS_Store, ._filename)
        if os.path.basename(fname).startswith('.'):
            logger.warning(f"Skipping hidden file: {fname}")
            return []

        data_list = []

        try:
            doc = Document(fname)
            for para in doc.paragraphs:
                line = para.text.strip()
                if line:
                    data = data_entry.copy()
                    data[self.output_field] = line
                    data_list.append(DataEntry(data=data))
        except Exception as e:
            logger.error(f"Error reading document {fname}: {e}")

        return data_list


class ExtractFromBrackets(BaseParallelProcessor):
    """
    A class for extracting text contained within specified bracket types from strings,
    handling nested brackets.

    Example Input:
        data_entry = {
            "text": "This is a [test] string with [multiple [nested] brackets]."
        }

    Example Output:
        [
            {
                "text": "test"
            },
            {
                "text": "multiple [nested] brackets"
            }
        ]

    Explanation:
        - It extracts "test" from the first occurrence of brackets.
        - It extracts "multiple [nested] brackets" from the second occurrence, handling nested brackets correctly.

    Attributes:
        brackets (List[str]): A list where each element is a pair of strings representing
                              the opening and closing brackets.
        text_key (str): The key in the input data from which to extract text, defaults to "text".
    """

    def __init__(
        self,
        brackets: List[str],
        text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.brackets = brackets
        self.text_key = text_key

    def extract_text_within_brackets(self, text, brackets):
        """
        Extracts text within the specified brackets, including handling nested brackets.

        Args:
            text (str): The string from which to extract text.
            brackets (tuple[str, str]): A tuple containing the opening and closing bracket.

        Returns:
            List[str]: A list of strings, each representing a segment of text found within
                       the outermost brackets, including any nested brackets content.
        """
        open_bracket, close_bracket = brackets
        depth = 0
        buffer = ""
        sentences = []

        for char in text:
            if char == open_bracket:
                if depth > 0:
                    buffer += char  # Add to buffer if already inside brackets
                depth += 1
            elif char == close_bracket:
                depth -= 1
                if depth == 0:  # Exiting outermost brackets
                    if buffer:
                        sentences.append(buffer)
                        buffer = ""  # Reset buffer for next possible extraction
                elif depth > 0:
                    buffer += char  # Still inside nested brackets, continue adding
            elif depth > 0:
                buffer += char  # Add characters inside brackets to buffer

        return sentences

    def process_dataset_entry(self, data_entry) -> List:
        data: list[dict] = []
        sentences = []
        text_in = data_entry[self.text_key]

        for bracket in self.brackets:
            sentences.extend(self.extract_text_within_brackets(text_in, bracket))

        for sentence in sentences:
            new_entry = data_entry.copy()
            new_entry[self.text_key] = sentence
            # new_entry["ORIGINAL TEXT"] = text_in  # for testing
            data.append(new_entry)

        data_list = []
        for data_point in data:
            data_list.append(DataEntry(data=data_point))

        return data_list


class GetWER(BaseParallelProcessor):
    """This processor calculates Word Error Rate (WER) between predicted text and ground truth text.

    It computes the WER for each entry in the manifest and adds the result as a new field.
    
    Args:
        text_key (str): Key for the ground truth text field in the manifest. Default: "text".
        pred_text_key (str): Key for the predicted text field in the manifest. Default: "pred_text".
    
    Returns:
        The same data as in the input manifest with an additional 'wer' field containing 
        the calculated Word Error Rate between the specified text fields.
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

    def process_dataset_entry(self, data_entry) -> List:
        data_entry['wer'] = get_wer(data_entry[self.text_key], data_entry[self.pred_text_key])
        return [DataEntry(data=data_entry)]


class MakeSentence(BaseParallelProcessor):
    """This processor formats text strings into proper sentences.

    It capitalizes the first character of the text (if enabled) and appends
    an end symbol if the text does not already end with punctuation.

    Args:
        text_key (str): The key in the manifest containing the text to be processed.
            Default: "text".
        end_symbol (str): The punctuation symbol to add at the end of the text if it
            doesn't already have one. Default: ":".
        make_uppercase (bool): Whether to capitalize the first character of the text.
            Default: True.

    Returns:
        The same data as in the input manifest with the text field modified to have
        proper sentence formatting.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.modify_manifest.data_to_data.MakeSentence
              input_manifest_file: ${workspace_dir}/dataset.json
              output_manifest_file: ${workspace_dir}/dataset_formatted.json
              text_key: "transcript"
              end_symbol: "."
              make_uppercase: true
    """
    def __init__(
        self,
        text_key: str = "text",
        end_symbol: str = ":",
        make_uppercase: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.make_uppercase = make_uppercase
        self.text_key = text_key
        self.end_symbol = end_symbol

    def process_dataset_entry(self, data_entry) -> List:
        if self.make_uppercase:
            data_entry[self.text_key] = data_entry[self.text_key][0].upper() + data_entry[self.text_key][1:]

        # Append end_symbol only if the text doesn't end with punctuation
        if data_entry[self.text_key][-1].isalpha():
            data_entry[self.text_key] += self.end_symbol
        return [DataEntry(data=data_entry)]


class ASRFileCheck(BaseProcessor):
    """This processor validates audio files in the manifest and identifies corrupted files.

    It attempts to load each audio file using the torchaudio library and moves corrupted
    files to a specified directory.

    Args:
        audio_filepath_key (str): The key in the manifest that contains the path to
            the audio file. Default: "audio_filepath".
        corrupted_audio_dir (str): The directory where corrupted audio files will be moved.
        workspace_dir (str, optional): The base directory for resolving relative paths.
            Default: None.

    Returns:
        A manifest with corrupted audio files removed.

    """
    def __init__(self, audio_filepath_key: str = "audio_filepath", corrupted_audio_dir: str = None, workspace_dir: str = None, **kwargs):
        """
        Constructs the necessary attributes for the ASRFileCheck class.

        Parameters:
        ----------
        audio_filepath_key : str, optional
            The key in the manifest entries used to retrieve the path to the audio file. Defaults to 'audio_filepath'.
        corrupted_audio_dir : str
            The directory where corrupted audio files will be moved. This is required.
        workspace_dir : str, optional
            The base directory where audio files are stored. If provided, audio file paths will be resolved
            relative to this directory. Defaults to None.
        """
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        
        if corrupted_audio_dir is None:
            raise ValueError("corrupted_audio_dir parameter is required. Please specify a directory to move corrupted files.")
        
        self.corrupted_audio_dir = corrupted_audio_dir
        self.workspace_dir = workspace_dir
        self.failed_files = []

    def process(self):
        """
        Check each file listed in the manifest to ensure it can be loaded with torchaudio.

        This method reads through the manifest file, attempts to load each audio file using torchaudio,
        and moves corrupted files. A new manifest file is created with only the valid entries.
        
        Specific errors handled:
        - FileNotFoundError: File doesn't exist
        - RuntimeError: File format issues or codec problems
        - Other exceptions: General issues with file loading
        """
        from sdp.logging import logger
        
        # Debug print to show workspace_dir
        logger.info(f"ASRFileCheck workspace_dir: {self.workspace_dir}")
        
        with open(self.input_manifest_file, 'r') as f:
            lines = f.readlines()

        entries = []
        total_lines = len(lines)

        # Ensure the corrupted files directory exists
        os.makedirs(self.corrupted_audio_dir, exist_ok=True)

        for idx in tqdm(range(total_lines), desc="Checking Audio Files"):
            line = lines[idx]
            entry = json.loads(line)
            audio_path = entry[self.audio_filepath_key]
            
            # Debug print first file path
            if idx == 0:
                logger.info(f"First audio_path from manifest: {audio_path}")
            
            # If workspace_dir is provided, join it with audio_path to get absolute path
            if self.workspace_dir is not None:
                full_audio_path = os.path.join(self.workspace_dir, audio_path)
            else:
                full_audio_path = audio_path
            
            # Debug print first full path
            if idx == 0:
                logger.info(f"First full_audio_path: {full_audio_path}")
                logger.info(f"Path exists: {os.path.exists(full_audio_path)}")
            
            try:
                # Attempt to load the audio file to check if it is corrupted
                torchaudio.load(full_audio_path)
                entries.append(entry)  # File is good, append to entries list
            except FileNotFoundError:
                logger.warning(f"File not found: {full_audio_path}")
                self.failed_files.append(audio_path)
            except RuntimeError as e:
                logger.warning(f"Audio format error in {audio_path}: {e}")
                self.failed_files.append(audio_path)
                
                # Move the corrupted audio file
                if os.path.exists(full_audio_path):
                    dest_path = os.path.join(self.corrupted_audio_dir, os.path.basename(audio_path))
                    os.rename(full_audio_path, dest_path)
                    logger.info(f"Moved corrupted file to: {dest_path}")
            except Exception as e:
                logger.warning(f"Unknown error loading {audio_path}: {e}")
                self.failed_files.append(audio_path)
                
                # Move the corrupted audio file
                if os.path.exists(full_audio_path):
                    dest_path = os.path.join(self.corrupted_audio_dir, os.path.basename(audio_path))
                    os.rename(full_audio_path, dest_path)
                    logger.info(f"Moved corrupted file to: {dest_path}")

        # Output non-corrupted entries to a new manifest file
        with open(self.output_manifest_file, 'w', encoding='utf-8') as f_out:
            for entry in entries:
                json.dump(entry, f_out, ensure_ascii=False)
                f_out.write("\n")

        if self.failed_files:
            logger.warning(f"Failed to process {len(self.failed_files)} files.")
            logger.debug(f"Failed files: {self.failed_files}")
