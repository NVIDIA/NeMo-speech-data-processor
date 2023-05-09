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

import re
import string
from typing import Dict

from nemo.utils import logging

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


def is_same(orig_word, norm_word):
    # word is the same, except last symbol, which could indicate punctuation
    if orig_word[-1] in string.punctuation and orig_word[:-1].lower() == norm_word.lower():
        return True, 1
    # word is the same, except last symbol, which could indicate punctuation
    # (but by mistake it's been put in norm text)
    if norm_word[-1] in string.punctuation and norm_word[:-1].lower() == orig_word.lower():
        return True, 0
    # word is the same, but casing could be different
    if orig_word.lower() == norm_word.lower():
        return True, 1

    return False, None


def restore_pc(orig_words, norm_words):
    # separate out any "¿" so they have a space either side
    orig_words = orig_words.replace("¿", " ¿ ")
    orig_words = " ".join(orig_words.split())
    norm_words = norm_words.replace("¿", " ¿ ")
    norm_words = " ".join(norm_words.split())

    orig_words_list = orig_words.split()
    norm_words_list = norm_words.split()

    # copy so not to corrupt
    # merging any commas and dots between numbers right away to simplify logic below
    orig_text = list([re.sub(r'(\d)[\.,](\d)', r"\1\2", word) for word in orig_words_list])
    norm_text = list(norm_words_list)
    # to simplify logic below, so that we can assume last word always matches
    orig_text.append("end_text")
    norm_text.append("end_text")

    idx_orig = 0
    idx_norm = 0
    merged_text = []
    while idx_orig < len(orig_text) and idx_norm < len(norm_text):
        same, is_orig = is_same(orig_text[idx_orig], norm_text[idx_norm])
        if same:
            merged_text.append(orig_text[idx_orig] if is_orig else norm_text[idx_norm])
            idx_orig += 1
            idx_norm += 1
            continue

        # add all "¿" 'words' in orig_text (which didnt have match in norm_text) to merged_text
        if orig_text[idx_orig] == "¿":
            merged_text.append("¿")
            idx_orig += 1
            continue

        # checking if first letter is a number, but the whole word is not - that happens
        # on typos like 37a which should really be 37 a. So fixing those
        # another case is for number + punctuation, like 2017, - handling separately
        # another case is for numbers separated by comma, like this "1,5". Those are spelled out
        # separately in normalized form, so just removing the comma here
        add_punct = ""
        if orig_text[idx_orig][0].isdigit() and not orig_text[idx_orig].isdigit():
            number, word = re.split('(\d+)', orig_text[idx_orig])[1:]
            orig_text[idx_orig] = number
            if word in string.punctuation:
                add_punct = word
            else:
                orig_text.insert(idx_orig + 1, word)

        # another annoying case is if typo ends with number like here "dell'11"
        # same logic, but need to go back to the first check, so doing "continue" below
        if orig_text[idx_orig][-1].isdigit() and not orig_text[idx_orig].isdigit():
            word, number = re.split('(\d+)', orig_text[idx_orig])[:-1]
            orig_text[idx_orig] = word
            orig_text.insert(idx_orig + 1, number)
            continue

        # word is different, but original is a number - take from normalized in this case until
        # get same word again (as number might be represented with multiple words)
        # also handling case for number + punctuation
        while orig_text[idx_orig].isdigit():
            idx_orig += 1

        while idx_norm < len(norm_text) and not is_same(orig_text[idx_orig], norm_text[idx_norm])[0]:
            merged_text.append(norm_text[idx_norm])
            idx_norm += 1

        # if there is any trailing punctuation from last digit, let's add it
        merged_text[-1] = merged_text[-1] + add_punct

    if idx_norm != len(norm_text):
        print(idx_orig, idx_norm, len(orig_text), len(norm_text), orig_text, norm_text, merged_text)
        raise RuntimeError("Something went wrong during merging")

    return " ".join(merged_text[:-1])  # removing end_text token


class NormalizeFromNonPCText(BaseParallelProcessor):
    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)

    def process_dataset_entry(self, data_entry: Dict):
        try:
            restored_norm_text = restore_pc(data_entry["text"], data_entry["provided_norm_text"])
            data_entry["text"] = restored_norm_text
            return [DataEntry(data=data_entry)]
        except:
            logging.warning(f"failed to restore normalization for text {data_entry['text']}. Dropping utterance")
            return [DataEntry(data=None)]


if __name__ == "__main__":
    # usage example
    try:
        # raw_text = ". So 623 Abc?"
        raw_text = "¿So, it's ¿ 62.3 Abc Abc?"
        norm_text = "so it's six two point three abc abc;"

        transcript_text = restore_pc(raw_text, norm_text)
        print(transcript_text)
    except:
        print("Failed to restore punctuation! Skipping utterance")
