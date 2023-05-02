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

import string
import re


def abbreviations(text):
    text = (
        text.replace("Cap'n", "Captain")
        .replace("cap'n", "captain")
        .replace("o'shot", "o shot")
        .replace("o' shot", "o shot")
        .replace("on'y", "only")
        .replace("on' y", "only")
        .replace(" 'a ", " a ")
        .replace(" 'em ", " em ")
        .replace("gen'leman", "gentleman")
    )
    return text


def process(text):
    text = (
        text.replace("www.gutenberg.org", "www dot gutenberg dot org")
        .replace(".txt", "dot txt")
        .replace(".zip", "dot zip")
    )

    text = (
        text.replace("’", "'")
        .replace("_", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("…", "...")
        .replace("»", '"')
        .replace("«", '"')
        .replace("\\", "")
        .replace("”", '"')
        .replace("„", '"')
        .replace("´", "'")
        .replace("-- --", "--")
        .replace("--", " -- ")
        .replace(". . .", "...")
        .replace("’", "'")
        .replace("“", '"')
        .replace("“", '"')
        .replace("‘", "'")
        .replace("_", " ")
        .replace("*", " ")
        .replace("—", "-")
        .replace("- -", "--")
        .replace("•", " ")
        .replace("^", " ")
        .replace(">", " ")
        .replace("■", " ")
        .replace("/", " ")
        .replace("––––", "...")
        .replace("W⸺", "W")
        .replace("`", "'")
        .replace("<", " ")
        .replace("{", " ")
        .replace("Good-night", "Good night")
        .replace("good-night", "good night")
        .replace("good-bye", "goodbye")
        .replace("Good-bye", "Goodbye")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace("!!", "!")
        .replace("--", "-")
        .replace("“", '"')
        .replace(", , ", ", ")
        .replace("=", " ")
        .replace("l,000", "1,000")
        .replace("–", "-")
    )
    # remove dash in between the words
    text = re.sub(r"([A-za-z0-9]+)(-)([A-Za-z0-9]+)", r"\g<1> \g<3>", text)
    text = re.sub(r"([A-za-z0-9]+)(\.)([A-Za-z]+)", r"\g<1>\g<2> \g<3>", text)
    text = re.sub(r"([A-za-z]+)(\.)([A-Za-z0-9]+)", r"\g<1>\g<2> \g<3>", text)

    # # remove text inside square brackets
    # text = re.sub(r"(\[.*?\])", " ", text)

    def __fix_space(text):
        # remove commas between digits
        text = re.sub(r"([0-9]+)(,)(\d\d\d)", r"\g<1>\g<3>", text)
        text = re.sub(r"([A-Za-z]+)(,)([A-Za-z0-9]+)", r"\g<1>\g<2> \g<3>", text)
        return text

    for _ in range(3):
        text = __fix_space(text)

    text = re.sub(r" +", " ", text)

    # make sure the text starts with an alpha
    start_idx = 0
    while not text[start_idx].isalpha():
        start_idx += 1

    end_text = "END OF THIS PROJECT GUTENBERG"
    end_idx = len(text)
    if end_text in text:
        end_idx = text.find(end_text)

    end_text = "End of the Project Gutenberg"
    if end_text in text:
        end_idx = text.find(end_text)

    return text[start_idx:end_idx]


def read_text(text_f):
    with open(text_f, "r") as f:
        text = f.read()
    return text


def remove_punctuation(text: str, remove_spaces=True, do_lower=True, exclude=None, remove_accents=False):
    all_punct_marks = string.punctuation + "¿¡⸘"  # TODO: document these changes

    if exclude is not None:
        for p in exclude:
            all_punct_marks = all_punct_marks.replace(p, "")

        # a weird bug where commas is getting deleted when dash is present in the list of punct marks
        all_punct_marks = all_punct_marks.replace("-", "")
    text = re.sub("[" + all_punct_marks + "]", " ", text)

    if exclude and "-" not in exclude:
        text = text.replace("-", " ")

    text = re.sub(r" +", " ", text)
    if remove_spaces:
        text = text.replace(" ", "").replace("\u00A0", "").strip()

    if do_lower:
        text = text.lower()

    if remove_accents:
        text = text.replace("á", "a")
        text = text.replace("é", "e")
        text = text.replace("í", "i")
        text = text.replace("ó", "o")
        text = text.replace("ú", "u")
        text = text.replace("à", "a")
        text = text.replace("è", "e")
        text = text.replace("ù", "u")
        text = text.replace("â", "a")
        text = text.replace("ê", "e")
        text = text.replace("î", "i")
        text = text.replace("ô", "o")
        text = text.replace("û", "u")

    return text.strip()
