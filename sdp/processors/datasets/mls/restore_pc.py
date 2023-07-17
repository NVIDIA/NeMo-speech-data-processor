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
import json
import os
import re
import string
import sys
from glob import glob
from pathlib import Path
from typing import Optional

import regex
from joblib import Parallel, delayed
from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import download_file, extract_archive

sys.setrecursionlimit(1000000)

NA = "n/a"
MLS_TEXT_URL = "https://dl.fbaipublicfiles.com/mls/lv_text.tar.gz"


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
    text = re.sub(r"([A-Za-z0-9]+)(-)([A-Za-z0-9]+)", r"\g<1> \g<3>", text)
    text = re.sub(r"([A-Za-z0-9]+)(\.)([A-Za-z]+)", r"\g<1>\g<2> \g<3>", text)
    text = re.sub(r"([A-Za-z]+)(\.)([A-Za-z0-9]+)", r"\g<1>\g<2> \g<3>", text)

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
    all_punct_marks = string.punctuation + "¿¡⸘"

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


def recover_lines(manifest, processed_text, output_dir, restored_text_field):
    manifest_recovered = f"{output_dir}/{os.path.basename(manifest)}"
    if os.path.exists(manifest_recovered):
        return

    lines = []
    with open(manifest, "r") as f:
        for line in f:
            line = json.loads(line)
            lines.append(line["text"])

    logger.debug(f"processing {manifest}")
    logger.debug(f"processing - {len(lines)} lines")

    last_found_start_idx = 0
    recovered_lines = {}

    for idx, cur_line in enumerate(lines):
        stop_search_for_line = False
        cur_word_idx = 0
        cur_line = abbreviations(cur_line)
        cur_line = cur_line.split()
        end_match_found = False

        while not stop_search_for_line:
            cur_word = cur_line[cur_word_idx]

            pattern = cur_word
            max_start_match_len = min(4, len(cur_line))
            for i in range(1, max_start_match_len):
                pattern += f"[^A-Za-z]+{cur_line[i]}"

            pattern = re.compile(pattern)

            for i, m in enumerate(pattern.finditer(processed_text[last_found_start_idx:].lower())):
                if end_match_found:
                    break
                match_idx = m.start() + last_found_start_idx
                processed_text_list = processed_text[match_idx:].split()
                raw_text_pointer = (
                    len(cur_line) - 3
                )  # added in case some dash separated words and split into multiple words in the cur_line
                stop_end_search = False
                right_offset = 20
                while not end_match_found and raw_text_pointer <= len(processed_text_list) and not stop_end_search:
                    if cur_line[-1].replace("'", "") == remove_punctuation(
                        processed_text_list[raw_text_pointer - 1],
                        remove_spaces=True,
                        do_lower=True,
                        remove_accents=False,
                    ):
                        # processed text could contain apostrophes that are parts of quotes, let's remove them from the processed text as well
                        if "'" not in cur_line[-1] and "'" in processed_text_list[raw_text_pointer - 1]:
                            processed_text_list[raw_text_pointer - 1] = processed_text_list[
                                raw_text_pointer - 1
                            ].replace("'", "")
                        recovered_line = " ".join(processed_text_list[:raw_text_pointer])
                        if not is_valid(" ".join(cur_line), recovered_line):
                            raw_text_pointer += 1
                        else:
                            recovered_lines[idx] = recovered_line
                            end_match_found = True
                            raw_text_pointer += 1
                            stop_search_for_line = True
                            last_found_start_idx = raw_text_pointer

                    else:
                        raw_text_pointer += 1
                        if raw_text_pointer > (len(cur_line) + right_offset):
                            stop_end_search = True

            if not end_match_found:
                stop_search_for_line = True

    logger.debug(
        f"recovered {len(recovered_lines)} lines out of {len(lines)} -- {round(len(recovered_lines)/len(lines)*100, 2)}% -- {os.path.basename(manifest)}"
    )

    with open(manifest_recovered, "w") as f_out, open(manifest, "r") as f_in:
        for idx, line in enumerate(f_in):
            line = json.loads(line)
            if idx in recovered_lines:
                line[restored_text_field] = recovered_lines[idx]
            else:
                line[restored_text_field] = NA
            f_out.write(json.dumps(line, ensure_ascii=False) + "\n")


def split_text_into_sentences(text: str):
    """
    Split text into sentences.

    Args:
        text: text

    Returns list of sentences
    """
    # TODO: should this be filled up and exposed as a parameter?
    lower_case_unicode = ""
    upper_case_unicode = ""

    # end of quoted speech - to be able to split sentences by full stop
    text = re.sub(r"([\.\?\!])([\"\'])", r"\g<2>\g<1> ", text)

    # remove extra space
    text = re.sub(r" +", " ", text)

    # remove space in the middle of the lower case abbreviation to avoid splitting into separate sentences
    matches = re.findall(rf"[a-z{lower_case_unicode}]\.\s[a-z{lower_case_unicode}]\.", text)
    for match in matches:
        text = text.replace(match, match.replace(". ", "."))

    # Read and split transcript by utterance (roughly, sentences)
    split_pattern = (
        rf"(?<!\w\.\w.)(?<![A-Z{upper_case_unicode}][a-z{lower_case_unicode}]+\.)"
        rf"(?<![A-Z{upper_case_unicode}]\.)(?<=\.|\?|\!|\.”|\?”\!”)\s(?![0-9]+[a-z]*\.)"
    )
    sentences = regex.split(split_pattern, text)
    return sentences


def normalize_text(text_f: str, normalizer: Optional['Normalizer'] = None):
    """
    Pre-process and normalized text_f file.

    Args:
        text_f: path to .txt file to normalize
        normalizer:
    """
    raw_text = read_text(text_f)
    processed_text = abbreviations(process(raw_text))
    if normalizer is not None:
        processed_text_list = normalizer.split_text_into_sentences(processed_text)
    else:
        processed_text_list = split_text_into_sentences(processed_text)
    processed_text_list_merged = []
    last_segment = ""
    max_len = 7500
    for i, text in enumerate(processed_text_list):
        if len(last_segment) < max_len:
            last_segment += " " + text
        else:
            processed_text_list_merged.append(last_segment.strip())
            last_segment = ""

        if i == len(processed_text_list) - 1 and len(last_segment) > 0:
            processed_text_list_merged.append(last_segment.strip())

    for i, text in enumerate(tqdm(processed_text_list_merged)):
        if normalizer is not None:
            processed_text_list_merged[i] = normalizer.normalize(
                text=text, punct_post_process=True, punct_pre_process=True
            )
        else:
            processed_text_list_merged[i] = re.sub(r"\d", r"", processed_text_list_merged[i])
    processed_text = " ".join(processed_text_list_merged)
    return processed_text


import diff_match_patch as dmp_module

dmp = dmp_module.diff_match_patch()
dmp.Diff_Timeout = 0


def is_valid(line, recovered_line):
    """Checks that the restore line matches the original line in everything but casing and punctuation marks"""
    line = abbreviations(line)
    line_no_punc = remove_punctuation(line, remove_spaces=True, do_lower=True, remove_accents=True)
    recovered_line_no_punc = remove_punctuation(recovered_line, remove_spaces=True, do_lower=True, remove_accents=True)

    is_same = line_no_punc == recovered_line_no_punc

    return is_same


def process_book(book_manifest, texts_dir, submanifests_dir, output_dir, restored_text_field, normalizer):
    book_id = os.path.basename(book_manifest).split(".")[0]
    text_f = f"{texts_dir}/{book_id}.txt"
    manifests = glob(f"{submanifests_dir}/{book_id}_*.json")
    logger.info(f"{book_id} -- {len(manifests)} manifests")

    # only continue (i.e. do not make early 'return') if there are {book_id}_{spk_id}.json files in submanifests_dir
    # that are not in output dir - else return early
    for book_id_spk_id in [os.path.basename(x).strip(".json") for x in manifests]:
        if not os.path.exists(os.path.join(output_dir, f"{book_id_spk_id}.json")):
            logger.info(f"Did not find {book_id_spk_id} in {output_dir} => will process this book")
            break
    else:
        return

    try:
        processed_text = normalize_text(text_f, normalizer)
        # re-run abbreviations since new are being added
        processed_text = abbreviations(processed_text)
        [
            recover_lines(
                manifest=manifest,
                processed_text=processed_text,
                output_dir=output_dir,
                restored_text_field=restored_text_field,
            )
            for manifest in manifests
        ]
    except:
        logger.info(f"{text_f} failed")
        return


class RestorePCForMLS(BaseProcessor):
    """Recovers original text from the MLS Librivox texts.

    This processor can be used to restore punctuation and capitalization for the
    MLS data. Uses the original data in https://dl.fbaipublicfiles.com/mls/lv_text.tar.gz.
    Saves recovered text in ``restored_text_field`` field.
    If text was not recovered, ``restored_text_field`` will be equal to ``n/a``.

    Args:
        language_long (str): the full name of the language, used for
            choosing the folder of the contents of
            "https://dl.fbaipublicfiles.com/mls/lv_text.tar.gz".
            E.g., "english", "spanish", "italian", etc.
        language_short (str): the short name of the language, used for
            specifying the normalizer we want to use. E.g., "en", "es", "it", etc.
        lv_text_dir (str): the directory where the contents of
            https://dl.fbaipublicfiles.com/mls/lv_text.tar.gz will be saved.
        submanifests_dir (str): the directory where submanifests (one for each
            combo of speaker + book) will be stored.
        restored_submanifests_dir (str): the directory where restored
            submanifests (one for each combo of speaker + book) will be stored.
        restored_text_field (str): the field where the recovered text will be stored.
        n_jobs (int): number of jobs to use for parallel processing. Defaults to -1.
        show_conversion_breakdown (bool): whether to show how much of each
            submanifest was restored. Defaults to True.
        dont_try_nemo_tn (bool): whether to skip trying to apply NeMo TN to the LibroVox
            text. Defaults to False.

    Returns:
        All the same data as in the input manifest with an additional key::

            <restored_text_field>: <restored text or n/a if match was not found>``
    """

    def __init__(
        self,
        language_long: str,
        language_short: str,
        lv_text_dir: str,
        submanifests_dir: str,
        restored_submanifests_dir: str,
        restored_text_field: str,
        n_jobs: int = -1,
        show_conversion_breakdown: bool = True,
        dont_try_nemo_tn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.language_long = language_long
        self.language_short = language_short
        self.lv_text_dir = Path(lv_text_dir)
        self.submanifests_dir = Path(submanifests_dir)
        self.restored_submanifests_dir = Path(restored_submanifests_dir)
        self.restored_text_field = restored_text_field
        self.n_jobs = n_jobs
        self.show_conversion_breakdown = show_conversion_breakdown
        self.dont_try_nemo_tn = dont_try_nemo_tn

    def process(self):
        """Main processing happens here.

        * Download & extract lv_text.
        * Create submanifests.
        * Restore P&C to submanifests.
        * Group back submanifests into a single manifest
        """
        from nemo_text_processing.text_normalization.normalize import Normalizer

        os.makedirs(self.lv_text_dir, exist_ok=True)

        # Download & extract lv_text.
        download_file(MLS_TEXT_URL, str(self.lv_text_dir))
        lv_text_data_folder = extract_archive(
            str(self.lv_text_dir / os.path.basename(MLS_TEXT_URL)), str(self.lv_text_dir)
        )

        # Create submanifests
        os.makedirs(self.submanifests_dir, exist_ok=True)

        data = {}
        with open(self.input_manifest_file, "r") as f:
            for line in tqdm(f):
                item = json.loads(line)
                name = Path(item["audio_filepath"]).stem
                reader_id, lv_book_id, sample_id = name.split("_")
                key = f"{lv_book_id}_{reader_id}"
                if key not in data:
                    data[key] = {}
                data[key][sample_id] = line

        for key, v in data.items():
            with open(f"{self.submanifests_dir}/{key}.json", "w") as f_out:
                for sample_id in sorted(v.keys()):
                    line = v[sample_id]
                    f_out.write(line)

        # Restore P&C to submanifests.
        os.makedirs(str(self.restored_submanifests_dir), exist_ok=True)

        if self.dont_try_nemo_tn:
            normalizer = None
        else:
            try:
                normalizer = Normalizer(
                    input_case="cased",
                    lang=self.language_short,
                    cache_dir="CACHE_DIR",
                    overwrite_cache=False,
                    post_process=True,
                )
            except NotImplementedError:  # some languages don't support text normalization
                normalizer = None

        # TODO: rename to maybe books_ids_in_datasplit
        books_ids_in_submanifests = set([x.split("_")[0] for x in data.keys()])

        Parallel(n_jobs=self.n_jobs)(
            delayed(process_book)(
                book_id,
                str(Path(lv_text_data_folder) / self.language_long),
                str(self.submanifests_dir),
                str(self.restored_submanifests_dir),
                self.restored_text_field,
                normalizer,
            )
            for book_id in tqdm(books_ids_in_submanifests)
        )

        # get stats --- keep track of book/spk ids in  our datasplit
        book_id_spk_ids_in_datasplit = set()  # set of tuples (book_id, spk_id), ...
        original_manifest_duration = 0
        with open(self.input_manifest_file, "r") as f:
            for line in f:
                line = json.loads(line)
                book_id, spk_id = Path(line["audio_filepath"]).stem.split("_")[:2]
                book_id_spk_ids_in_datasplit.add((book_id, spk_id))
                original_manifest_duration += line["duration"]
        logger.info(
            f"duration ORIGINAL total (for current datasplit): {round(original_manifest_duration / 60 / 60, 2)} hrs"
        )

        # make dicts to record durations of manifests
        filename_to_sub_manifest_durs = collections.defaultdict(float)
        filename_to_restored_sub_manifest_durs = collections.defaultdict(float)

        # duration in submanifests
        for book_id, spk_id in book_id_spk_ids_in_datasplit:
            manifest = os.path.join(self.submanifests_dir, f"{spk_id}_{book_id}.json")
            with open(manifest, "r") as f:
                for line in f:
                    line = json.loads(line)
                    filename_to_sub_manifest_durs[f"{spk_id}_{book_id}.json"] += line["duration"]

        # duration in restored_submanifests
        for book_id, spk_id in book_id_spk_ids_in_datasplit:
            manifest = os.path.join(self.restored_submanifests_dir, f"{spk_id}_{book_id}.json")
            if os.path.exists(manifest):
                with open(manifest, "r") as f:
                    for line in f:
                        line = json.loads(line)
                        if line[self.restored_text_field] != NA:
                            filename_to_restored_sub_manifest_durs[f"{spk_id}_{book_id}.json"] += line["duration"]
            else:
                filename_to_restored_sub_manifest_durs[f"{spk_id}_{book_id}.json"] = 0

        if self.show_conversion_breakdown:
            for filename in filename_to_sub_manifest_durs.keys():
                orig_dur = filename_to_sub_manifest_durs[filename]
                restored_dur = filename_to_restored_sub_manifest_durs[filename]

                pc_restored = 100 * restored_dur / orig_dur

                logger.info(
                    f"{filename}: {orig_dur/60:.2f} mins -> {restored_dur/60:.2f} mins\t({pc_restored:.2f}% restored)"
                )

        sub_manifest_duration = sum(list(filename_to_sub_manifest_durs.values()))
        restored_manifest_duration = sum(list(filename_to_restored_sub_manifest_durs.values()))

        logger.info("duration in submanifests (for current datasplit): %.2f hrs", sub_manifest_duration / 60 / 60)
        logger.info(
            "duration restored (for current datasplit): %.2f hrs (%.2f%%), lost: %.2f hrs",
            restored_manifest_duration / 60 / 60,
            restored_manifest_duration / sub_manifest_duration * 100,
            (sub_manifest_duration - restored_manifest_duration) / 60 / 60,
        )

        logger.info(
            "Combining restored manifest for current datasplit into single manifest at %s", self.output_manifest_file
        )

        # duration in restored_submanifests
        with open(self.output_manifest_file, 'w') as fout:
            for book_id, spk_id in book_id_spk_ids_in_datasplit:
                manifest = os.path.join(self.restored_submanifests_dir, f"{spk_id}_{book_id}.json")
                if os.path.exists(manifest):
                    with open(manifest, "r") as fin:
                        for line in fin:
                            fout.write(line)
