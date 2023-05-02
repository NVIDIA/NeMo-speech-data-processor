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
from pathlib import Path
from tqdm import tqdm

from nemo.utils import logging

from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import download_file, extract_archive

MLS_TEXT_URL = "https://dl.fbaipublicfiles.com/mls/lv_text.tar.gz"


from .utils import remove_punctuation, read_text, process, abbreviations
from nemo_text_processing.text_normalization.normalize import Normalizer
from glob import glob
from joblib import Parallel, delayed
import re
import sys

sys.setrecursionlimit(1000000)


NA = "n/a"


def recover_lines(manifest, processed_text, output_dir):
    manifest_recovered = f"{output_dir}/{os.path.basename(manifest)}"
    if os.path.exists(manifest_recovered):
        return

    lines = []
    with open(manifest, "r") as f:
        for line in f:
            line = json.loads(line)
            lines.append(line["text"])

    logging.debug(f"processing {manifest}")
    logging.debug(f"processing - {len(lines)} lines")

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

    logging.debug(
        f"recovered {len(recovered_lines)} lines out of {len(lines)} -- {round(len(recovered_lines)/len(lines)*100, 2)}% -- {os.path.basename(manifest)}"
    )

    with open(manifest_recovered, "w") as f_out, open(manifest, "r") as f_in:
        for idx, line in enumerate(f_in):
            line = json.loads(line)
            if idx in recovered_lines:
                line["text_pc"] = recovered_lines[idx]
            else:
                line["text_pc"] = NA
            f_out.write(json.dumps(line, ensure_ascii=False) + "\n")


def normalize_text(text_f: str, normalizer: Normalizer):
    """
    Pre-process and normalized text_f file.

    Args:
        text_f: path to .txt file to normalize
        normalizer: 
    """
    raw_text = read_text(text_f)
    processed_text = abbreviations(process(raw_text))

    processed_text_list = normalizer.split_text_into_sentences(processed_text)
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
        processed_text_list_merged[i] = normalizer.normalize(
            text=text, punct_post_process=True, punct_pre_process=True
        )
    processed_text = " ".join(processed_text_list_merged)
    return processed_text


import diff_match_patch as dmp_module

dmp = dmp_module.diff_match_patch()
dmp.Diff_Timeout = 0


def is_valid(line, recovered_line):
    """ Checks that the restore line matches the original line in everything but casing and punctuation marks"""
    line = abbreviations(line)
    line_no_punc = remove_punctuation(line, remove_spaces=True, do_lower=True, remove_accents=True)
    recovered_line_no_punc = remove_punctuation(recovered_line, remove_spaces=True, do_lower=True, remove_accents=True)

    is_same = line_no_punc == recovered_line_no_punc

    return is_same


def process_book(book_manifest, texts_dir, submanifests_dir, output_dir, normalizer):
    book_id = os.path.basename(book_manifest).split(".")[0]
    text_f = f"{texts_dir}/{book_id}.txt"
    manifests = glob(f"{submanifests_dir}/{book_id}_*.json")
    logging.info(f"{book_id} -- {len(manifests)} manifests")

    # only continue (i.e. do not make early 'return') if there are {book_id}_{spk_id}.json files in submanifests_dir
    # that are not in output dir - else return early
    for book_id_spk_id in [os.path.basename(x).strip(".json") for x in manifests]:
        if not os.path.exists(os.path.join(output_dir, f"{book_id_spk_id}.json")):
            logging.info(f"Did not find {book_id_spk_id} in {output_dir} => will process this book")
            break
    else:
        # logging.info(
        #    f"All manifests {manifests} were also found in {output_dir} => skipping processing of this book"
        # )
        return

    try:
        processed_text = normalize_text(text_f, normalizer)
        # re-run abbreviations since new are being added
        processed_text = abbreviations(processed_text)
        [
            recover_lines(manifest=manifest, processed_text=processed_text, output_dir=output_dir)
            for manifest in manifests
        ]
    except:
        logging.info(f"{text_f} failed")
        return


class AddPCToMLSInitialManifest(BaseProcessor):
    """
    Downloads and unzips raw MLS data for the specified language, and creates an initial manifest using
    the transcripts provided in the raw data. 
    Args:
        TODO: update
        language: the language of the data you wish to be downloaded. This will be used to format the 
            URL from which we attempt to download the data.
        download_dir: the directory where the downloaded data will be saved.
        data_split: the data split for which the initial manifest will be created.
        resampled_audio_dir: the directory where the resampled (16kHz) wav files will be stored.
        use_test_data: if `True`, will use the test data manifest located at `TEST_DATA_PATH` to carry out tests.
    """

    def __init__(
        self,
        language_long: str,
        language_short: str,
        lv_text_dir: str,
        submanifests_dir: str,
        restored_submanifests_dir: str,
        n_jobs: int,
        show_conversion_breakdown: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.language_long = language_long
        self.language_short = language_short
        self.lv_text_dir = Path(lv_text_dir)
        self.submanifests_dir = Path(submanifests_dir)
        self.restored_submanifests_dir = Path(restored_submanifests_dir)
        self.n_jobs = n_jobs
        self.show_conversion_breakdown = show_conversion_breakdown

    def process(self):
        """
        Download & extract lv_text.
        Create submanifests.
        Restore P&C to submanifests.
        Group back submanifests into 1 single manifest
        """
        # Download & extract lv_text.
        download_file(MLS_TEXT_URL, str(self.lv_text_dir))
        lv_text_data_folder = extract_archive(
            str(self.lv_text_dir / os.path.basename(MLS_TEXT_URL)), str(self.lv_text_dir)
        )

        # Create submanifests
        os.makedirs(str(self.submanifests_dir), exist_ok=True)

        data = {}
        with open(self.input_manifest_file, "r") as f:
            for line in tqdm(f):
                item = json.loads(line)
                name = item["audio_filepath"].split("/")[-1].replace(".wav", "")
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

        normalizer = Normalizer(
            input_case="cased",
            lang=self.language_short,
            cache_dir="CACHE_DIR",
            overwrite_cache=False,
            post_process=True,
        )

        # books_ids_in_submanifests = set(
        #     [
        #         os.path.basename(x).split("_")[0]
        #         for x in glob(f"{str(self.submanifests_dir)}/*.json")
        #     ]
        # )
        # TODO: rename to maybe books_ids_in_datasplit
        books_ids_in_submanifests = set([x.split("_")[0] for x in data.keys()])

        Parallel(n_jobs=self.n_jobs)(
            delayed(process_book)(
                book_id,
                str(Path(lv_text_data_folder) / self.language_long),
                str(self.submanifests_dir),
                str(self.restored_submanifests_dir),
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
                book_id, spk_id = os.path.basename(line["audio_filepath"]).strip('.wav').split("_")[:2]
                book_id_spk_ids_in_datasplit.add((book_id, spk_id))
                original_manifest_duration += line["duration"]
        logging.info(
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
                        if line["text_pc"] != NA:
                            filename_to_restored_sub_manifest_durs[f"{spk_id}_{book_id}.json"] += line["duration"]
            else:
                filename_to_restored_sub_manifest_durs[f"{spk_id}_{book_id}.json"] = 0

        if self.show_conversion_breakdown:
            for filename in filename_to_sub_manifest_durs.keys():
                orig_dur = filename_to_sub_manifest_durs[filename]
                restored_dur = filename_to_restored_sub_manifest_durs[filename]

                pc_restored = 100 * restored_dur / orig_dur

                logging.info(
                    f"{filename}: {orig_dur/60:.2f} mins -> {restored_dur/60:.2f} mins\t({pc_restored:.2f}% restored)"
                )

        sub_manifest_duration = sum(list(filename_to_sub_manifest_durs.values()))
        restored_manifest_duration = sum(list(filename_to_restored_sub_manifest_durs.values()))

        logging.info(
            f"duration in submanifests (for current datasplit): {round(sub_manifest_duration / 60 / 60, 2)} hrs"
        )
        logging.info(
            f"duration restored (for current datasplit): {round(restored_manifest_duration / 60 / 60, 2)} hrs ({round(restored_manifest_duration/sub_manifest_duration * 100, 2)}%), lost: {round((sub_manifest_duration - restored_manifest_duration) / 60 / 60, 2)} hrs"
        )

        logging.info(
            f"Combining restored manifest for current datasplit into single manifest at {self.output_manifest_file}"
        )

        # duration in restored_submanifests
        with open(self.output_manifest_file, 'w') as fout:

            for book_id, spk_id in book_id_spk_ids_in_datasplit:
                manifest = os.path.join(self.restored_submanifests_dir, f"{spk_id}_{book_id}.json")
                if os.path.exists(manifest):
                    with open(manifest, "r") as fin:
                        for line in fin:
                            fout.write(line)
