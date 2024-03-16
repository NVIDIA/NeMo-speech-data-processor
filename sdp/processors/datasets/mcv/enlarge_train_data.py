# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Optional

import pandas as pd

from sdp.logging import logger


def merge_tsvs(data_folder: str, dest_folder_name: str = 'enlarged', corrupted: Optional[str] = None) -> str:
    """
    Merges Train Dev and Other splits of the default MCV dataset into a single Train!
    In addition, filters repeated sentences and corrupted records; and adds duratoin to tsvs files.

    Args:
        data_folder (str): path to folder where .tsv files are stored
        dest_folder_name (str): destination folder name. Defaults to "enlarged"
        corrupted (str): path to Metadata (.csv) of corrupted audios - same column names as in tsv files

    Returns:
        str: A path to merged train.tsv file.
    """
    dest_folder = os.path.join(data_folder, dest_folder_name) if dest_folder_name else data_folder

    os.makedirs(dest_folder, exist_ok=True)
    dev = pd.read_csv(os.path.join(data_folder, 'dev.tsv'), sep='\t')
    test = pd.read_csv(os.path.join(data_folder, 'test.tsv'), sep='\t')
    other = pd.read_csv(os.path.join(data_folder, 'other.tsv'), sep='\t')
    train = pd.read_csv(os.path.join(data_folder, 'train.tsv'), sep='\t')
    clip_durations = pd.read_csv(os.path.join(data_folder, 'clip_durations.tsv'), sep='\t')

    clip_durations.iloc[:, -1] /= 1000  # convert to seconds
    clip_durations.columns = ['path', 'duration']
    clip_durations.set_index('path', inplace=True)

    for data_split in [dev, test, train, other]:
        data_split["duration"] = data_split['path'].apply(lambda x: clip_durations.loc[x, "duration"])

    logger.debug(
        f"Train Size before Enlarging (merging): {len(train)} audios with total of {round(train.duration.sum() / 3600, 2)} hours"
    )
    logger.debug("Merging the datasets ...")

    repeat_texts = set(other.sentence) & set(test.sentence)
    filtered_other = other[~other.sentence.isin(list(repeat_texts))]
    logger.debug(
        f'Removed {len(other[other.sentence.isin(list(repeat_texts))])} records from other.tsv repeated in '
        f'test.tsv'
    )

    enlarged_train = pd.concat([train, dev, filtered_other])
    if os.path.exists(corrupted):
        corrupted = pd.read_csv(corrupted)
        dev = dev[~dev.path.isin(corrupted.path)]
        test = test[~test.path.isin(corrupted.path)]
        enlarged_train = enlarged_train[~enlarged_train.path.isin(corrupted.path)]

    logger.success(
        f"Train Size after Enlarging (merging): {len(enlarged_train)} audios with total of {round(enlarged_train.duration.sum() / 3600, 2)} hours"
    )

    dev.to_csv(f"{dest_folder}/dev.tsv", index=False, sep='\t')
    test.to_csv(f"{dest_folder}/test.tsv", index=False, sep='\t')
    enlarged_train.to_csv(f"{dest_folder}/train.tsv", index=False, sep='\t')

    return f"{dest_folder}/train.tsv"
