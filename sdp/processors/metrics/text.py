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

import re
import os
import tempfile
import shutil
import requests
import wget
import tarfile
from glob import glob
from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

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
        num_words_key: str = "num_words",
        alphabet: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.num_words_key = num_words_key
        self.pattern = None
        if alphabet:
            self.pattern = re.compile("[^" + alphabet + "]")

    def process_dataset_entry(self, data_entry):
        text = data_entry[self.text_key]
        cleaned_string = text
        if self.pattern:
            cleaned_string = self.pattern.sub("", cleaned_string).strip()
        cleaned_string = re.sub("\\s+", " ", cleaned_string).strip()
        words = cleaned_string.split()
        num_words = len(words)
        data_entry[self.num_words_key] = num_words
        return [DataEntry(data=data_entry)]


class CharacterHistograms(BaseParallelProcessor):
    HISTOGRAMS_URL = 'https://dl.fbaipublicfiles.com/m2m_100/histograms.tar.gz'
    
    def __init__(self,
                 text_field: str,
                 lang_field: str = None,
                 lang: str = None,
                 threshold: float = 0.8,
                 cache_dir: str = None,
                 threshold_char: str = "]",
                 output_score_field: str = "hist_token_ratio",
                 **kwargs):
        super().__init__(**kwargs)
        self.text_field = text_field
        
        if lang_field is None and lang is None: 
            raise ValueError("One of the arguments `lang` or `lang_field` must be provided.")
                
        if lang_field is not None and lang is not None: 
            raise ValueError(
                f"Both `lang` ({lang}) and `lang_field` ({lang_field}) are provided, which makes the source of language ambiguous. Please provide only one of them."
            )
        
        self.text_field = text_field
        self.lang_field = lang_field
        self.lang = lang
        self.threshold = threshold
        self.cache_dir = cache_dir
        self.threshold_char = threshold_char
        self.output_score_field = output_score_field
        self.histograms = dict()

    def _read_hist(self, lang: str):
        hist_file = os.path.join(self.cache_dir, lang)
        chars = []
        with open(hist_file) as hist:
            for line in hist:
                char = line[0] 
                chars.append(char)
                if char == self.threshold_char:
                    break
        self.histograms[lang] =set(chars)
    
    def _download_histograms(self):
        logger.info(f'Downloading histograms collection..')
        response = requests.get(self.HISTOGRAMS_URL)

        if response.status_code != 200:
            raise requests.exceptions.RequestException(
            f"Failed to download model file. Status code: {response.status_code}"
        )

        if self.cache_dir is None:
            self.cache_dir = tempfile.mkdtemp()
        
        os.makedirs(self.cache_dir, exist_ok=True)

        histograms_tarfile = wget.download(self.HISTOGRAMS_URL, out=self.cache_dir)
        with tarfile.open(histograms_tarfile, "r:gz") as tar:
            tar.extractall(path=self.cache_dir)

        histograms_filepaths = glob(f'{self.cache_dir}/checkpoint/edunov/cc60_multilingual/clean_hists/*')
        for histogram_filepath in histograms_filepaths:
            shutil.move(histogram_filepath, os.path.join(self.cache_dir, os.path.basename(histogram_filepath)))
        
        os.remove(histograms_tarfile)
        shutil.rmtree(f'{self.cache_dir}/checkpoint/edunov/cc60_multilingual/clean_hists/')
        logger.info(f'Histograms has been downloaded to {self.cache_dir}.')

    def prepare(self):
        if (self.cache_dir is None or 
            not os.path.exists(self.cache_dir) or 
            not os.path.isdir(self.cache_dir) or 
            len(os.listdir(self.cache_dir)) == 0):
            
            self._download_histograms()

        logger.info(f'Reading histograms')
        available_langs = os.listdir(self.cache_dir)
        if self.lang is not None:
            if self.lang in available_langs:
                self._read_hist(self.lang)
            else:
                raise ValueError(f"Invalid value for `lang`: {self.lang}. Please provide one of the following: {available_langs}")
            logger.info(f'Histogram for `{self.lang}` has been read.')
        else:
            for lang in tqdm(available_langs):
                self._read_hist(lang)
            logger.info(f'Histograms have been read.')
        
        print(self.output_manifest_file)
        
    def process_dataset_entry(self, data_entry):
        lang = self.lang if self.lang is not None else data_entry[self.lang_field]
        if lang not in self.histograms:
            raise ValueError(f'lang `{lang} is not supported.')

        text = data_entry[self.text_field].strip()
        cnt = len([c for c in text if c in self.histograms[lang]])
        token_ratio = cnt / len(text)
        data_entry[self.output_score_field] = token_ratio
        return [DataEntry(data=data_entry)]