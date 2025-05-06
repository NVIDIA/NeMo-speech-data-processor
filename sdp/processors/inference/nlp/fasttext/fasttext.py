# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import requests
import tempfile
import wget

import fasttext

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class FastTextLangIdClassifier(BaseParallelProcessor):
    """
    This processor supports language identification using pretrained FastText models.
    It classifies text and adds the predicted label and probability to the dataset entry.
    If needed, it downloads the model, loads it into memory, and performs prediction on the 
    specified input text field.

    Args:
        model_name_or_path (str): Path to a FastText model file or the name of a supported remote model ('lid.176.bin' or 'lid.176.ftz').
        text_field (str): The name of the field in the dataset entry that contains the input text for classification.
        output_field (str): The name of the field to store the predicted label. Defaults to "label".
        cache_dir (str, optional): Directory to store the downloaded model file. If not provided, a temporary directory is used.
        **kwargs: Additional keyword arguments passed to `BaseParallelProcessor`.

    Returns:
        A manifest where each entry contains the original data fields plus:
            - `<output_field>`: The predicted label (e.g., language code for `lid.176.bin`).
            - `<output_field>_prob`: The probability of the prediction.
            
    """

    SUPPROTED_MODELS_URLS = {
        'lid.176.bin' : 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin',
        'lid.176.ftz' : 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'
    }
    
    def __init__(
        self,
        model_name_or_path: str,
        text_field: str,
        output_field: str = "label",
        cache_dir: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.text_field = text_field
        self.output_field = output_field
        self.cache_dir = cache_dir
        self._model = None

    def _download_model(self):
        """Downloads the FastText model from a predefined URL and stores it in the cache directory."""
        model_url = self.SUPPROTED_MODELS_URLS[self.model_name_or_path]
        logger.info(f'Downloading {self.model_name_or_path}..')
        response = requests.get(model_url)

        if response.status_code != 200:
            raise requests.exceptions.RequestException(
            f"Failed to download model file. Status code: {response.status_code}"
        )

        if self.cache_dir is None:
            self.cache_dir = tempfile.mkdtemp()
        os.makedirs(self.cache_dir, exist_ok=True)

        self.model_name_or_path = wget.download(model_url, out=self.cache_dir)
        logger.info(f'Model `{self.model_name_or_path}` has been downloaded to {self.cache_dir}.')

    def _load_model(self):
        """Lazily loads the FastText model into memory."""
        if self._model is None:
            self._model = fasttext.load_model(self.model_name_or_path)
    
    def prepare(self):
        """
        Prepares the model for classification:
        - Checks if the model file exists locally.
        - Downloads the model if only the name is given and it's known.
        - Raises ValueError if the path or model name is invalid.
        """
        if not os.path.exists(self.model_name_or_path):
            if os.path.exists(os.path.join(self.cache_dir, self.model_name_or_path)):
                self.model_name_or_path = os.path.join(self.cache_dir, self.model_name_or_path)
            elif self.model_name_or_path in self.SUPPROTED_MODELS_URLS:
                self._download_model()
            else:
                raise ValueError(f'Current model is not supported or filepath is invalid: {self.model_name_or_path}.')

    def process_dataset_entry(self, data_entry: dict):
        """Applies the classifier to a single dataset entry."""

        self._load_model()
        label, prob = self._model.predict(data_entry[self.text_field])
        data_entry[self.output_field] = label[0].replace('__label__', '')
        data_entry[f"{self.output_field}_prob"] = prob[0]

        return [DataEntry(data=data_entry)]
