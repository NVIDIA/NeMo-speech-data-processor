import os
import requests
import tempfile
import wget

import fasttext

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

class FastTextClassifier(BaseParallelProcessor):
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
        if self._model is None:
            self._model = fasttext.load_model(self.model_name_or_path)
    
    def prepare(self):
        if not os.path.exists(self.model_name_or_path):
            if os.path.exists(os.path.join(self.cache_dir, self.model_name_or_path)):
                self.model_name_or_path = os.path.join(self.cache_dir, self.model_name_or_path)
            elif self.model_name_or_path in self.SUPPROTED_MODELS_URLS:
                self._download_model()
            else:
                raise ValueError(f'Current model is not supported or filepath is invalid: {self.model_name_or_path}.')

    def process_dataset_entry(self, data_entry: dict):
        self._load_model()
        label, prob = self._model.predict(data_entry[self.text_field])
        data_entry[self.output_field] = label[0].replace('__label__', '')
        data_entry[f"{self.output_field}_prob"] = prob[0]

        return [DataEntry(data=data_entry)]
