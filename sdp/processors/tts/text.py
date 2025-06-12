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

import json
from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor, DataEntry
from sdp.utils.common import load_manifest, save_manifest
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
from nemo.collections.nlp.models import PunctuationCapitalizationModel

class InverseTextNormalizationProcessor(BaseParallelProcessor):
    """This processor performs inverse text normalization on text data.

    It normalizes text data from various languages into a standard format using the InverseNormalizer.
    The processor converts written text representations into their spoken form (e.g. "123" -> "one hundred twenty three").

    Args:
        language (str): Language code for the text normalization. Defaults to "en"

    Returns:
        The same data as in the input manifest, but with an additional "text_ITN" field containing
        the inverse normalized text for each segment.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.tts.text.InverseTextNormalizationProcessor
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_itn.json
              language: "en"
    """
    def __init__(self, 
                 language="en",
                 **kwargs):
        super().__init__(**kwargs)
        self.normalizer = InverseNormalizer(lang=language)
    
    def read_manifest(self):
        ''' Reads metadata from JSONL file in the input manifest
        and converts it to data entries '''

        dataset_entries = load_manifest(self.input_manifest_file, encoding="utf8")

        return dataset_entries

    def process_dataset_entry(self, metadata: DataEntry):
        for segment in metadata["segments"]:
            if "text" in segment:
                text = segment["text"]
                sentences = self.normalizer.split_text_into_sentences(text)
                text_ITN = " ".join(self.normalizer.normalize_list(sentences))
                segment["text_ITN"] = text_ITN
        return [DataEntry(data=metadata)]


class PunctuationAndCapitalizationOnSegmentsProcessor(BaseProcessor):
    """This processor performs punctuation and capitalization on segments text data.

    It capitalizes the first letter of each sentence and adds punctuation marks to the text.

    Args:
        model_name (str): Name of the pretrained model to use. Defaults to "punctuation_en_bert"
        model_path (str, Optional): Path to the local PNC model file. If provided, overrides model_name
        batch_size (int, Optional): Batch size for processing. Defaults to 64

    Returns:
        The same data as in the input manifest, but with punctuation and capitalization added
        to each segment.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.tts.text.PunctuationAndCapitalizationOnSegmentsProcessor
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_pnc.json
    """
    def __init__(self,
            model_name="punctuation_en_bert",
            model_path=None,
            batch_size=64,
            **kwargs):

        super().__init__(**kwargs)
        if model_path is not None:
            self.pnc_model = PunctuationCapitalizationModel.restore_from(model_path)
        else:
            self.pnc_model = PunctuationCapitalizationModel.from_pretrained(model_name)
        
        self.batch_size= batch_size
        self.pnc_model.cuda()
    
    def process(self):
        manifest = load_manifest(self.input_manifest_file)

        results = []
        all_text = []
        for metadata in manifest:
            for segment in metadata["segments"]:
                if "text" in segment:
                    text = segment["text"]
                    all_text.append(text)
                    
        text_PNC = self.pnc_model.add_punctuation_capitalization(all_text, batch_size=self.batch_size)

        i = 0
        for metadata in manifest:
            for segment in metadata["segments"]:
                if "text" in segment:
                    segment["text"] = text_PNC[i]
                    i+=1
            results.append(metadata)

        save_manifest(results, self.output_manifest_file)

class PunctuationAndCapitalizationProcessor(BaseProcessor):
    """This processor performs punctuation and capitalization on text data.

    It capitalizes the first letter of each sentence and adds punctuation marks to the text.

    Args:
        model_name (str): Name of the pretrained model to use. Defaults to "punctuation_en_bert"
        model_path (str, Optional): Path to the local PNC model file. If provided, overrides model_name
        batch_size (int, Optional): Batch size for processing. Defaults to 64

    Returns:
        The same data as in the input manifest, but with punctuation and capitalization added
        to each segment.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.tts.text.PunctuationAndCapitalizationProcessor
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_pnc.json
    """
    def __init__(self,
            model_name="punctuation_en_bert",
            model_path=None,
            batch_size=64,
            **kwargs):

        super().__init__(**kwargs)
        if model_path is not None:
            self.pnc_model = PunctuationCapitalizationModel.restore_from(model_path)
        else:
            self.pnc_model = PunctuationCapitalizationModel.from_pretrained(model_name)
        
        self.batch_size= batch_size
        self.pnc_model.cuda()
    
    def process(self):
        manifest = load_manifest(self.input_manifest_file)

        all_text = []
        
        for metadata in manifest:
            is_segmented_entry = ('split_filepaths' in metadata and metadata['split_filepaths'] is None) or ('split_filepaths' not in metadata)
            if  is_segmented_entry and ('text' in metadata and metadata['text'] != ''):
                text = ' '.join([x['word'] for x in metadata['alignment']]).strip()
                all_text.append(text)
                    
        text_PNC = self.pnc_model.add_punctuation_capitalization(all_text, batch_size=self.batch_size)

        i = 0
        with open(self.output_manifest_file, 'w') as f:
            for metadata in manifest:
                is_segmented_entry = ('split_filepaths' in metadata and metadata['split_filepaths'] is None) or ('split_filepaths' not in metadata)
                if  is_segmented_entry and ('text' in metadata and metadata['text'] != ''):
                    pnc_words = text_PNC[i].split()
                    pnc_words_idx = 0
                    for word in metadata['alignment']:
                        if word['word'] != '':
                            word['word'] = pnc_words[pnc_words_idx]
                            pnc_words_idx += 1
                    i+=1
                f.write(json.dumps(metadata) + "\n")

