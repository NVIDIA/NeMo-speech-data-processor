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


from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor, DataEntry
import json
import ndjson
import re
from tqdm import tqdm
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
import nemo.collections.asr as nemo_asr
from nemo.collections.nlp.models import PunctuationCapitalizationModel
import pyarabic.araby as araby

class InverseTextNormalizationProcessor(BaseParallelProcessor):

    def __init__(self, 
                 language="en",
                 **kwargs):
        super().__init__(**kwargs)
        self.normalizer = InverseNormalizer(lang=language)
    
    def read_manifest(self):
        ''' Reads metadata from NDJSON file in the input manifest
        and converts it to data entries '''

        with open(self.input_manifest_file, "r", encoding="utf8") as fin:
            dataset_entries = ndjson.load(fin)

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
        with open(self.input_manifest_file) as f:
            manifest = ndjson.load(f)

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

        with open(self.output_manifest_file, 'w') as f:
            ndjson.dump(results, f)

class PunctuationAndCapitalizationProcessor(BaseProcessor):
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
        with open(self.input_manifest_file) as f:
            manifest = ndjson.load(f)

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

class ArabicRemoveDiacriticsProcessor(BaseProcessor):

    def __init__(self, 
                 **kwargs):
        super().__init__(**kwargs)

    def process(self):
        with open(self.input_manifest_file) as f:
            manifest = ndjson.load(f)
        
        results = []
        for metadata in manifest:
            is_segmented_entry = ('split_filepaths' in metadata and metadata['split_filepaths'] is None) or ('split_filepaths' not in metadata)
            if  is_segmented_entry and ('text' in metadata and metadata['text'] != ''):
                metadata["text"] = araby.strip_diacritics(metadata["text"])
                for word in metadata['alignment']:
                    if word['word'] != '':
                        word['word'] = araby.strip_diacritics(word['word'])
            results.append(metadata)
                
        with open(self.output_manifest_file, 'w') as f:
            ndjson.dump(results, f)
