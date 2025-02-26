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


from sdp.processors.base_processor import BaseProcessor
import ndjson
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
from nemo.collections.nlp.models import PunctuationCapitalizationModel

class InverseTextNormalizationProcessor(BaseProcessor):
    """A processor for performing inverse text normalization on transcribed text.

    This processor converts written-form numbers and expressions into their spoken form
    (e.g., "123" -> "one hundred twenty three"). It processes text segment by segment
    and stores both original and normalized versions.

    Args:
        language (str): Language code for normalization (e.g., "en" for English). Defaults to "en".
        **kwargs: Additional arguments passed to the base processor.
    """
    def __init__(self, 
                 language="en",
                 **kwargs):
        """Initialize the inverse text normalization processor.

        Args:
            language (str): Language code for normalization. Defaults to "en".
            **kwargs: Additional arguments passed to the base processor.
        """
        super().__init__(**kwargs)
        self.normalizer = InverseNormalizer(lang=language)

    def process(self):
        """Process the input manifest file to normalize text segments.

        This method:
        1. Reads the input manifest
        2. For each text segment:
            - Splits text into sentences
            - Applies inverse text normalization
            - Stores normalized text in 'text_ITN' field
        3. Saves results to the output manifest

        The output manifest preserves the original text in 'text' field
        and adds normalized text in 'text_ITN' field.
        """
        with open(self.input_manifest_file) as f:
            manifest = ndjson.load(f)

        results = []

        for metadata in manifest:
            for segment in metadata["segments"]:
                if "text" in segment:
                    text = segment["text"]
                    sentences = self.normalizer.split_text_into_sentences(text)
                    text_ITN = " ".join(self.normalizer.normalize_list(sentences))
                    segment["text_ITN"] = text_ITN
            results.append(metadata)

        with open(self.output_manifest_file, 'w') as f:
            ndjson.dump(results, f)


class PunctuationAndCapitalizationProcessor(BaseProcessor):
    """A processor for adding punctuation and proper capitalization to text.

    This processor uses a pre-trained model to add appropriate punctuation
    and capitalization to plain text transcriptions. It processes text in batches
    for improved efficiency.

    Args:
        model_name (str): Name of the pretrained model to use. Defaults to "punctuation_en_bert".
        model_path (str, optional): Path to a local model file. If provided, overrides model_name.
        batch_size (int): Number of texts to process in each batch. Defaults to 64.
        **kwargs: Additional arguments passed to the base processor.
    """
    def __init__(self,
            model_name="punctuation_en_bert",
            model_path=None,
            batch_size=64,
            **kwargs):
        """Initialize the punctuation and capitalization processor.

        Args:
            model_name (str): Name of the pretrained model to use. Defaults to "punctuation_en_bert".
            model_path (str, optional): Path to a local model file. If provided, overrides model_name.
            batch_size (int): Number of texts to process in each batch. Defaults to 64.
            **kwargs: Additional arguments passed to the base processor.
        """

        super().__init__(**kwargs)
        if model_path is not None:
            self.pnc_model = PunctuationCapitalizationModel.restore_from(model_path)
        else:
            self.pnc_model = PunctuationCapitalizationModel.from_pretrained(model_name)
        
        self.batch_size= batch_size
        self.pnc_model.cuda()
    
    def process(self):
        """Process the input manifest file to add punctuation and capitalization.

        This method:
        1. Reads the input manifest
        2. Extracts all text segments
        3. Processes texts in batches to add punctuation and capitalization
        4. Updates the original text segments with processed versions
        5. Saves results to the output manifest

        The output manifest contains the original metadata with updated text
        that includes proper punctuation and capitalization.
        """
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
