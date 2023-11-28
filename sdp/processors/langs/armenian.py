import torch
import whisper # pip install -U openai-whisper
import os
import json
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Union
from operator import lt, le, eq, ne, ge, gt
from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor, DataEntry
from sdp.processors.modify_manifest.common import load_manifest
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline



class SplitBySentence(BaseParallelProcessor):
    """
        Args:
        input_field (str): where to get path to wav file.
        output_field (str): where to put to frequency bandwidth.
    """
    def __init__(
        self,
        input_field: str,
        output_field: str,
        pattern: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        self.pattern = re.compile(pattern)

    def process_dataset_entry(self, data_entry):
        line = data_entry[self.input_field]
        data_list = []
        start = 0
        ends = [m.start() for m in self.pattern.finditer(line)]
        if ends:
            for end in ends:
                sent = line[start:end+1].strip()
                # if sent and sent[0].isupper():
                data = data_entry.copy()
                data[self.output_field] = sent
                data_list.append(DataEntry(data=data))
                start = end+1
            if start<len(line):
                pass
        else:
            data = data_entry.copy()
            data[self.output_field] = line.strip()
            data_list.append(DataEntry(data=data))
        return data_list

class NumWords(BaseParallelProcessor):
    """
        Args:
        input_field (str): where to get path to wav file.
        output_field (str): where to put to frequency bandwidth.
    """
    def __init__(
        self,
        input_field: str,
        output_field: str,
        alphabet: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        self.pattern = re.compile("[^"+alphabet+"]")

    def process_dataset_entry(self, data_entry):
        text = data_entry[self.input_field]
        cleaned_string = self.pattern.sub('', text).strip()
        cleaned_string = re.sub('\s+', ' ', cleaned_string).strip()
        words = cleaned_string.split()
        num_words = len(words)
        data_entry[self.output_field] = num_words
        return [DataEntry(data=data_entry)]


class GetSource(BaseParallelProcessor):
    """
        Args:
        input_field (str): where to get path to wav file.
        output_field (str): where to put to frequency bandwidth.
    """
    def __init__(
        self,
        input_field: str,
        output_field: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field

    def process_dataset_entry(self, data_entry):
        input_values = os.path.splitext(data_entry[self.input_field])[0].split("/")
        
        data_entry[self.output_field] = input_values[-1]# + ", " +input_values[-2]
        return [DataEntry(data=data_entry)]


class MakeTsv(BaseProcessor):
    """
    """
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def process(self):
        df1 = pd.DataFrame.from_records(load_manifest(self.input_manifest_file))
        df1.to_csv(self.output_manifest_file, index=None, sep='\t')

class RandomTsvPart(BaseProcessor):
    """
    """
    def __init__(
        self,
        part: float,
        random_state: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.part = part
        self.random_state = random_state

    def process(self):
        df1 = pd.read_csv(self.input_manifest_file, sep='\t')
        df1.sample(frac=self.part, random_state = self.random_state).to_csv(self.output_manifest_file, index=None, sep='\t')