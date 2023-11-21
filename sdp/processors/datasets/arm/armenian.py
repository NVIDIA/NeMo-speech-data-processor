import torch
import whisper # pip install -U openai-whisper
import os
import json
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import subprocess
from typing import Dict, List, Union
from operator import lt, le, eq, ne, ge, gt
from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor, DataEntry
from sdp.logging import logger



    
def load_manifest(manifest: Path) -> List[Dict[str, Union[str, float]]]:
    result = []
    with manifest.open() as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            result.append(data)
    return result
    
class CreateInitialManifestByExt(BaseParallelProcessor):
    """
        Args:
        raw_data_dir (str): where to put raw downloaded data.
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
        target_samplerate (int): sample rate to resample to. Defaults to 16000.
        target_nchannels (int): target number of channels. Defaults to 1.
    """
    def __init__(
        self,
        raw_data_dir: str,
        output_field: str = "audio_filepath",
        extention: str = "mp3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.output_field = output_field
        self.extention = extention

    def read_manifest(self):
        input_files = [str(self.raw_data_dir / video) for video in \
                       self.raw_data_dir.rglob('*.' + self.extention)]
        return input_files
    
    def process_dataset_entry(self, data_entry):
        data = {self.output_field: data_entry}
        return [DataEntry(data=data)]


def ffmpeg_convert(jpg: str, wav: str, ar: int = 0, ac: int = 1):
    process_args = ["ffmpeg", "-i", jpg, '-ac', str(ac), "-map", "0:a", "-c:a", "pcm_s16le", "-y", wav]
    if ar:
        process_args = process_args[:-1]
        process_args.extend(["-ar", str(ar), wav])
    return subprocess.run(process_args, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)

class FfmpegConvert(BaseParallelProcessor):
    """
        Args:
        input_field (str): field with path to video file in the input manifest
        output_field (str): field with path to audio file in the output manifest
        key_field (str): field with key value
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
        target_samplerate (int): sample rate to resample to. Defaults to 16000.
        target_nchannels (int): target number of channels. Defaults to 1.
    """
    def __init__(
        self,
        resampled_audio_dir: str,
        input_field: str,
        output_field: str,
        key_field: str = None,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        self.key_field = key_field
        self.resampled_audio_dir = resampled_audio_dir
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def prepare(self):
        os.makedirs(os.path.split(self.output_manifest_file)[0], exist_ok=True)
        os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        video = data_entry[self.input_field]
        if self.key_field:
            key = data_entry[self.key_field]
            os.makedirs(os.path.join(self.resampled_audio_dir, key.split("/")[0]), exist_ok=True)
        else:
            key = os.path.splitext(video)[0].split("/")[-1]
        audio = os.path.join(self.resampled_audio_dir, key) + ".wav"

        if not os.path.isfile(audio):
            ffmpeg_convert(video, audio, self.target_samplerate, self.target_nchannels)

        data_entry[self.output_field]= audio
        if self.key_field:
            data_entry[self.key_field] = key
        return [DataEntry(data=data_entry)]


class AudioDuration(BaseParallelProcessor):
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
        audio_filepath = data_entry[self.input_field]
        try:
            data, samplerate = sf.read(audio_filepath)
            data_entry[self.output_field]=data.shape[0]/samplerate
        except Exception as e:
            logger.warning(str(e) + " file: " + audio_filepath)
            data_entry[self.output_field] = -1.0
        return [DataEntry(data=data_entry)]


class ASR_Whisper(BaseProcessor):
    """
        Transcribe usinf ASR model from HuggingFace.
        Args:
        pretrained_model (str): name of pretrained model on HuggingFace.
        output_text_field (str): field to save transcription result.
        device (str): Inference device.
        batch_size (str): Inference batch size.
    """
    def __init__(
        self,
        pretrained_model: str,
        output_text_field: str,
        device: str = None,
        batch_size: str = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_model = pretrained_model
        self.output_text_field = output_text_field
        self.device = device
        self.batch_size = batch_size
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        self.model = whisper.load_model(self.pretrained_model)
    
    def process(self):
        json_list = load_manifest(Path(self.input_manifest_file))
        
        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)
        
        with Path(self.output_manifest_file).open('w') as f:
            for item in tqdm(json_list):
                pred_text, pred_lang = self.whisper_infer(item["audio_filepath"])

                item[self.output_text_field] = pred_text
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def whisper_infer(self, audio_path):
        audio = whisper.load_audio(audio_path)

        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        mel = mel.to(self.device)

        _, probs = self.model.detect_language(mel)
        lang = max(probs, key=probs.get)
        
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)
        return result.text, lang
    
class ReadTxt(BaseParallelProcessor):
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
        fname = data_entry[self.input_field]
        data_list = []
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = data_entry.copy()
                    data[self.output_field] = line
                    data_list.append(DataEntry(data=data))
        return data_list


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


class PreserveByValue(BaseParallelProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        input_field: str,
        target_value: Union[int, str],
        operator: str = "eq",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.target_value = target_value
        if operator == "lt":
            self.operator = lt
        elif operator == "le":
            self.operator = le
        elif operator == "eq":
            self.operator = eq
        elif operator == "ne":
            self.operator = ne
        elif operator == "ge":
            self.operator = ge
        elif operator == "gt":
            self.operator = gt

    def process_dataset_entry(self, data_entry):
        input_value = data_entry[self.input_field]
        target = self.target_value
        if self.operator(input_value, target):
            return [DataEntry(data=data_entry)]
        else:
            return [DataEntry(data=None)]


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
        
        data_entry[self.output_field] = input_values[-1] + ", " +input_values[-2]
        if input_values[-2] == "Նար-Դոս":
            data_entry[self.output_field] += " (1867 - 1933), " + "https://hy.wikisource.org/wiki/%D5%80%D5%A5%D5%B2%D5%AB%D5%B6%D5%A1%D5%AF:%D5%86%D5%A1%D6%80-%D4%B4%D5%B8%D5%BD"
        elif input_values[-2] == "Ակսել Բակունց":
            data_entry[self.output_field] += " (1899 - 1937), " + "https://aybuben.com/axel-bakunts"
        return [DataEntry(data=data_entry)]

def read_jsonl(manifest_file):
    rec = []
    with open(manifest_file, 'r') as the_file:
        for l in the_file:
            rec.append(json.loads(l))
    return pd.DataFrame.from_records(rec)

class MakeTsv(BaseProcessor):
    """
    """
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def process(self):
        df1 = read_jsonl(self.input_manifest_file)
        df1.to_csv(self.output_manifest_file, index=None)

class RandomPart(BaseProcessor):
    """
    """
    def __init__(
        self,
        part: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.part = part

    def process(self):
        df1 = pd.read_csv(self.input_manifest_file)
        df1.sample(frac=self.part).to_csv(self.output_manifest_file, index=None)