import torch
import whisper # pip install -U openai-whisper
import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import subprocess
from typing import Dict, List, Union
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
        v_df = pd.DataFrame({self.output_field: input_files})
        return v_df.values
    
    def process_dataset_entry(self, data_entry):
        (inputf) = data_entry
        
        data = {self.output_field: inputf[0]}
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
    