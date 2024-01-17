import os
from tqdm import tqdm
import soundfile as sf
from typing import Dict, List, Union

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.logging import logger
from sdp.utils.common import load_manifest

class GetRttmSegments(BaseParallelProcessor):
    """
    Processor for language identification (LID) of audio files using a pre-trained LID model.

    Args:
        rttm_field (str): The field in the dataset containing the path to the audio files for language identification.
        pretrained_model (str): The name of the pre-trained ASR model for language identification.
        output_file_field (str): The field to store the identified language for each audio file.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseProcessor`.

    """
    def __init__(
        self,
        rttm_field: str,
        duration_threshold: float = 20.0,
        duration_field: str = "duration",
        output_file_field: str = "audio_segments",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rttm_field = rttm_field
        self.duration_threshold = duration_threshold
        self.duration_field = duration_field
        self.output_file_field = output_file_field
        
    def split_long_segment(self, slices, duration, last_slice):
        duration0 = self.duration_threshold
        while duration0 < duration:
            slices.append(last_slice+duration0)
            duration0 += self.duration_threshold
            if duration0>duration:
                duration0 = duration
        slices.append(last_slice+duration0)
        return slices, last_slice + duration0


    def process_dataset_entry(self, data_entry: Dict):
        file_duration = data_entry[self.duration_field]
        rttm_file = data_entry[self.rttm_field]
        
        starts = []
        with open(rttm_file, "r") as f:
            for line in f:
                starts.append(float(line.split(" ")[3]))
        starts.append(file_duration)

        slices = [0]
        last_slice, last_start, last_duration, duration = 0, 0, 0, 0
        for start in starts:
            duration = start - last_slice

            if duration<=self.duration_threshold:
                pass
            elif duration>self.duration_threshold and last_duration < self.duration_threshold:
                slices.append(last_start)
                last_slice = last_start
                last_start = start
                last_duration = duration
                duration = start - last_slice
                if duration<=self.duration_threshold:
                    slices.append(start)
                    last_slice = start
                else:
                    slices, last_slice = self.split_long_segment(slices, duration, last_slice)

            else:
                slices.append(start)
                last_slice = start
            last_start = start
            last_duration = duration

        data_entry[self.output_file_field] = sorted(list(set(slices)))
        
        return [DataEntry(data=data_entry)]

class SplitFile(BaseParallelProcessor):
    """
    Processor for language identification (LID) of audio files using a pre-trained LID model.

    Args:
        rttm_field (str): The field in the dataset containing the path to the audio files for language identification.
        pretrained_model (str): The name of the pre-trained ASR model for language identification.
        output_file_field (str): The field to store the identified language for each audio file.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseProcessor`.

    """
    def __init__(
        self,
        splited_audio_dir: str,
        segments_field: str = "audio_segments",
        input_file_field: str = "source_filepath",
        output_file_field: str = "audio_filepath",
        duration_field: str = "duration",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_file_field = input_file_field
        self.splited_audio_dir = splited_audio_dir
        self.segments_field = segments_field
        self.output_file_field = output_file_field
        self.duration_field = duration_field

    def write_segment(self, data, samplerate, start_sec, end_sec, input_file):
        wav_save_file = os.path.join(self.splited_audio_dir, os.path.splitext(os.path.split(input_file)[1])[0], str(int(start_sec*100))+"-"+str(int(end_sec*100))+".wav")
        if not os.path.isfile(wav_save_file):
            data_sample = data[int(start_sec*samplerate):int(end_sec*samplerate)]
            duration = len(data_sample)/samplerate
            os.makedirs(os.path.split(wav_save_file)[0], exist_ok=True)
            sf.write(wav_save_file, data_sample, samplerate)
        return wav_save_file, duration
    
    def process_dataset_entry(self, data_entry: Dict):
        slices = data_entry[self.segments_field]
        input_file = data_entry[self.input_file_field]
        input_data, samplerate = sf.read(input_file)
        data_entries = []
        for i in range(len(slices[:-1])):
            wav_save_file, duration = self.write_segment(input_data, samplerate, slices[i], slices[i+1], input_file)
            data_entry[self.output_file_field] = wav_save_file
            data_entry[self.duration_field] = duration
            data_entries.append(DataEntry(data=data_entry.copy()))
        return data_entries
    