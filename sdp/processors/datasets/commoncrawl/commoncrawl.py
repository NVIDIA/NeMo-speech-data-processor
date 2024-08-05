import os
from typing import List

import soundfile as sf
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.processors.datasets.commoncrawl.harv_utils import split_by_vtt



class SplitByVttSentence(BaseParallelProcessor):
    """
        A class for splitting audio files based on VTT (WebVTT) sentence-level segmentation in a dataset.

        Args:
            splited_audio_dir (str): The directory to store the split audio files.
            source_audio_key (str): The field in the dataset containing the path to the source audio files.
            target_audio_key (str): The field to store the paths of the split audio files.
            duration_key (str): The field to store the duration of each split audio segment.
            text_key (str): The field to store the transcriptions corresponding to each split audio segment.
            caption_file_key (str): The field in the dataset containing the path to the VTT (WebVTT) files for segmentation.
            additional_fields (List[str], optional): List of additional fields to copy from the original data entry to the split entries.
                Defaults to an empty list.
            duration_threshold (float, optional): The duration threshold in seconds for each split audio segment. Defaults to 10.0.
    """

    def __init__(
            self,
            splited_audio_dir: str,
            source_audio_field: str,
            target_audio_field: str,
            duration_field: str,
            text_field: str,
            vtt_field: str,
            additional_fields: List[str] = [],
            duration_threshold: float = 10.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.splited_audio_dir = splited_audio_dir
        self.source_audio_field = source_audio_field
        self.target_audio_field = target_audio_field
        self.duration_field = duration_field
        self.text_field = text_field
        self.vtt_field = vtt_field
        self.duration_threshold = duration_threshold
        self.additional_fields = additional_fields

    def prepare(self):
        os.makedirs(self.splited_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        vtt_file = data_entry[self.vtt_field]
        source_audio = data_entry[self.source_audio_field]
        res_list = []

        if os.path.isfile(source_audio):
            data, samplerate = sf.read(source_audio)
            text_list, start_s, end_s = split_by_vtt(vtt_file, samplerate)
            text_c = ''
            start_c, end_c = 0, 0
            if text_list:
                for text, start_sr, end_sr in zip(text_list, start_s, end_s):
                    text_c += " " + text
                    if start_c == 0:
                        start_c = start_sr
                    else:
                        pass
                    end_c = end_sr
                    if len(text_c) > 0 and (
                            end_c - start_c > self.duration_threshold * samplerate or
                            text_c[-1] == "." or text_c[-1] == "?"):
                        res_list.append(
                            self.makeDataEntry(data_entry, data, vtt_file, samplerate, text_c, start_c, end_c))
                        text_c = ''
                        start_c, end_c = 0, 0
                    else:
                        pass
                if len(text_c) > 0 and start_c != 0:
                    res_list.append(self.makeDataEntry(data_entry, data, vtt_file, samplerate, text_c, start_c, end_c))

        return res_list

    def makeDataEntry(self, data_entry, data, vtt_file, samplerate, text_c, start_c, end_c):
        data_sample = data[start_c:end_c]
        wav_save_file = os.path.join(self.splited_audio_dir, '/'.join(os.path.splitext(vtt_file)[0].split('/')[-2:]),
                                     str(int(start_c / (samplerate / 1000))) + "-" + str(
                                         int(end_c / (samplerate / 1000))) + ".wav")
        if not os.path.isfile(wav_save_file):
            os.makedirs(os.path.split(wav_save_file)[0], exist_ok=True)
            sf.write(wav_save_file, data_sample, samplerate)

        data = {self.target_audio_field: wav_save_file,
                self.duration_field: data_sample.shape[0] / samplerate,
                self.text_field: text_c.strip(),
                }
        for field in self.additional_fields:
            data[field] = data_entry[field]
        return DataEntry(data=data)

