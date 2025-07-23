import os
from typing import Dict

import soundfile as sf

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class GetRttmSegments(BaseParallelProcessor):
    """This processor extracts audio segments based on RTTM (Rich Transcription Time Marked) files.

    The class reads an RTTM file specified by the `rttm_key` in the input data entry and
    generates a list of audio segment start times. It ensures that segments longer than a specified
    duration threshold are split into smaller segments. The resulting segments are stored in the
    output data entry under the `output_file_key`.

    Args:
        rttm_key (str): The key in the manifest that contains the path to the RTTM file.
        output_file_key (str, optional): The key in the data entry where the list of audio segment
            start times will be stored. Defaults to "audio_segments".
        duration_key (str, optional): The key in the data entry that contains the total duration
            of the audio file. Defaults to "duration".
        duration_threshold (float, optional): The maximum duration for a segment before it is split.
            Segments longer than this threshold will be divided into smaller segments. Defaults to 20.0 seconds.

    Returns:
        A list containing a single `DataEntry` object with the updated data entry, which includes
        the `output_file_key` containing the sorted list of audio segment start times.
    """

    def __init__(
        self,
        rttm_key: str,
        output_file_key: str = "audio_segments",
        duration_key: str = "duration",
        duration_threshold: float = 20.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rttm_key = rttm_key
        self.duration_threshold = duration_threshold
        self.duration_key = duration_key
        self.output_file_key = output_file_key

    def split_long_segment(self, slices, duration, last_slice):
        duration0 = self.duration_threshold
        while duration0 < duration:
            slices.append(last_slice + duration0)
            duration0 += self.duration_threshold
            if duration0 > duration:
                duration0 = duration
        slices.append(last_slice + duration0)
        return slices, last_slice + duration0

    def process_dataset_entry(self, data_entry: Dict):
        file_duration = data_entry[self.duration_key]
        rttm_file = data_entry[self.rttm_key]

        starts = []
        with open(rttm_file, "r") as f:
            for line in f:
                starts.append(float(line.split(" ")[3]))
        starts.append(file_duration)

        slices = [0]
        last_slice, last_start, last_duration, duration = 0, 0, 0, 0
        for start in starts:
            duration = start - last_slice

            if duration <= self.duration_threshold:
                pass
            elif duration > self.duration_threshold and last_duration < self.duration_threshold:
                slices.append(last_start)
                last_slice = last_start
                last_start = start
                last_duration = duration
                duration = start - last_slice
                if duration <= self.duration_threshold:
                    slices.append(start)
                    last_slice = start
                else:
                    slices, last_slice = self.split_long_segment(slices, duration, last_slice)

            else:
                slices.append(start)
                last_slice = start
            last_start = start
            last_duration = duration

        data_entry[self.output_file_key] = sorted(list(set(slices)))

        return [DataEntry(data=data_entry)]


class SplitAudioFile(BaseParallelProcessor):
    """This processor splits audio files into segments based on provided timestamps.

    The class reads an audio file specified by the `input_file_key` and splits it into segments
    based on the timestamps provided in the `segments_key` field of the input data entry.
    The split audio segments are saved as individual WAV files in the specified `splited_audio_dir`
    directory. The `output_file_key` field of the data entry is updated with the path to the
    corresponding split audio file, and the `duration_key` field is updated with the duration
    of the split audio segment.

    Args:
        splited_audio_dir (str): The directory where the split audio files will be saved.
        segments_key (str, optional): The key in the manifest that contains the list of
            timestamps for splitting the audio. Defaults to "audio_segments".
        duration_key (str, optional): The key in the manifest where the duration of the
            split audio segment will be stored. Defaults to "duration".
        input_file_key (str, optional): The key in the manifest that contains the path
            to the input audio file. Defaults to "source_filepath".
        output_file_key (str, optional): The key in the manifest where the path to the
            split audio file will be stored. Defaults to "audio_filepath".

    Returns:
        A list of data entries, where each entry represents a split audio segment with
        the corresponding file path and duration updated in the data entry.
    """

    def __init__(
        self,
        splited_audio_dir: str,
        segments_key: str = "audio_segments",
        duration_key: str = "duration",
        input_file_key: str = "source_filepath",
        output_file_key: str = "audio_filepath",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.splited_audio_dir = splited_audio_dir
        self.segments_key = segments_key
        self.duration_key = duration_key
        self.input_file_key = input_file_key
        self.output_file_key = output_file_key

    def write_segment(self, data, samplerate, start_sec, end_sec, input_file):
        wav_save_file = os.path.join(
            self.splited_audio_dir,
            os.path.splitext(os.path.split(input_file)[1])[0],
            str(int(start_sec * 100)) + "-" + str(int(end_sec * 100)) + ".wav",
        )
        if not os.path.isfile(wav_save_file):
            data_sample = data[int(start_sec * samplerate) : int(end_sec * samplerate)]
            duration = len(data_sample) / samplerate
            os.makedirs(os.path.split(wav_save_file)[0], exist_ok=True)
            sf.write(wav_save_file, data_sample, samplerate)
            return wav_save_file, duration
        else:
            try:
                data, samplerate = sf.read(wav_save_file)
                duration = data.shape[0] / samplerate
            except Exception as e:
                logger.warning(str(e) + " file: " + wav_save_file)
                duration = -1.0
            return wav_save_file, duration

    def process_dataset_entry(self, data_entry: Dict):
        slices = data_entry[self.segments_key]
        input_file = data_entry[self.input_file_key]
        input_data, samplerate = sf.read(input_file)
        data_entries = []
        for i in range(len(slices[:-1])):
            wav_save_file, duration = self.write_segment(input_data, samplerate, slices[i], slices[i + 1], input_file)
            data_entry[self.output_file_key] = wav_save_file
            data_entry[self.duration_key] = duration
            data_entries.append(DataEntry(data=data_entry.copy()))
        return data_entries
