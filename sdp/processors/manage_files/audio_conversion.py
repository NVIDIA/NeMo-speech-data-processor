import os
from sox import Transformer

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import ffmpeg_convert


class FfmpegConvert(BaseParallelProcessor):
    """
    Processor for converting video or audio files to audio using FFmpeg and updating the dataset with the path to the resampled audio.
    If ``id_key`` is not None, the output file path will be ``<resampled_audio_dir>/<id_key>.wav``.
    If ``id_key`` is None, the output file path will be ``<resampled_audio_dir>/<input file name without extension>.wav``.

    .. note:: ``id_key`` can be used to create subdirectories inside ``resampled_audio_dir`` (by using forward slashes ``/``).
        e.g. if ``id_key`` takes the form ``dir_name1/dir_name2/filename``, the output file path will be

        ``<resampled_audio_dir>/dir_name1/dirname2/filename.wav``.

    Args:
        converted_audio_dir (str): The directory to store the resampled audio files.
        input_file_key (str): The field in the dataset representing the path to the input video or audio files.
        output_file_key (str): The field in the dataset representing the path to the resampled audio files with ``output_format``. If ``id_key`` is None, the output file path will be ``<resampled_audio_dir>/<input file name without extension>.wav``.
        id_key (str): (Optional) The field in the dataset representing the unique ID or identifier for each entry. If ``id_key`` is not None, the output file path will be ``<resampled_audio_dir>/<id_key>.wav``. Defaults to None.
        output_format (str): (Optional) Format of the output audio files. Defaults to `wav`.
        target_samplerate (int): (Optional) The target sampling rate for the resampled audio. Defaults to 16000.
        target_nchannels (int): (Optional) The target number of channels for the resampled audio. Defaults to 1.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        converted_audio_dir: str,
        input_file_key: str,
        output_file_key: str,
        id_key: str = None,
        output_format: str = "wav",
        base_dir: str = None,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.converted_audio_dir = converted_audio_dir
        self.input_file_key = input_file_key
        self.output_file_key = output_file_key
        self.output_format = output_format
        self.id_key = id_key
        self.base_dir = base_dir
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def prepare(self):
        assert self.output_format == "wav", "Currently only wav format is supported"
        os.makedirs(self.converted_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        input_file = data_entry[self.input_file_key]
        if self.id_key:
            key = data_entry[self.id_key]
            os.makedirs(os.path.join(self.converted_audio_dir, *key.split("/")[:-1]), exist_ok=True)
        else:
            key = os.path.splitext(input_file)[0].split("/")[-1]

        if self.base_dir:
            new_dir = os.path.dirname(os.path.relpath(input_file, self.base_dir))
            os.makedirs(os.path.join(self.converted_audio_dir, new_dir), exist_ok=True)

            key = os.path.join(new_dir, key)

        audio_file = os.path.join(self.converted_audio_dir, key) + "." + self.output_format

        if not os.path.isfile(audio_file):
            ffmpeg_convert(input_file, audio_file, self.target_samplerate, self.target_nchannels)

        data_entry[self.output_file_key] = audio_file
        return [DataEntry(data=data_entry)]


class SoxConvert(BaseParallelProcessor):
    """
    Processor for converting audio files from one format to another using Sox,
    and updating the dataset with the path to the converted audio files.

    Args:
        converted_audio_dir (str): Directory to store the converted audio files.
        input_audio_file_key (str): Field in the dataset representing the path to input audio files.
        output_audio_file_key (str): Field to store the path to the converted audio files in the dataset.
        output_format (str): Format of the output audio files (e.g., 'wav', 'mp3').
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.
    """

    def __init__(
        self,
        converted_audio_dir: str,
        input_audio_file_key: str,
        output_audio_file_key: str,
        output_format: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_audio_file_key = input_audio_file_key
        self.output_audio_file_key = output_audio_file_key
        self.converted_audio_dir = converted_audio_dir
        self.output_format = output_format

    def prepare(self):
        os.makedirs(self.converted_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        audio_file = data_entry[self.input_audio_file_key]

        key = os.path.splitext(audio_file)[0].split("/")[-1]
        converted_file = os.path.join(self.converted_audio_dir, key) + f".{self.output_format}"

        if not os.path.isfile(converted_file):
            transformer = Transformer()
            transformer.build(audio_file, converted_file)

        data_entry[self.output_audio_file_key] = converted_file
        return [DataEntry(data=data_entry)]