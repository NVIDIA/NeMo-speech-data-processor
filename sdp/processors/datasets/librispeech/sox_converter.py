import os

from sox import Transformer

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class Flac2Wav(BaseParallelProcessor):
    """
    Processor for converting flac files to wav using sox

    Args: 
        resampled_audio_dir (str): path where to put converted audiofiles
        input_field (str): where to read path to audio to convert
        output_field (str): where to put converted audio path

    Returns:
        All the same fields as in the input manifest without input_field plus output_field

    """
    def __init__(
        self,
        resampled_audio_dir: str,
        input_field: str,
        output_field: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        self.resampled_audio_dir = resampled_audio_dir

    def prepare(self):
        os.makedirs(os.path.split(self.output_manifest_file)[0], exist_ok=True)
        os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        flac_file = data_entry[self.input_field]
        key = os.path.splitext(flac_file)[0].split("/")[-1].split(".")[0]
        key = "_".join(key.split("-"))
        wav_file = os.path.join(self.resampled_audio_dir, key) + ".wav"

        if not os.path.isfile(wav_file):
            Transformer().build(flac_file, wav_file)

        data_entry[self.output_field] = wav_file

        return [DataEntry(data=data_entry)]
