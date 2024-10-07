import os

import soundfile as sf

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

class CreateInitialManifestHuggingFace(BaseParallelProcessor):
    """Processor to create initial manifest for HuggingFace dataset.

    Downloads HuggingFace dataset and creates an initial manifest.

    Args:
        dataset_name (str): the name of the dataset. E.g., "tarteel-ai/everyayah"
        raw_data_dir (str): the path to the directory containing the raw dataset files.
        resampled_audio_dir (str): directory where the resampled audio will be saved.
        data_split (str): "train", "validation" or "test".
        target_samplerate (int): sample rate (Hz) to use for resampling.
            Defaults to 16000.
        target_nchannels (int): number of channels to create during resampling process.
            Defaults to 1.

    Returns:
        This processor generates an initial manifest file with the following fields::
        
            {
                "audio_filepath": <path to the audio file>,
                "duration": <duration of the audio in seconds>,
                "text": <transcription (with capitalization and punctuation)>,
            }
    """

    def __init__(
        self,
        dataset_name: str,
        resampled_audio_dir: str,
        data_split: str,
        target_samplerate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_split = data_split
        self.target_samplerate = target_samplerate
        self.resampled_audio_dir = resampled_audio_dir
        self.dataset_name = dataset_name

    def prepare(self):
        os.makedirs(self.resampled_audio_dir, exist_ok=True)
        os.makedirs(self.resampled_audio_dir, exist_ok=True)
        

    def read_manifest(self):
        import datasets
        
        self.dataset = datasets.load_dataset(self.dataset_name, split=self.data_split)
        return range(0, len(self.dataset))

    def process_dataset_entry(self, data_id):
        sample_data = self.dataset[data_id]
        sample_audio = sample_data["audio"]["array"]
        audio_filepath = os.path.join(self.resampled_audio_dir, f"{data_id}.wav")
        sf.write(
            audio_filepath,
            sample_audio,
            self.target_samplerate,
        )
        duration = len(sample_audio) / self.target_samplerate
        text = sample_data["text"]

        return [
            DataEntry(
                data={
                    "audio_filepath": os.path.join("audios", f"{data_id}.wav"),
                    "duration": duration,
                    "text": text,
                }
            )
        ]