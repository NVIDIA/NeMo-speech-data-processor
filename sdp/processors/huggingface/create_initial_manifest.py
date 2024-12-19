import os
import glob

import soundfile as sf

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.logging import logger
from typing import Optional

class CreateInitialManifestHuggingFace(BaseParallelProcessor):
    """Processor to create initial manifest for HuggingFace dataset.

    Downloads HuggingFace dataset and creates an initial manifest.

    Args:
        dataset_name (str): the name of the dataset. E.g., "tarteel-ai/everyayah"
        raw_data_dir (str): the path to the directory containing the raw dataset files.
        resampled_audio_dir (str): directory where the resampled audio will be saved.
        data_split (str): "train", "validation" or "test".
        already_downloaded (bool): if True, we will not try to load dataset from HuggingFace.
            Defaults to False.
        target_samplerate (int): sample rate (Hz) to use for resampling.
            Defaults to 16000.

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
        raw_data_dir: Optional[str] = None,
        already_downloaded: bool = False,
        target_samplerate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_split = data_split
        self.target_samplerate = target_samplerate
        self.resampled_audio_dir = resampled_audio_dir
        self.dataset_name = dataset_name
        self.raw_data_dir = raw_data_dir
        self.already_downloaded = already_downloaded

    def prepare(self):
        os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def read_manifest(self):
        import datasets
        
        # checking if dataset should be loaded from disk
        if self.already_downloaded:
            if os.path.exists(self.raw_data_dir):
                hf_files = glob.glob(f'{self.raw_data_dir}/*.hf')
                self.dataset = datasets.load_from_disk(os.path.join(self.raw_data_dir, hf_files[0]))
            else:
                logger.info("Dataset not found locally. Initiating download from Hugging Face.")
        else:
            logger.info(f"Initiating download of dataset '{self.dataset_name}' from Hugging Face.")
            self.dataset = datasets.load_dataset(self.dataset_name, split=self.data_split)
            logger.info(f"Finished download of dataset '{self.dataset_name}' from Hugging Face.")
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