import os
from pathlib import Path
from typing import List
import librosa
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive

MTEDX_URL = "https://www.openslr.org/resources/100/mtedx_{language_id}.tgz"

class CreateInitialManifestMTEDX(BaseParallelProcessor):
    """Processor to create initial manifest for the Multilingual TEDx (MTedX dataset.

        Dataset link: https://www.openslr.org/100/

        Downloads dataset for the specified language and creates initial manifest with the provided
        audio and vtt files.

        Args:
            raw_data_dir (str): the directory where the downloaded data will be/is saved.
                                This is also where the extracted and processed data will be.
            data_split (str): "train", "dev" or "test".
            language_id (str): the ID of the language of the data. E.g., "en", "es", "it", etc.
            target_samplerate (int): sample rate (Hz) to use for resampling.
            already_extracted: (bool): if True, we will not try to extract the raw data.
                Defaults to False.

        Returns:
            This processor generates an initial manifest file with the following fields::

                {
                    "audio_filepath": <path to the audio file>,
                    "vtt_filepath": <path to the corresponding vtt file>
                    "duration": <duration of the audio in seconds>
                }
        """
    def __init__(
            self,
            raw_data_dir: str,
            language_id: str,
            data_split: str,
            already_extracted: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.language_id = language_id
        self.data_split = data_split
        self.already_extracted = already_extracted

    def prepare(self):
        """Downloading and extracting data (unless already done)."""
        os.makedirs(self.raw_data_dir, exist_ok=True)


        url = MTEDX_URL.format(language_id=self.language_id)
        if not (self.raw_data_dir / f"mtedx_{self.language_id}.tgz").exists():
            download_file(url, str(self.raw_data_dir))
        if not self.already_extracted:
            data_folder = extract_archive(str(self.raw_data_dir / os.path.basename(url)), str(self.raw_data_dir))
            self.audio_path_prefix = Path(data_folder) / 'data' / self.data_split / "wav"
            self.vtt_path_prefix = Path(data_folder) / 'data' / self.data_split / "vtt"        
        else:
            data_folder = self.raw_data_dir 
            self.audio_path_prefix = Path(data_folder) / f"{self.language_id}-{self.language_id}" / 'data' / self.data_split / "wav"
            self.vtt_path_prefix = Path(data_folder) / f"{self.language_id}-{self.language_id}" / 'data' / self.data_split / "vtt"        

    def read_manifest(self):
        """Creating entries of initial manifest with flac and vtt files"""
        audio_filepaths = []
        for audio_file in os.listdir(self.audio_path_prefix):
            vtt_filepath = os.path.join(self.vtt_path_prefix, audio_file.split('.')[0] + "." + self.language_id  + ".vtt")
            audio_filepath = os.path.join(self.audio_path_prefix, audio_file)
            audio_filepaths.append((audio_filepath, vtt_filepath))
        return audio_filepaths

    def process_dataset_entry(self, data_entry) -> List[DataEntry]:
        """Processing the data entries."""
        audio_filepath, vtt_filepath = data_entry

        data = {
            'audio_filepath': audio_filepath,
            'vtt_filepath': vtt_filepath,
            'duration': float(librosa.get_duration(path=audio_filepath)),
        }
        return [DataEntry(data=data)]
