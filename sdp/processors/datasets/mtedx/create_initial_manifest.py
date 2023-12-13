import os
from pathlib import Path
from typing import List
import librosa
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive

MTEDX_URL = "https://www.openslr.org/resources/100/mtedx_{language_id}.tgz"

class CreateInitialManifestMTEDX(BaseParallelProcessor):
    def __init__(
            self,
            raw_data_dir: str,
            language_id: str,
            data_split: str,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.language_id = language_id
        self.data_split = data_split

    def prepare(self):
        """Downloading and extracting data (unless already done)."""
        os.makedirs(self.raw_data_dir, exist_ok=True)


        url = MTEDX_URL.format(language_id=self.language_id)
        if not (self.raw_data_dir / f"mls_{self.language_id}.tar.gz").exists():
            download_file(url, str(self.raw_data_dir))

        data_folder = extract_archive(str(self.raw_data_dir / os.path.basename(url)), str(self.raw_data_dir))


        self.audio_path_prefix = str(Path(data_folder) / 'data' / self.data_split/ "wav")
        self.vtt_path_prefix = str(str(Path(data_folder) / 'data' / self.data_split / "vtt"))

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