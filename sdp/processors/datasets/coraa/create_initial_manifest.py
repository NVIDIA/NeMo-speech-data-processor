import glob
import os
from pathlib import Path
from typing import List
from huggingface_hub import snapshot_download
import pandas as pd

import rarfile  #Needs to be installed
import sox
from sox import Transformer

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import extract_archive

class CreateInitialManifestCORAA(BaseParallelProcessor):
    """
        Processor to create initial manifest file fo CORAA ASR dataset

         Dataset link: https://github.com/nilc-nlp/CORAA

         Args:
            raw_data_dir (str): the path to the directory in which all the data will be downloaded.
            extract_archive_dir (str): directory where the extracted data will be saved.
            data_split (str): "train", "dev" or "test".
            resampled_audio_dir (str): the directory where the resampled wav files will be stored.
            already_extracted (bool): if True, we will not try to extract the raw data.
                Defaults to False.
            already_downloaded (bool): if True, we will not try to download files.
            target_samplerate (int): sample rate (Hz) to use for resampling. This parameter will
                Defaults to 16000.
            target_nchannels (int): number of channels to create during resampling process.
                Defaults to 1.
            exclude_dataset: list: list of the dataset names that will be excluded when creating initial manifest.
                Options 'SP2010', 'C-ORAL-BRASIL I', 'NURC-Recife', 'TEDx Talks', 'ALIP'

    """
    def __init__(
            self,
            raw_data_dir: str,
            extract_archive_dir: str,
            data_split: str,
            resampled_audio_dir: str,
            already_extracted: bool = False,
            already_downloaded: bool = False,
            target_samplerate: int = 16000,
            target_nchannels: int = 1,
            exclude_dataset: list = [],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.extract_archive_dir = extract_archive_dir
        self.data_split = data_split
        self.already_downloaded = already_downloaded
        self.already_extracted = already_extracted
        self.exclude_dataset = exclude_dataset
        self.resampled_audio_dir = resampled_audio_dir
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def prepare(self):
        """Downloading and extracting data (unless already done)."""
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.resampled_audio_dir, exist_ok=True)
        os.makedirs(self.extract_archive_dir, exist_ok=True)
        if not self.already_downloaded:
            snapshot_download(repo_id="gabrielrstan/CORAA-v1.1", repo_type='dataset', local_dir=self.raw_data_dir)
        if not self.already_extracted:

            if self.data_split == 'train':
                first_rar_file = glob.glob(str(self.raw_data_dir) + "/train_dividido"+f"/*{self.data_split}*1.rar")
                if first_rar_file and not isinstance(first_rar_file, str):
                    first_rar_file = first_rar_file[0]

                if rarfile.is_rarfile(first_rar_file):
                    rar = rarfile.RarFile(first_rar_file)
                    rar.extractall(path=self.extract_archive_dir)
            else:

                zip_files =glob.glob(str(self.raw_data_dir) + f"/*{self.data_split}.zip")
                if not zip_files:
                    raise RuntimeError(
                        f"Did not find any file matching {self.raw_data_dir}/*.zip. "
                        "Make sure your download passed succesfully."
                    )
                elif len(zip_files) > 1:
                    raise RuntimeError(
                        f"Expecting exactly one {self.data_split}.zip file in directory {self.raw_data_dir}"
                    )

                extract_archive(zip_files[0], self.extract_archive_dir)
        self.transcription_file = self.raw_data_dir / f"metadata_{self.data_split}_final.csv"
        self.audio_path_prefix = self.extract_archive_dir

    def read_manifest(self):
        self.df = pd.read_csv(self.transcription_file)
        data_entries = self.df[~self.df['dataset'].isin(self.exclude_dataset)][['file_path','text']]
        res = [tuple(row[1]) for row in data_entries.iterrows()]
        return res

    def process_dataset_entry(self, data_entry) -> List[DataEntry]:

        file_path, text = data_entry
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        transcript_text = text.strip()

        audio_path = os.path.join(self.audio_path_prefix, file_path)
        output_wav_path = os.path.join(self.resampled_audio_dir, file_name + ".wav")

        if not os.path.exists(output_wav_path):
            tfm = Transformer()
            tfm.rate(samplerate=self.target_samplerate)
            tfm.channels(n_channels=self.target_nchannels)
            tfm.build(input_filepath=audio_path, output_filepath=output_wav_path)

        data = {
            "audio_filepath": output_wav_path,
            "duration": float(sox.file_info.duration(output_wav_path)),
            "text": transcript_text,
        }

        return [DataEntry(data=data)]
