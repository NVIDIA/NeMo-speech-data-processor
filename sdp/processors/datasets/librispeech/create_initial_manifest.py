import glob
import os
import urllib.request
from pathlib import Path
import typing as tp
import fnmatch
import json

import pandas as pd
from sox import Transformer

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry, BaseProcessor
from sdp.utils.common import download_file, extract_archive

def get_librispeech_url_list() -> list[str]:
    # TODO: automatically get the urls

    # urls: list[str] = [
    #     "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    #     "https://www.openslr.org/resources/12/dev-other.tar.gz",
    #     "https://www.openslr.org/resources/12/test-clean.tar.gz",
    #     "https://www.openslr.org/resources/12/test-other.tar.gz",
    #     "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    #     "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    #     "https://www.openslr.org/resources/12/train-other-500.tar.gz",
    #     "https://www.openslr.org/resources/12/intro-disclaimers.tar.gz",
    #     "https://www.openslr.org/resources/12/original-mp3.tar.gz",
    #     "https://www.openslr.org/resources/12/original-books.tar.gz",
    #     "https://www.openslr.org/resources/12/raw-metadata.tar.gz"
    # ]

    urls: list[str] = [
        "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "https://www.openslr.org/resources/12/dev-other.tar.gz"
    ]
    return urls

class CreateInitialManifestLibrispeech(BaseProcessor):
    """Processor to create initial manifest for the Librispeech dataset.

    Dataset link: https://www.openslr.org/12

    Will download all files, extract tars and split wav files based on the
    provided durations in the transcripts."""

    def __init__(
        self,
        # output_manifest_file: str, 
        raw_data_dir: str,
        resampled_audio_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # self.output_manifest_file = Path(output_manifest_file)
        self.raw_data_dir = Path(raw_data_dir)
        self.resampled_audio_dir = Path(resampled_audio_dir)

    def process_transcrip(self, file_path: str, dst_folder: str) -> list[dict[str, tp.Any]]:
        """
        Converts flac files to a mav files from a given transcript. 
        We assume that flac files are located in the same directory as mav files.
        """


        entries = []
        root = os.path.dirname(file_path)

        with open(file_path, encoding="utf-8") as fin:
            for line in fin:
                id, text = line[: line.index(" ")], line[line.index(" ") + 1 :]
                transcript_text = text.lower().strip()

                # # Convert FLAC file to WAV
                flac_file = os.path.join(root, id + ".flac")
                # wav_file = os.path.join(dst_folder, id + ".wav")
                # if not os.path.exists(wav_file):
                #     Transformer().build(flac_file, wav_file)

                entry = {}
                entry["audio_filepath"] = os.path.abspath(flac_file)
                entry["text"] = transcript_text
                entries.append(entry)
        return entries

    def process_data(self, data_folder: str, dst_folder: str, manifest_file: str) -> None:
        """
        Converts flac to wav and build manifests's json
        """

        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        files = []
        entries = []

        for root, dirnames, filenames in os.walk(data_folder):
            for filename in fnmatch.filter(filenames, "*.trans.txt"):
                files.append(os.path.join(root, filename))

        for file in files:
            result = self.process_transcrip(file, dst_folder)
            entries.extend(result)

        with open(manifest_file, "w") as fout:
            for m in entries:
                fout.write(json.dumps(m) + "\n")


    def download_extract_files(self, dst_folder: str) -> None:
        """downloading and extracting files"""

        os.makedirs(dst_folder, exist_ok=True)

        # downloading all files
        for file_url in get_librispeech_url_list():
            download_file(file_url, str(dst_folder))
        for data_file in glob.glob(f'{dst_folder}/*.tar.gz'):
            extract_archive(data_file, dst_folder, force_extract=True)

    def process(self):
        self.download_extract_files(self.raw_data_dir)
        self.process_data(self.raw_data_dir, self.resampled_audio_dir, self.output_manifest_file)
