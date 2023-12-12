import glob
import os
import typing as tp
import fnmatch
import json

from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import download_file, extract_archive


def get_librispeech_url_list(names: list[str]) -> list[str]:
    urls = [
        "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "https://www.openslr.org/resources/12/dev-other.tar.gz",
        "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "https://www.openslr.org/resources/12/test-other.tar.gz",
        "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "https://www.openslr.org/resources/12/train-other-500.tar.gz",
    ]
    if "all" not in names:
        filtered_urls = [url for url in urls if url.split('/')[-1].split('.tar')[0] in names]
    else:
        filtered_urls = urls

    if len(filtered_urls) == 0:
        raise ValueError("No data found")

    return filtered_urls


class CreateInitialManifestLibrispeech(BaseProcessor):
    """Processor to create initial manifest for the Librispeech dataset.

    Dataset link: https://www.openslr.org/12

    Will download all files, extract tars and create manifest file with the 
    'audio_filepath' and 'text' fields 
    
    Args: 
        names (list[str]): Which data sets or their combinations shoudld be processed
            - options are:
            ["dev-clean"],
            ["dev-other"],
            ["test-clean"],
            ["test-other"],
            ["train-clean-100"],
            ["train-clean-360"],
            ["train-other-500"],
            ["all"] (for all datasets avalable)

        raw_data_dir (str): Path to folder where should the filed be donwloaded and extracted

    Returns:
       This processor generates an initial manifest file with the following fields::

            {
                "audio_filepath": <path to the audio file>,
                "text": <transcription>,
            }
    """

    def __init__(
        self,
        names: list[str],
        raw_data_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.names = names
        self.raw_data_dir = raw_data_dir

    def process_transcrip(self, file_path: str) -> list[dict[str, tp.Any]]:
        """Parse transcript file and put it inside manyfest
        We assume that flac files are located in the same directory as transcript file.
        """

        entries = []
        root = os.path.dirname(file_path)

        with open(file_path, encoding="utf-8") as fin:
            for line in fin:
                id, text = line[: line.index(" ")], line[line.index(" ") + 1:]
                transcript_text = text.lower().strip()

                flac_file = os.path.join(root, id + ".flac")

                entry = {}
                entry["audio_filepath"] = os.path.abspath(flac_file)
                entry["text"] = transcript_text
                entries.append(entry)
        return entries

    def process_data(self, data_folder: str, manifest_file: str) -> None:

        files = []
        entries = []

        for root, _, filenames in os.walk(data_folder):
            for filename in fnmatch.filter(filenames, "*.trans.txt"):
                files.append(os.path.join(root, filename))

        for file in files:
            result = self.process_transcrip(file)
            entries.extend(result)

        with open(manifest_file, "w") as fout:
            for m in entries:
                fout.write(json.dumps(m) + "\n")

    def download_extract_files(self, dst_folder: str) -> None:
        """downloading and extracting files"""

        os.makedirs(dst_folder, exist_ok=True)

        # downloading all files
        for file_url in get_librispeech_url_list(self.names):
            download_file(file_url, str(dst_folder))
        for data_file in glob.glob(f'{dst_folder}/*.tar.gz'):
            extract_archive(str(data_file), str(dst_folder), force_extract=True)

    def process(self):
        self.download_extract_files(self.raw_data_dir)
        self.process_data(self.raw_data_dir, self.output_manifest_file)
