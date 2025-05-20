# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import json
from tqdm import tqdm
import tempfile
import importlib.util

from sdp.processors import ListToEntries
from sdp.processors.huggingface.huggingface_hub import ListRepoFiles, SnapshotDownload, HfHubDownload
from sdp.logging import logger


class ListYodas2Data(ListRepoFiles):
    """
    Processor for generating a manifest of the YODAS2 dataset stored on the Hugging Face Hub.

    This processor supports two modes:
        1. Using the `meta.py` file from the dataset repo (if `use_metadata=True`), which provides
        structured metadata including the number of shards per language subset.
        
        2. Parsing the repository's file list directly to infer the manifest structure (default mode).

    Args:
        use_metadata (bool): Whether to use `meta.py` to generate the manifest (default: False).
        **kwargs: Passed to the parent class `sdp.processors.ListRepoFiles`, including Hugging Face repo config.
    
    Returns:
        A line-delimited JSON manifest, where each line represents information about a YODAS2 dataset shard and 
        includes keys pointing to the audio files (in .tar.gz format), transcriptions (in .json format), and 
        durations (in .txt format) stored in the repository.
    """

    def __init__(self, use_metadata: bool = False, **kwargs):
        # Initialize parent class with hardcoded repo_id and repo_type for YODAS2
        super().__init__(repo_id="espnet/yodas2", repo_type="dataset", **kwargs)
        self.use_metadata = use_metadata

    def process(self):
        from huggingface_hub import hf_hub_download

        """
        Main entry point for generating the YODAS2 manifest.

        If `use_metadata=True`, it uses the `meta.py` file to generate expected paths.
        Otherwise, it lists all files from Hugging Face and organizes them by language and shard.
        The final manifest is written as line-delimited JSON.
        """
        if self.use_metadata:
            metadata = None
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download the meta.py script containing shard information
                yodas_metafile = hf_hub_download(
                    repo_id="espnet/yodas2",
                    filename="meta.py",
                    repo_type="dataset",
                    local_dir=tmp_dir
                )
                # Dynamically import meta.py
                spec = importlib.util.spec_from_file_location("script", yodas_metafile)
                metadata = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata)

            # Write the manifest based on metadata.lang2shard_cnt
            with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
                for lang_subset in sorted(metadata.lang2shard_cnt.keys()):
                    for shard_no in range(metadata.lang2shard_cnt[lang_subset]):
                        shard_id = str(shard_no).zfill(8)
                        data_entry = {
                            "lang_subset": lang_subset,
                            "shard_id": shard_id,
                            "audio_key": f"data/{lang_subset}/audio/{shard_id}.tar.gz",
                            "duration_key": f"data/{lang_subset}/duration/{shard_id}.txt",
                            "text_key": f"data/{lang_subset}/text/{shard_id}.json",
                        }
                        line = json.dumps(data_entry)
                        fout.writelines(f"{line}\n")
        else:
            logger.info("Receiving files list of espnet/yodas2 dataset from Hugging Face...")
            self.list_repo_files()
            logger.info("Metadata has been successfully received. Aggregating filenames into shards...")

            lang2shard_files = {}

            # Parse each file path and organize by language subset and shard
            for file in tqdm(self.files):
                if not file.startswith("data/"):
                    continue

                path = Path(file)
                lang_subset = path.parts[1]  # e.g., "en000"
                if lang_subset not in lang2shard_files:
                    lang2shard_files[lang_subset] = {}
                lang_shards = lang2shard_files[lang_subset]

                shard_no = path.parts[3].split('.')[0]  # e.g., "00000001"
                if shard_no not in lang_shards:
                    lang_shards[shard_no] = {}
                shard_files = lang_shards[shard_no]

                data_type = path.parts[2]  # e.g., "audio", "duration", "text"
                shard_files[data_type] = file

            logger.info("Writing data into manifest...")

            # Write aggregated entries to manifest
            with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
                for lang_subset in sorted(lang2shard_files.keys()):
                    lang_subset_shards = lang2shard_files[lang_subset]
                    for shard_id in sorted(lang_subset_shards.keys()):
                        data_entry = {
                            "lang_subset": lang_subset,
                            "shard_id": shard_id,
                        }

                        # Add keys for each data type (audio_key, duration_key, etc.)
                        shard_data = lang_subset_shards[shard_id]
                        for data_type in sorted(shard_data.keys()):
                            data_entry[f"{data_type}_key"] = shard_data[data_type]

                        line = json.dumps(data_entry)
                        fout.writelines(f"{line}\n")

        logger.info("Metadata successfully saved!")
        

class SnapshotDownloadYodas2Data(SnapshotDownload):
    """
    A specialized processor for downloading the YODAS2 dataset from Hugging Face
    and updating the input manifest with local file paths to the downloaded files.

    This class:
    - Loads an input manifest that contains HF repo-relative paths to audio, text, and duration files.
    - Downloads only the referenced files using Hugging Face `snapshot_download` with `allow_patterns`.
    - Updates the manifest with paths to the locally downloaded files (under keys `local_audio`, `local_duration`, `local_text`).

    Args:
        output_manifest_file (str): Path to write the updated output manifest file.
        input_manifest_file (str): Path to the input manifest listing the files to fetch.
            Each line must be a JSON object that contains the following **optional but expected** fields:

                - "audio_key" (str): Relative path in the Hugging Face dataset repository to the audio file.
                - "duration_key" (str): Relative path to a file containing the duration of the audio sample.
                - "text_key" (str): Relative path to the transcription or label file for the audio sample.

                At least one of these keys should be present in each entry. These keys are used to determine
                which files to download from the Hugging Face dataset and to map them to local files after download.
        
        **kwargs: Additional arguments passed to the base SnapshotDownload processor.
    
    Returns:
        A line-delimited JSON manifest, where each line represents a sample entry
        and contains absolute local paths to the audio, duration, and text files.
    
    .. admonition:: Example
                
        Input line::
            
            {
                "lang_subset": "en000", 
                "shard_id": "00000000", 
                "audio_key": "data/en000/audio/00000000.tar.gz", 
                "duration_key": "data/en000/duration/00000000.txt", 
                "text_key": "data/en000/text/00000000.json"
            }
 
        Output line::
            
            {
                "lang_subset": "en000", 
                "shard_id": "00000000", 
                "audio_key": "data/en000/audio/00000000.tar.gz", 
                "duration_key": "data/en000/duration/00000000.txt", 
                "text_key": "data/en000/text/00000000.json", 
                "local_audio": "/path/to/data/en000/audio/00000000.tar.gz", 
                "local_duration": "/path/to/data/en000/duration/00000000.txt"}
                "local_text": "/path/to/data/en000/text/00000000.json"
            }
    
    """

    def __init__(self, **kwargs):
        # Hardcoded to download the espnet/yodas2 dataset from Hugging Face
        if not 'snapshot_download_args' in kwargs:
            kwargs['snapshot_download_args'] = dict()
        kwargs['snapshot_download_args']['repo_id'] = 'espnet/yodas2'
        kwargs['snapshot_download_args']['repo_type'] = 'dataset'

        super().__init__(**kwargs)

    def write_output_manifest_file(self):
        """
        Write a new manifest file that includes local paths to audio, text, and duration files.

        For each line in the input manifest, checks for keys `audio_key`, `duration_key`, and `text_key`.
        If the corresponding file exists locally after download, adds the local path under
        `local_audio`, `local_duration`, and `local_text`, respectively.
        """

        # Open input manifest for reading and output manifest for writing
        with open(self.input_manifest_file, 'r', encoding='utf8') as fin, open(self.output_manifest_file, 'w', encoding='utf8') as fout:
            for line in fin:
                sample = json.loads(line)

                # Try to find and attach local paths for each possible file type
                audio_file = sample.get('audio_key')
                if audio_file:
                    local_audio_file = os.path.join(self.local_dir, audio_file)
                    if os.path.exists(local_audio_file):
                        sample['local_audio'] = local_audio_file

                duration_file = sample.get('duration_key')
                if duration_file:
                    local_duration_file = os.path.join(self.local_dir, duration_file)
                    if os.path.exists(local_duration_file):
                        sample['local_duration'] = local_duration_file

                text_file = sample.get('text_key')
                if text_file:
                    local_text_file = os.path.join(self.local_dir, text_file)
                    if os.path.exists(local_text_file):
                        sample['local_text'] = local_text_file

                # Write the modified sample to the output manifest
                line = json.dumps(sample)
                fout.writelines(f'{line}\n')

    def process(self):
        """
        Main processing function: collects the list of files to download,
        performs the download, and then writes the output manifest.

        This method:
        - Scans the input manifest to extract all relevant repo-relative file paths.
        - Adds them to the `allow_patterns` argument of `snapshot_download_kwargs`.
        - Triggers the download of only the needed files.
        - Updates the manifest with local paths.
        """
        allow_patterns = []

        # Parse input manifest to extract all file paths to allow in the snapshot
        with open(self.input_manifest_file, 'r', encoding='utf8') as fin:
            for line in fin:
                sample = json.loads(line)

                audio_file = sample.get('audio_key')
                if audio_file:
                    allow_patterns.append(audio_file)

                duration_file = sample.get('duration_key')
                if duration_file:
                    allow_patterns.append(duration_file)

                text_file = sample.get('text_key')
                if text_file:
                    allow_patterns.append(text_file)

        # Restrict snapshot download to only needed files
        self.snapshot_download_kwargs['allow_patterns'] = allow_patterns

        # Download the snapshot and write updated manifest
        self.download()
        self.write_output_manifest_file()


class HfHubDownloadYodas2Data(HfHubDownload):
    def __init__(self, filename_field: str = 'audio_key', output_filepath_field = 'local_audio', **kwargs):
        if not 'hf_hub_download_args' in kwargs:
            kwargs['hf_hub_download_args'] = dict()
        kwargs['hf_hub_download_args']['repo_id'] = 'espnet/yodas2'
        kwargs['hf_hub_download_args']['repo_type'] = 'dataset'

        super().__init__(filename_field = filename_field, output_filepath_field = output_filepath_field, **kwargs)
    
    def process(self):
        super().process()

class CreateInitialManifestYodas2(ListToEntries):
    """
    A dataset processor specialized for the YODAS2 dataset.

    This processor extends ``ListToEntries`` to:
    
    - Expand each input record that contains a list of audio items (e.g., under a ``shards`` or ``files`` field).
    - Append duration and YODAS ID metadata to each resulting entry using a provided durations file.

    Each input entry must include a field (e.g., ``local_duration``) pointing to a `.txt` file
    with lines in the format:

    .. code-block:: text

        <yodas_id> <duration>

    Example input line:

    .. code-block:: json

        {
            "lang_subset": "en000",
            "shard_id": "00000000",
            "audio_key": "data/en000/audio/00000000.tar.gz",
            "duration_key": "data/en000/duration/00000000.txt",
            "text_key": "data/en000/text/00000000.json",
            "src_lang": "en",
            "local_audio": "/data3/sdp_test/en/data/en000/audio/00000000.tar.gz",
            "local_duration": "/data3/sdp_test/en/data/en000/duration/00000000.txt",
            "extracted_audios": [
                "/path/to/data/en000/00000000/YuseO8GhcWk.wav",
                "/path/to/data/en000/00000000/Y9zJ8mT5Bou.wav"
            ]
        }
    
    Args:
        **kwargs: Passed directly to the ``sdp.processors.ListToEntries`` base processor.

    Returns:
        A manifest where each entry corresponds to one YODAS2 sample with the following structure:

        .. code-block:: json

            {
                "lang_subset": "en000",
                "shard_id": "00000000",
                "src_lang": "en",
                "source_audio_filepath": "/path/to/data/en000/00000000/YuseO8GhcWk.wav",
                "yodas_id": "YuseO8GhcWk",
                "duration": 216.9
            }
    """
    def __init__(self, **kwargs):
        # Initialize with ListToEntries configuration
         super().__init__(**kwargs)
        
    def get_samples_durations(self, durations_filepath: str):
        """
        Parse durations file into a dict mapping yodas_id -> duration.

        Args:
            durations_filepath (str): Path to durations.txt file, where each line
                contains "<yodas_id> <duration>".

        Returns:
            dict[str, float]: Mapping from yodas_id to duration.
        """
        durations = dict()
        with open(durations_filepath, 'r') as durations_txt:
            for line in durations_txt:
                yodas_id, duration = line.strip().split()
                durations[yodas_id] = float(duration)
        return durations
    
    def process_dataset_entry(self, data_entry):
        """
        Process a single dataset entry.

        - Reads durations file from the entry
        - Expands list field using base ListToEntries
        - Extracts `yodas_id` from audio path and adds `duration` if found

        Args:
            data_entry (dict): Original manifest entry with a list field and a pointer
                to a local durations.txt file.

        Returns:
            List[DataEntry]: Processed and filtered list of entries with yodas_id and duration.
        """
        # Load duration metadata for current group of items
        durations = self.get_samples_durations(data_entry['local_duration'])

        # Expand the list of items into individual entries (inherited logic)
        data_entries = super().process_dataset_entry(data_entry)

        yodas_entries = []
        for entry in data_entries:
            # Extract YODAS ID from filename (e.g., "YABC1234" from "YABC1234.wav")
            yodas_id = os.path.basename(entry.data['source_audio_filepath']).split('.')[0]
            entry.data['yodas_id'] = yodas_id

            # Attach duration if available
            if yodas_id in durations:
                entry.data['duration'] = durations[yodas_id]
                yodas_entries.append(entry)
            else:
                logger.warning(f'Skipping `{yodas_id}` because there is no duration info in metadata.')
                
        return yodas_entries