# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import librosa
from sox import Transformer

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive

MLS_URL_NO_OPUS = "https://dl.fbaipublicfiles.com/mls/mls_{language}.tar.gz"
MLS_URL_OPUS = "https://dl.fbaipublicfiles.com/mls/mls_{language}_opus.tar.gz"


class CreateInitialManifestMLS(BaseParallelProcessor):
    """Processor to create initial manifest for the Multilingual LibriSpeech (MLS) dataset.

    Dataset link: https://www.openslr.org/94/

    Downloads and unzips raw MLS data for the specified language,
    and creates an initial manifest using the transcripts provided in the raw data.

    Args:
        raw_data_dir (str): the directory where the downloaded data will be/is saved.
            This is also where the extracted and processed data will be.
        language (str): the language of the data you wish to be downloaded.
            This will be used to format the URL from which we attempt to download the data.
            E.g., "english", "italian", "spanish", etc.
        data_split (str): "train", "dev" or "test".
        resampled_audio_dir (str or None): if specified, the directory where the resampled
            wav files will be stored. If not specified, the audio will not be resampled and
            the parameters ``target_samplerate`` and ``target_nchannels`` will be ignored.
        target_samplerate (int): sample rate (Hz) to use for resampling. This parameter will
            be ignored if ``resampled_audio_dir`` is ``None``.
            Defaults to 16000.
        target_nchannels (int): number of channels to create during resampling process. This
            parameter will be ignored if ``resampled_audio_dir`` is ``None``.
            Defaults to 1.
        use_opus_archive (bool): if ``True``, will use the version of the archive file which
            contains audio files saved in the OPUS format, instead of FLAC. The OPUS files take up
            less memory than the FLAC files, at the cost of the OPUS files being lower quality than
            the FLAC files.
            If ``True``, the parameter ``resampled_audio_dir`` must be ``None``, as resampling OPUS
            audio files is currently not supported.
            Defaults to False.

    Returns:
        This processor generates an initial manifest file with the following fields::

            {
                "audio_filepath": <path to the audio file>,
                "duration": <duration of the audio in seconds>,
                "text": <transcription>,
            }
    """

    def __init__(
        self,
        raw_data_dir: str,
        language: str,
        data_split: str,
        resampled_audio_dir: Optional[str],
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        use_opus_archive: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.language = language
        self.data_split = data_split
        self.resampled_audio_dir = Path(resampled_audio_dir) if resampled_audio_dir else None
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels
        self.use_opus_archive = use_opus_archive

        # validate params
        if self.use_opus_archive and self.resampled_audio_dir:
            raise ValueError(
                f"`use_opus_archive` is True and `resampled_audio_dir` is not None, but we currently do not"
                " support resampling OPUS-format audio, please either set `use_opus_archive` to False or"
                " resampled_audio_dir to None."
            )

        if not resampled_audio_dir:
            logger.info(
                "`resampled_audio_dir` is None => will not attempt to resample audio. Please note if you have"
                " specified `target_samplerate` or `target_nchannels`, they will be ignored."
            )



        # will be initialized in self.prepare method
        self.audio_path_prefix = None
        self.transcription_file = None

    def prepare(self):
        """Downloading and extracting data (unless already done)."""
        os.makedirs(self.raw_data_dir, exist_ok=True)

        if self.use_opus_archive:
            url = MLS_URL_OPUS.format(language=self.language)
            if not (self.raw_data_dir / f"mls_{self.language}_opus.tar.gz").exists():
                download_file(url, str(self.raw_data_dir))

        else:
            url = MLS_URL_NO_OPUS.format(language=self.language)
            if not (self.raw_data_dir / f"mls_{self.language}.tar.gz").exists():
                download_file(url, str(self.raw_data_dir))

        data_folder = extract_archive(str(self.raw_data_dir / os.path.basename(url)), str(self.raw_data_dir))

        self.audio_path_prefix = str(Path(data_folder) / self.data_split / "audio")
        self.transcription_file = str(Path(data_folder) / self.data_split / "transcripts.txt")

    def read_manifest(self):
        """Reading the initial data line-by-line."""
        if self.transcription_file is None:
            raise RuntimeError("self.process has to be called before processing the data.")

        with open(self.transcription_file, "rt", encoding="utf8") as fin:
            dataset_entries = fin.readlines()

        return dataset_entries

    def process_dataset_entry(self, data_entry: str):
        """Processing the data entries.

        Converts all audio into wav format and outputs filepath, duration and
        transcription text.
        """
        if len(data_entry.split("\t")) != 2:
            raise RuntimeError(f"have more than one tab in line {data_entry}")

        utt_id, text = data_entry.split("\t")
        transcript_text = text.strip()

        # specify src_audio_path
        if self.use_opus_archive:
            src_audio_path = os.path.join(self.audio_path_prefix, *utt_id.split("_")[:2], utt_id + ".opus")
        else:
            src_audio_path = os.path.join(self.audio_path_prefix, *utt_id.split("_")[:2], utt_id + ".flac")

        # specify tgt_audio_path
        if self.resampled_audio_dir:
            tgt_audio_path = os.path.join(self.resampled_audio_dir, *utt_id.split("_")[:2], utt_id + ".wav")

            if not os.path.exists(os.path.dirname(tgt_audio_path)):
                os.makedirs(os.path.dirname(tgt_audio_path), exist_ok=True)
            if not os.path.exists(tgt_audio_path):
                tfm = Transformer()
                tfm.rate(samplerate=self.target_samplerate)
                tfm.channels(n_channels=self.target_nchannels)
                tfm.build(input_filepath=src_audio_path, output_filepath=tgt_audio_path)

        else:
            tgt_audio_path = src_audio_path

        data = {
            "audio_filepath": tgt_audio_path,
            "duration": float(librosa.get_duration(path=tgt_audio_path)),
            "text": transcript_text,
        }

        return [DataEntry(data=data)]
