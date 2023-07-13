# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#

import glob
import os
import urllib.request
from pathlib import Path

import pandas as pd
from sox import Transformer

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive


def get_coraal_url_list():
    """Returns url list for CORAAL dataset.

    There are a few mistakes in the official url list that are fixed here.
    Can be overridden by tests to select a subset of urls.
    """
    dataset_url = "http://lingtools.uoregon.edu/coraal/coraal_download_list.txt"
    urls = []
    for file_url in urllib.request.urlopen(dataset_url):
        file_url = file_url.decode('utf-8').strip()
        # fixing known errors in the urls
        if file_url == 'http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2018.10.06.txt':
            file_url = 'http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2021.07.txt'
        if file_url == 'http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_metadata_2018.10.06.txt':
            file_url = 'http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_metadata_2021.07.txt'
        urls.append(file_url)
    return urls


class CreateInitialManifestCORAAL(BaseParallelProcessor):
    """Processor to create initial manifest for the Corpus of Regional African American Language (CORAAL) dataset.

    Dataset link: https://oraal.uoregon.edu/coraal/

    Will download all files, extract tars and split wav files based on the
    provided durations in the transcripts.

    Args:
        raw_data_dir (str): where to put raw downloaded data.
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
        target_samplerate (int): sample rate to resample to. Defaults to 16000.
        target_nchannels (int): target number of channels. Defaults to 1.
        drop_pauses (bool): if True, will drop all transcriptions that contain
            only silence (indicated by ``(pause X)`` in the transcript).
            Defaults to True.
        group_duration_threshold (float): can be used to group consecutive
            utterances from the same speaker to a longer duration. Set to 0
            to disable this grouping (but note that many utterances are
            transcribed with only a few seconds, so grouping is generally
            advised). Defaults to 20.

    Returns:
        This processor generates an initial manifest file with the following fields::

            {
                "audio_filepath": <path to the audio file>,
                "duration": <duration of the audio in seconds>,
                "text": <transcription>,
                "original_file": <name of the original file in the dataset this audio came from>,
                "speaker": <speaker id>,
                "is_interviewee": <whether this is an interviewee (accented speech)>,
                "gender": <speaker gender>,
                "age": <speaker age>,
                "education": <speaker education>,
                "occupation": <speaker occupation>,
            }
    """

    def __init__(
        self,
        raw_data_dir: str,
        resampled_audio_dir: str,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        drop_pauses: bool = True,
        group_duration_threshold: float = 20.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.resampled_audio_dir = resampled_audio_dir
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels
        self.drop_pauses = drop_pauses
        self.group_duration_threshold = group_duration_threshold

    def prepare(self):
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.resampled_audio_dir, exist_ok=True)

        # downloading all files
        for file_url in get_coraal_url_list():
            download_file(file_url, str(self.raw_data_dir))

        os.makedirs(self.raw_data_dir / "audio", exist_ok=True)
        os.makedirs(self.raw_data_dir / "transcripts", exist_ok=True)
        # extracting all files
        for data_file in glob.glob(f'{self.raw_data_dir}/*_audio_*.tar.gz'):
            # need to set force_extract=True, since there is no folder inside, just a list of files
            # and we extract data from multiple tars. Ideally, should change the way we check
            # for extracted data (currently there is an assumption that all data in archive is in a single folder)
            extract_archive(data_file, self.raw_data_dir / "audio", force_extract=True)
        for data_file in glob.glob(f'{self.raw_data_dir}/*_textfiles_*.tar.gz'):
            extract_archive(data_file, self.raw_data_dir / "transcripts", force_extract=True)

    def read_manifest(self):
        dfs = []
        for data_file in glob.glob(f'{self.raw_data_dir}/transcripts/*.txt'):
            df = pd.read_csv(data_file, delimiter='\t')
            df['Basefile'] = os.path.basename(data_file)[:-4]  # dropping .wav in the end
            dfs.append(df)
        df = pd.concat(dfs)

        if self.drop_pauses:
            df = df[~df['Content'].str.contains(r'\(pause \d+(?:\.\d+)?\)')]

        # grouping consecutive segments from the same speaker
        if self.group_duration_threshold > 0:
            df['Duration'] = df['EnTime'] - df['StTime']
            # puts each sequence of same speaker utts in a "bin"
            speaker_bins = (~df['Spkr'].eq(df['Spkr'].shift())).cumsum()
            # within each bin, computes cumulative duration and then int-divides by the threshold
            df['ThresholdMult'] = df.groupby(speaker_bins)['Duration'].transform(
                lambda x: pd.Series.cumsum(x) // self.group_duration_threshold
            )
            # finally, we take all positions where the int-division changes,
            # which indicates that cumsum exceded the threshold. And combine those
            # with speaker-change positions to get the final groups for utterance merging
            final_bins = (
                (~df['Spkr'].eq(df['Spkr'].shift())) | (~df['ThresholdMult'].eq(df['ThresholdMult'].shift()))
            ).cumsum()
            df = df.groupby(final_bins).agg(
                {
                    'StTime': 'min',
                    'EnTime': 'max',
                    'Content': ' '.join,
                    # will be the same in the group
                    'Spkr': lambda x: x.iloc[0],
                    'Basefile': lambda x: x.iloc[0],
                }
            )
        # assigning label for interviewee vs interviewer (can be used to select a subset later)
        df['is_interviewee'] = df.apply(lambda x: x['Spkr'] in x['Basefile'], axis=1)

        # matching with metadata (age, gender, etc.)
        metadata_dfs = []
        for data_file in glob.glob(f'{self.raw_data_dir}/*_metadata_*.txt'):
            metadata_dfs.append(pd.read_csv(data_file, delimiter='\t'))
        metadata_df = pd.concat(metadata_dfs)
        # only selecting a subset of columns - can be changed if more are needed
        # dropping duplicates since there are multiple rows per speaker because of
        # bit-rate, tar name and other file-specific information
        metadata_df = metadata_df[['CORAAL.Spkr', 'Gender', 'Age', 'Education', 'Occupation']].drop_duplicates()
        df = df.merge(metadata_df, left_on='Spkr', right_on='CORAAL.Spkr', how='left')
        df = df.drop('CORAAL.Spkr', axis=1)

        # would be better to keep it as df, but .values is way faster than .iterrows
        return df.values

    def process_dataset_entry(self, data_entry):
        (
            start_time,
            end_time,
            content,
            speaker,
            basefile,
            is_interviewee,
            gender,
            age,
            education,
            occupation,
        ) = data_entry

        src_file = str(self.raw_data_dir / 'audio' / (basefile + '.wav'))
        output_wav_path = os.path.join(
            self.resampled_audio_dir,
            f"{basefile}_{int(start_time * 1000)}_{int(end_time * 1000)}.wav",
        )

        if not os.path.exists(output_wav_path):
            tfm = Transformer()
            tfm.trim(start_time, end_time)
            tfm.rate(samplerate=self.target_samplerate)
            tfm.channels(n_channels=self.target_nchannels)
            tfm.build(input_filepath=src_file, output_filepath=output_wav_path)

        data = {
            "audio_filepath": output_wav_path,
            "duration": end_time - start_time,
            "text": content.strip(),
            "original_file": basefile,
            "speaker": speaker,
            "is_interviewee": is_interviewee,
            "gender": gender,
            "age": age,
            "education": education,
            "occupation": occupation,
        }

        return [DataEntry(data=data)]
