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

from sdp.processors.base_processor import BaseProcessor, DataEntry
import json
import os
import torchaudio
import math
from copy import deepcopy
from sdp.utils.common import load_manifest, save_manifest

class SplitLongAudio(BaseProcessor):
    """This processor splits long audio files into smaller segments.

    It processes audio files that exceed a specified maximum length by splitting them into
    smaller segments at natural pauses in the audio to maintain speech coherence.

    Args:
        suggested_max_len (float): Target maximum length for audio segments in seconds. Defaults to 3600
        min_pause_len (float): Minimum length of a pause to consider for splitting in seconds. Defaults to 1.0
        min_len (float): Minimum length for any split segment in seconds. Defaults to 1.0

    Returns:
        The same data as in the input manifest, but with long audio files split into
        multiple segments with updated paths and durations.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.tts.split.SplitLongAudio
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_split.json
              suggested_max_len: 3600
    """
    def __init__(self,
                 suggested_max_len: float = 3600,
                 min_pause_len: float = 1.0,
                 min_len: float = 1.0,
                 **kwargs
        ):
        super().__init__(**kwargs)
        self.suggested_max_len = suggested_max_len
        self.min_pause_len = min_pause_len
        self.min_len = min_len

    def process(self):
        """Process the input manifest to split long audio files into smaller segments.

        This method:
        1. Reads the input manifest
        2. For each audio file longer than suggested_max_len:
            - Identifies suitable pause points for splitting
            - Creates new audio files for each segment
            - Updates metadata for each split segment
        3. Saves the results to the output manifest

        The output manifest includes:
        - Original entries for audio files shorter than suggested_max_len
        - Split entries with updated paths and durations
        - Meta-entries containing split information for later joining
        """
        manifest = load_manifest(self.input_manifest_file)

        results = []
        for metadata in manifest:
            if metadata['duration'] < self.suggested_max_len:
                metadata['split_filepaths'] = None
                results.append(metadata)
                continue
            splits = []
            split_start = 0
            prev_end = 0
            for segment in metadata['segments']:
                start = segment['start']
                end = segment['end']

                # Calculate duration of the current turn
                turn_duration = end - start

                # Calculate pause duration
                pause_duration = start - prev_end

                if end - split_start > self.suggested_max_len:
                    # Add the timestamp of the pause to the list
                    splits.append(prev_end)
                    split_start = prev_end

                # Update previous start and end times
                prev_end = end
            metadata['split_timestamps'] = splits
            # Now that we have all splits, generate splitted wav files
            audio, sr = torchaudio.load(metadata['resampled_audio_filepath'])
            path, filename = os.path.split(metadata['resampled_audio_filepath'])
            split_start = 0
            split_filepaths = []
            actual_splits = []
            split_durations = []
            for k, split in enumerate(splits):
                split_filepath = os.path.join(path, filename[:-4] + '.{}_of_{}.wav'.format(k+1, 1+len(splits)))
                split_end = math.ceil(split*sr)
                if split_end-split_start > self.min_len * sr:
                    torchaudio.save(split_filepath, audio[:, split_start:split_end], sr)
                    split_filepaths.append(split_filepath)
                    actual_splits.append(split_start/sr)
                    split_durations.append((split_end-split_start)/sr)
                    split_start = split_end

            # Write last split
            split_filepath = os.path.join(path, filename[:-4] + '.{}_of_{}.wav'.format(1+len(splits), 1+len(splits)))
            last_frame = len(audio[0])-1
            # skip audios that are too short
            if last_frame-split_start > self.min_len * sr and last_frame-split_start < (self.suggested_max_len + 1)*sr:
                torchaudio.save(split_filepath, audio[:, split_start:], sr)
                split_filepaths.append(split_filepath)
                split_durations.append((last_frame-split_start)/sr)
                actual_splits.append(split_start/sr)

            # Add split_filepaths to results without split_filepaths field and resampled_audio_filepath replaced
            # with the corresponding splits
            for idx, split in enumerate(split_filepaths):
                split_metadata = deepcopy(metadata)
                split_metadata['resampled_audio_filepath'] = split
                split_metadata['duration'] = split_durations[idx]
                results.append(split_metadata)
            # We keep an entry with 'split_filepaths' in it as a meta-entry
            # to be used when joining metadatas together
            assert len(split_filepaths) == len(actual_splits)
            metadata['split_filepaths'] = split_filepaths
            metadata['split_offsets'] = actual_splits
            results.append(metadata)

        save_manifest(results, self.output_manifest_file)


class JoinSplitAudioMetadata(BaseProcessor):
    """A processor for joining metadata of previously split audio files.

    This processor combines the metadata (transcripts and alignments) of audio files 
    that were previously split by the SplitLongAudio processor. It adjusts timestamps
    and concatenates transcripts to recreate the original audio's metadata.

    Args:
        None

    Returns:
        The same data as in the input manifest, but with split audio files joined together.
    """
    def __init__(self,
                 **kwargs
        ):
        super().__init__(**kwargs)

    def process(self):
        """Process the input manifest to join metadata of split audio files.

        This method:
        1. Reads the input manifest
        2. Identifies meta-entries containing split information
        3. For each meta-entry:
            - Concatenates transcripts from all splits
            - Adjusts alignment timestamps based on split offsets
            - Creates a single combined metadata entry
        4. Saves the results to the output manifest

        The output manifest contains:
        - Original entries for unsplit audio files
        - Combined entries for previously split audio files
        """
        manifest = load_manifest(self.input_manifest_file)

        fp_w = open(self.output_manifest_file, 'w')

        meta_entries = []
        for metadata in manifest:
            if 'split_filepaths' in metadata:
                meta_entries.append(metadata)

        for meta_entry in meta_entries:
            # Find all parts
            transcripts = []
            alignments = []
            if meta_entry['split_filepaths'] is None:
                del meta_entry['split_filepaths']
                fp_w.write(f"{json.dumps(meta_entry)}\n")
                continue
            for idx, split in enumerate(meta_entry['split_filepaths']):
                entry = next(filter(lambda x: x['resampled_audio_filepath'] == split, manifest))
                transcripts.append(entry['text'])
                alignment = entry['alignment']
                for word in alignment:
                    word['start'] += meta_entry['split_offsets'][idx]
                    word['end'] += meta_entry['split_offsets'][idx]
                alignments += alignment
            # Concatenate transcripts and alignment together
            meta_entry['text'] = ' '.join(transcripts)
            meta_entry['alignment'] = alignments

            # Remove 'split_filepaths' field from meta entry to turn it into a real entry
            del meta_entry['split_filepaths']
            fp_w.write(f"{json.dumps(meta_entry)}\n")
        
        fp_w.close()


