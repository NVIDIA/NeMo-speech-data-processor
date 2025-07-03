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

from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest, save_manifest

class MergeAlignmentDiarization(BaseProcessor):
    """This processor merges alignment and diarization information from a manifest file.

    It takes a manifest file containing both alignment and diarization information and
    merges the alignment information into the diarization segments.

    Args:
        None

    Returns:
        The same data as in the input manifest, but with alignment information merged into
        the diarization segments.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.tts.merge_alignment_diarization.MergeAlignmentDiarization
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_merged.json
    """
    def __init__(self,
            **kwargs):
        super().__init__(**kwargs)

    def process(self):
        manifest = load_manifest(self.input_manifest_file)

        # Manifest here needs to contain both paths to alignment files and 'segments'
        # from pyannote. We identify all the words that belong in each pyannote segment
        # and join them together. 
        
        for metadata in manifest:
            alignment = metadata['alignment']
            segments = metadata['segments']
            last_word_idx = 0

            if len(alignment) > 0 and len(segments) > 0:
                for i, segment in enumerate(segments):
                    words_in_segment = []

                    while last_word_idx < len(alignment):
                        word = alignment[last_word_idx]
                        word_start = word['start']
                        word_end = word['end']

                        if word_start >= segment['end']:
                            break

                        if word_start >= segment['start'] and word_end <= segment['end']:
                            words_in_segment.append(word)
                            last_word_idx += 1
                        # If the word overlaps with both current and next segment
                        else:
                            # Check overlap with the current segment
                            current_overlap = max(0, min(word_end, segment['end']) - max(word_start, segment['start']))

                            # Check overlap with the next segment, if it exists
                            if i < len(segments) - 1:
                                next_segment = segments[i + 1]
                                next_overlap = max(0, min(word_end, next_segment['end']) - max(word_start, next_segment['start']))
                            else:
                                next_overlap = 0

                            # Assign based on overlap comparison
                            if current_overlap >= next_overlap and current_overlap > 0:
                                words_in_segment.append(word)
                                last_word_idx += 1  # Move to the next word
                            elif next_overlap > current_overlap:
                                break  # Move to the next segment if the word fits better there
                            else:
                                # If no overlap with current or next segment, increment to avoid infinite loop
                                last_word_idx += 1
                        
                        # If we are at the last word, break
                        if last_word_idx == len(alignment):
                            break

                    segment['text'] = ' '.join([x['word'] for x in words_in_segment])
                    segment['words'] = words_in_segment

        save_manifest(manifest, self.output_manifest_file)

