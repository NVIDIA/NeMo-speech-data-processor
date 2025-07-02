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


from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import load_manifest

class PrepareTTSSegmentsProcessor(BaseParallelProcessor):
    """This processor merges adjacent segments from the same speaker and splits segments to have a complete utterance.

    It processes segments by merging those from the same speaker that are adjacent, then
    splits segments based on duration limits, punctuation marks, and audio quality metrics
    like bandwidth.

    Args:
        min_duration (float): Minimum duration in seconds for a segment. Defaults to 5
        max_duration (float): Maximum duration in seconds for a segment. Defaults to 20
        max_pause (float): Maximum pause duration in seconds between merged segments. Defaults to 2
        terminal_punct_marks (str): String containing punctuation marks to split on. Defaults to ".!?。？？！。"
        punctuation_split_only (bool): Whether to only split on punctuation. Defaults to False

    Returns:
        The same data as in the input manifest, but with segments merged and split according
        to the specified parameters.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.tts.prepare_tts_segments.PrepareTTSSegmentsProcessor
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_processed.json
              min_duration: 5
              max_duration: 20
    """
    def __init__(self, 
                min_duration: float = 5, 
                max_duration: float = 20, 
                max_pause: float = 2, 
                terminal_punct_marks: str = ".!?。？？！。",
                punctuation_split_only: bool = False,
                **kwargs):
        super().__init__(**kwargs)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_pause = max_pause
        self.terminal_punct_marks = terminal_punct_marks
        self.punctuation_split_only = punctuation_split_only
    
    def read_manifest(self):
        ''' Reads metadata from JSONL file in the input manifest
        and converts it to data entries '''

        dataset_entries = load_manifest(self.input_manifest_file, encoding="utf8")

        return dataset_entries

    def get_words_list_from_all_segments(self, segments):
        """
        This method gets the words list from all the speaker segments
        """
        words = []
        for segment in segments:
            if ("text" in segment and segment["text"].strip() == "") or (segment["speaker"]=="no-speaker") or (not "text" in segment):
                continue

            if 'words' in segment:
                for word in segment['words']:
                    word['speaker'] = segment['speaker']
                    if 'metrics' in segment:
                        word['stoi_squim'] = segment['metrics'].get('stoi_squim', None)
                        word['sisdr_squim'] = segment['metrics'].get('sisdr_squim', None)
                        word['pesq_squim'] = segment['metrics'].get('pesq_squim', None)
                        word['bandwidth'] = segment['metrics'].get('bandwidth', None)
                    else:
                        word['stoi_squim'] = None
                        word['sisdr_squim'] = None
                        word['pesq_squim'] = None
                        word['bandwidth'] = None
                    words.append(word)
            else:
                logger.info('Found no words in segment')

        return words
    
    def is_valid_segment(self, segment):
        """
        This method checks if the segment is valid
        """
        if len(segment["words"]) ==1 and segment["words"][0]["end"] - segment["words"][0]["start"] > self.max_duration:
            return False
        sentence = " ".join([word["word"] for word in segment["words"]])
        if sentence:
            return True
        return False
 
    
    def split_segment_by_duration(self, segment):
        """
        This method splits the segment by duration, pauses, and bandwidth changes
        """
        words = segment["words"]
        current_segment = {
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "words": [],
        }
        segments = []
        
        for word in words:
            if not current_segment["words"]:
                current_segment = {
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
                continue
            
            # break the current segment if the duration is greater than the max duration and start a new segment
            if (word["end"] - current_segment["start"]) > self.max_duration:
                if self.is_valid_segment(current_segment):
                    segments.append(current_segment)
                current_segment = {
                    "speaker": segment["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
                continue
            
            # break the current segment if the pause is greater than the max pause and start a new segment
            if (word["start"] - current_segment["end"] > self.max_pause)  and (current_segment["end"] - current_segment["start"] >= self.min_duration):
                if self.is_valid_segment(current_segment):
                    segments.append(current_segment)
                current_segment = {
                    "speaker": segment["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
                continue
            
            # break the current segment if the bandwidth is different and start a new segment
            if (current_segment['words'] and word['bandwidth']!=current_segment['words'][-1]['bandwidth'] ) and (current_segment["end"] - current_segment["start"] >= self.min_duration):
                if self.is_valid_segment(current_segment):
                    segments.append(current_segment)
                current_segment = {
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
                continue

            current_segment["words"].append(word)
            current_segment["end"] = word["end"]
        
        # add the last segment if it is valid
        if current_segment["words"]:
            if self.is_valid_segment(current_segment):
                segments.append(current_segment)
        
        return segments
    
    def split_segment_by_punctuation(self, segment):
        """
        This method splits the given single speaker segment by punctuation marks, if no punctuation marks are found then it splits the segment by duration.
        If the punctuation_split_only flag is set to True, then it only splits the segment by punctuation marks.
        It calculates the split points based on the punctuation marks and then merges the adjacent split points when the duration of the current split point is less than min_duration.
        It then splits the segment at the new split points.
        """
        words = segment["words"]
        # get the punctuation split points
        split_points = [
            i
            for i, word in enumerate(words)
            if word["word"] and word["word"][-1] in self.terminal_punct_marks
        ]
        segments = []
        # if no punctuation marks, split the segment by duration
        if not split_points:
            if self.punctuation_split_only:
                return segments
            else:
                segments = self.split_segment_by_duration(segment)
                return segments

        # merge the split points with adjacent split points if the duration of the current split point is less than min_duration
        current_end = 0
        current_start = 0
        new_split_points = []
        while current_end < len(split_points):
            current_duration = words[split_points[current_end]]["end"] - words[split_points[current_start]]["start"]
            if current_duration < self.min_duration:
                # merge with the next split points until the maximum duration is reached
                next_end = current_end + 1
                while next_end < len(split_points) and words[split_points[next_end]]["end"] - words[split_points[current_start]]["start"] <= self.max_duration:
                    next_end += 1

                if next_end > current_end + 1:
                    new_split_points.append(split_points[next_end - 1])
                    current_start = next_end
                    current_end = next_end
                else:
                    new_split_points.append(split_points[current_end])
                    current_end += 1
            else:
                new_split_points.append(split_points[current_end])
                current_start = current_end + 1
                current_end = current_end + 1
        
        # now split the segment at the new split points
        # if the duration of the segment is greater than the max duration, split the segment by duration
        start = 0
        for end in new_split_points:
            duration = words[end]["end"] - words[start]["start"]
            sub_segment = {
                    "speaker": segment["speaker"],
                    "start": words[start]["start"],
                    "end": words[end]["end"],
                    "words": words[start : end + 1],
            }
            if duration <= self.max_duration:
                if self.is_valid_segment(sub_segment):
                    segments.append(sub_segment)
            else:
                segments.extend(self.split_segment_by_duration(sub_segment))
            start = end + 1
        
        # remaining clause in a new segment
        if start < len(words):
            remaining_segment = {
                "speaker": segment["speaker"],
                "start": words[start]["start"],
                "end": words[-1]["end"],
                "words": words[start:],
            }
            segments.extend(self.split_segment_by_duration(remaining_segment))

        return segments

    def add_new_segments_to_metadata(self, metadata, new_segments):

        segments = []

        for new_segment in new_segments:
            seg = {
                "speaker": new_segment["speaker"],
                "start": new_segment["start"],
                "end": new_segment["end"],
                "text": " ".join(word["word"] for word in new_segment["words"]),
                "words": [{"word": word["word"], "start": word["start"], "end": word["end"]} for word in new_segment["words"]],
                "pesq_squim": [word["pesq_squim"] for word in new_segment["words"]],
                "stoi_squim": [word["stoi_squim"] for word in new_segment["words"]],
                "sisdr_squim": [word["sisdr_squim"] for word in new_segment["words"]],
                "bandwidth": [word["bandwidth"] for word in new_segment["words"]],
            }
            segments.append(seg)

        metadata['segments'] = segments

    def process_dataset_entry(self, metadata: DataEntry):
        """
        This function processes a dataset entry and splits it into segments based on the duration, punctuation, and bandwidth.
        It then adds the new segments to the metadata.
        """

        if 'segments' in metadata:
            words = self.get_words_list_from_all_segments(metadata['segments'])
            new_segments = []

            # split the segments by speakers first
            speaker_segments = []
            current_segment = {
                "speaker": None,
                "start": None,
                "end": None,
                "words": [],
            }

            for word in words:
                if current_segment["speaker"] is None:
                    current_segment = {
                        "speaker": word["speaker"],
                        "start": word["start"],
                        "end": word["end"],
                        "words": [word],
                    }
                elif word["speaker"] != current_segment["speaker"]:
                    speaker_segments.append(current_segment)
                    current_segment = {
                        "speaker": word["speaker"],
                        "start": word["start"],
                        "end": word["end"],
                        "words": [word],
                    }
                else:
                    current_segment["words"].append(word)
                    current_segment["end"] = word["end"]
            
            if current_segment["words"]:
                speaker_segments.append(current_segment)
            

            # split the segments at the punctuation marks, pauses, and bandwidth changes
            for speaker_segment in speaker_segments:
                if speaker_segment['speaker'] == 'no-speaker' or speaker_segment['speaker'] == None:
                    continue
                new_segments.extend(self.split_segment_by_punctuation(speaker_segment))

            # add the new segments to the metadata
            self.add_new_segments_to_metadata(metadata, new_segments)
            
        else:
            logger.info('Found no segments in metadata for audio file: ', metadata['audio_filepath'])
        
        return [DataEntry(data=metadata)]


