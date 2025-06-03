# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import csv
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import librosa
import soundfile as sf

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, BaseProcessor, DataEntry
from sdp.utils.common import extract_archive


# Step 1: Create Initial Audio and Manifest (Full Audio)
class CreateInitialAudioAndManifest(BaseParallelProcessor):
    """
    Step 1: Create initial manifest with full audio files.
    
    Features:
    - Supports both earnings21 and earnings22
    - Creates manifest pointing to original audio files
    - No text processing (placeholder text)
    - Gets audio duration from files
    """

    def __init__(
        self,
        dataset_root: str,
        raw_audio_source_dir: str,
        output_manifest_file: str,
        dataset_type: str = "earnings21",  # "earnings21" or "earnings22"
        subset: str = "full",
        test_mode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_root = Path(dataset_root)
        self.raw_audio_source_dir = Path(raw_audio_source_dir)
        self.output_manifest_file = output_manifest_file
        self.dataset_type = dataset_type
        self.subset = subset
        self.test_mode = test_mode
        
        # Create converted audio directory
        self.converted_audio_dir = Path(self.output_manifest_file).parent / "converted_audio"
        self.converted_audio_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self):
        """Prepare the processor by loading file metadata."""
        # Extract archives if needed (earnings21 only)
        if self.dataset_type == "earnings21":
            for archive_name in ["transcripts.tar.gz", "audio.tar.gz"]:
                archive_path = self.dataset_root / archive_name
                if archive_path.exists():
                    extract_archive(str(archive_path), str(self.dataset_root))

        # Load file list based on dataset type and subset
        if self.dataset_type == "earnings21":
            if self.subset == "eval10":
                metadata_file = self.dataset_root / "eval10-file-metadata.csv"
            else:
                metadata_file = self.dataset_root / "earnings21-file-metadata.csv"
        else:  # earnings22
            metadata_file = self.dataset_root / "metadata.csv"
        
        # If metadata file doesn't exist, discover files from audio directory
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}. Discovering files from audio directory.")
            file_ids = []
            for ext in ['*.mp3', '*.wav']:
                file_ids.extend([f.stem for f in self.raw_audio_source_dir.glob(ext)])
            self.file_ids = file_ids
        else:
            file_metadata_df = pd.read_csv(metadata_file)
            # Handle different column names between earnings21 and earnings22
            if 'file_id' in file_metadata_df.columns:
                self.file_ids = file_metadata_df['file_id'].astype(str).tolist()
            elif 'File ID' in file_metadata_df.columns:
                self.file_ids = file_metadata_df['File ID'].astype(str).tolist()
            else:
                raise ValueError(f"Neither 'file_id' nor 'File ID' column found in {metadata_file}")
        
        if self.test_mode:
            self.file_ids = self.file_ids[:2]
            
        logger.info(f"Loaded {len(self.file_ids)} file IDs for {self.dataset_type} subset {self.subset}.")

    def _convert_audio_if_needed(self, audio_file: Path, file_id: str) -> Path:
        """
        Convert audio file to single-channel 16kHz WAV if needed.
        
        Args:
            audio_file: Path to the original audio file
            file_id: File ID for naming the converted file
            
        Returns:
            Path to the audio file to use (original or converted)
        """
        try:
            # Load audio to check properties
            audio_data, sample_rate = librosa.load(str(audio_file), sr=None, mono=False)
            
            # Check if conversion is needed
            needs_conversion = False
            conversion_reasons = []
            
            # Check if it's MP3
            if audio_file.suffix.lower() == '.mp3':
                needs_conversion = True
                conversion_reasons.append("MP3 format")
            
            # Check if it's multi-channel
            if audio_data.ndim > 1:
                needs_conversion = True
                conversion_reasons.append(f"{audio_data.shape[0]} channels")
            
            # Check if sample rate is not 16kHz
            if sample_rate != 16000:
                needs_conversion = True
                conversion_reasons.append(f"{sample_rate}Hz sample rate")
            
            if not needs_conversion:
                logger.debug(f"No conversion needed for {file_id}")
                return audio_file
            
            # Convert audio
            logger.info(f"Converting {file_id}: {', '.join(conversion_reasons)} -> single-channel 16kHz WAV")
            
            # Load as mono and resample to 16kHz
            audio_mono, _ = librosa.load(str(audio_file), sr=16000, mono=True)
            
            # Save as WAV
            converted_file = self.converted_audio_dir / f"{file_id}.wav"
            sf.write(str(converted_file), audio_mono, 16000)
            
            logger.debug(f"Converted audio saved to {converted_file}")
            return converted_file
            
        except Exception as e:
            logger.error(f"Error converting audio file {audio_file}: {e}")
            # Return original file if conversion fails
            return audio_file

    def read_manifest(self):
        """Read and process all files to create manifest entries."""
        return self.file_ids

    def process_dataset_entry(self, file_id: str) -> List[DataEntry]:
        """Process a single file to create full audio manifest entry."""
        file_id = str(file_id)
        
        # Find audio file
        audio_file = None
        for ext in ['.mp3', '.wav']:
            potential_path = self.raw_audio_source_dir / f"{file_id}{ext}"
            if potential_path.exists():
                audio_file = potential_path
                break
        
        if not audio_file:
            logger.warning(f"Audio file not found for {file_id}")
            return []

        try:
            # Convert audio if needed (handles MP3, multi-channel, non-16kHz)
            final_audio_file = self._convert_audio_if_needed(audio_file, file_id)
            
            # Get audio duration from the final audio file
            duration = librosa.get_duration(path=str(final_audio_file))
            
            # Create manifest entry
            entry_data = {
                "audio_filepath": str(final_audio_file),
                "duration": duration,
                "text": "",  # Placeholder text
                "file_id": file_id,
            }
            
            return [DataEntry(data=entry_data)]
            
        except Exception as e:
            logger.error(f"Error processing audio file {file_id}: {e}")
            return []


# Step 2: Populate Full Text for Manifest
class CreateFullAudioManifestEarnings21(BaseParallelProcessor):
    """
    Step 2: Add ground truth text from NLP files to the manifest.
    
    Features:
    - Supports both earnings21 and earnings22
    - Reconstructs full text from NLP tokens
    - Preserves punctuation and capitalization
    """

    def __init__(
        self,
        input_manifest_file: str,
        dataset_root: str,
        output_manifest_file: str,
        dataset_type: str = "earnings21",  # "earnings21" or "earnings22"
        preserve_punctuation: bool = True,
        preserve_capitalization: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_manifest_file = input_manifest_file
        self.dataset_root = Path(dataset_root)
        self.output_manifest_file = output_manifest_file
        self.dataset_type = dataset_type
        self.preserve_punctuation = preserve_punctuation
        self.preserve_capitalization = preserve_capitalization

    def _get_nlp_file_path(self, file_id: str) -> Path:
        """Get NLP file path based on dataset type."""
        if self.dataset_type == "earnings21":
            return self.dataset_root / "transcripts" / "nlp_references" / f"{file_id}.nlp"
        else:  # earnings22
            # Check both possible locations for earnings22
            nlp_path1 = self.dataset_root / "transcripts" / "nlp_references" / f"{file_id}.nlp"
            nlp_path2 = self.dataset_root / "subset10" / "nonverbatim_transcripts" / "nlp_references" / f"{file_id}.nlp"
            
            if nlp_path1.exists():
                return nlp_path1
            elif nlp_path2.exists():
                return nlp_path2
            else:
                return nlp_path1  # Return first path for error reporting

    def _load_nlp_file(self, file_id: str) -> List[Dict[str, Any]]:
        """Load NLP file containing tokens and metadata."""
        nlp_file = self._get_nlp_file_path(file_id)
        
        if not nlp_file.exists():
            logger.warning(f"NLP file not found: {nlp_file}")
            return []
        
        tokens_list = []
        try:
            with open(nlp_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                try:
                    header = next(reader)
                except StopIteration:
                    logger.warning(f"NLP file {nlp_file} is empty or has no header.")
                    return []

                for i, row_values in enumerate(reader):
                    if len(row_values) == len(header):
                        token_data = dict(zip(header, row_values))
                        
                        # Parse 'tags' and 'wer_tags' fields if they are string representations of lists
                        for key_to_parse in ['tags', 'wer_tags']:
                            if key_to_parse in token_data:
                                field_value = token_data[key_to_parse]
                                if isinstance(field_value, str):
                                    try:
                                        token_data[key_to_parse] = json.loads(field_value)
                                    except json.JSONDecodeError:
                                        if field_value and field_value != "[]":
                                            logger.debug(f"Field '{key_to_parse}' in {nlp_file} non-JSON: {field_value}")
                        tokens_list.append(token_data)
                    else:
                        logger.warning(f"Skipping malformed row in {nlp_file} (row {i+2})")
            return tokens_list
                
        except Exception as e:
            logger.error(f"Error processing NLP file {nlp_file}: {e}")
            return []

    def _reconstruct_text(self, tokens: List[Dict[str, Any]]) -> str:
        """Reconstruct text from tokens with proper spacing and punctuation."""
        if not tokens:
            return ""
        
        text_parts = []
        for token in tokens:
            token_text = token.get('token', '').strip()
            if not token_text:
                continue
            
            text_parts.append(token_text)
            # Add punctuation if preserving and it exists
            if self.preserve_punctuation and token.get('punctuation'):
                text_parts.append(token.get('punctuation'))

        # Join with spaces and clean up punctuation spacing
        text = " ".join(text_parts)
        if self.preserve_punctuation:
            # Remove spaces before common punctuation marks
            text = re.sub(r'\s+([,.!?;:])', r'\1', text)

        if not self.preserve_capitalization:
            text = text.lower()
        
        # Final cleanup of multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_dataset_entry(self, data_entry: Dict[str, Any]) -> List[DataEntry]:
        """Process a single manifest entry to add full text."""
        file_id = data_entry['file_id']
        tokens = self._load_nlp_file(file_id)
        
        if not tokens:
            logger.warning(f"No NLP tokens for {file_id}, text will be empty.")
            data_entry['text'] = data_entry.get('text', '')
        else:
            data_entry['text'] = self._reconstruct_text(tokens)
            
        return [DataEntry(data=data_entry)]


# Step 3: Create Speaker-level Segmented Manifest (renamed from CreateFinalSegmentedManifest)
class SpeakerSegmentedManifest(BaseParallelProcessor):
    """
    Step 6: Create speaker-segmented manifest without duration calculation.
    
    Features:
    - Supports both earnings21 and earnings22
    - Speaker-level segmentation based on NLP files
    - No duration calculation (set to None)
    - Optional speaker name mapping
    """

    def __init__(
        self,
        input_manifest_file: str,
        dataset_root: str,
        output_manifest_file: str,
        dataset_type: str = "earnings21",  # "earnings21" or "earnings22"
        preserve_punctuation: bool = True,
        preserve_capitalization: bool = True,
        include_speaker_info: bool = True,
        include_tags: bool = False,
        use_speaker_metadata_csv: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_manifest_file = input_manifest_file
        self.dataset_root = Path(dataset_root)
        self.output_manifest_file = output_manifest_file
        self.dataset_type = dataset_type
        self.preserve_punctuation = preserve_punctuation
        self.preserve_capitalization = preserve_capitalization
        self.include_speaker_info = include_speaker_info
        self.include_tags = include_tags
        self.use_speaker_metadata_csv = use_speaker_metadata_csv
        self.speaker_name_map = {}

    def prepare(self):
        """Prepare the processor by loading speaker metadata if needed."""
        # Load speaker metadata if requested (earnings21 only)
        if self.use_speaker_metadata_csv and self.dataset_type == "earnings21":
            self._load_speaker_metadata()

    def _load_speaker_metadata(self):
        """Load speaker metadata for earnings21."""
        metadata_file = self.dataset_root / "speaker-metadata.csv"
        if not metadata_file.exists():
            logger.warning(f"Speaker metadata file not found: {metadata_file}")
            return
        
        try:
            df = pd.read_csv(metadata_file)
            for _, row in df.iterrows():
                file_id_key = str(row['file_id'])
                speaker_id_key = str(row['speaker_id'])
                if file_id_key not in self.speaker_name_map:
                    self.speaker_name_map[file_id_key] = {}
                self.speaker_name_map[file_id_key][speaker_id_key] = row['speaker_name']
            logger.info(f"Loaded speaker metadata from {metadata_file}")
        except Exception as e:
            logger.error(f"Error loading speaker metadata {metadata_file}: {e}")

    def _get_nlp_file_path(self, file_id: str) -> Path:
        """Get NLP file path based on dataset type."""
        if self.dataset_type == "earnings21":
            return self.dataset_root / "transcripts" / "nlp_references" / f"{file_id}.nlp"
        else:  # earnings22
            # Check both possible locations for earnings22
            nlp_path1 = self.dataset_root / "transcripts" / "nlp_references" / f"{file_id}.nlp"
            nlp_path2 = self.dataset_root / "subset10" / "nonverbatim_transcripts" / "nlp_references" / f"{file_id}.nlp"
            
            if nlp_path1.exists():
                return nlp_path1
            elif nlp_path2.exists():
                return nlp_path2
            else:
                return nlp_path1  # Return first path for error reporting

    def _load_nlp_file(self, file_id: str) -> List[Dict[str, Any]]:
        """Load NLP file containing tokens and metadata."""
        nlp_file = self._get_nlp_file_path(file_id)
        
        if not nlp_file.exists():
            logger.warning(f"NLP file not found: {nlp_file}")
            return []
        
        tokens_list = []
        try:
            with open(nlp_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                try:
                    header = next(reader)
                except StopIteration:
                    logger.warning(f"NLP file {nlp_file} is empty or has no header.")
                    return []

                for i, row_values in enumerate(reader):
                    if len(row_values) == len(header):
                        token_data = dict(zip(header, row_values))
                        
                        # Parse 'tags' and 'wer_tags' fields if they are string representations of lists
                        for key_to_parse in ['tags', 'wer_tags']:
                            if key_to_parse in token_data:
                                field_value = token_data[key_to_parse]
                                if isinstance(field_value, str):
                                    try:
                                        token_data[key_to_parse] = json.loads(field_value)
                                    except json.JSONDecodeError:
                                        if field_value and field_value != "[]":
                                            logger.debug(f"Field '{key_to_parse}' in {nlp_file} non-JSON: {field_value}")
                        tokens_list.append(token_data)
                    else:
                        logger.warning(f"Skipping malformed row in {nlp_file} (row {i+2})")
            return tokens_list
                
        except Exception as e:
            logger.error(f"Error processing NLP file {nlp_file}: {e}")
            return []

    def _load_entity_tags(self, file_id: str) -> Dict[str, Dict[str, str]]:
        """Load entity tags file (earnings21 only)."""
        if self.dataset_type != "earnings21":
            return {}
            
        tags_file = self.dataset_root / "transcripts" / "tags" / f"{file_id}.tags.json"
        if not tags_file.exists():
            return {}
        
        try:
            with open(tags_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading tags file {tags_file}: {e}")
            return {}

    def _reconstruct_text(self, tokens: List[Dict[str, Any]]) -> str:
        """Reconstruct text from tokens with proper spacing and punctuation."""
        if not tokens:
            return ""
        
        text_parts = []
        for token in tokens:
            token_text = token.get('token', '').strip()
            if not token_text:
                continue
            
            text_parts.append(token_text)
            # Add punctuation if preserving and it exists
            if self.preserve_punctuation and token.get('punctuation'):
                text_parts.append(token.get('punctuation'))

        # Join with spaces and clean up punctuation spacing
        text = " ".join(text_parts)
        if self.preserve_punctuation:
            # Remove spaces before common punctuation marks
            text = re.sub(r'\s+([,.!?;:])', r'\1', text)

        if not self.preserve_capitalization:
            text = text.lower()
        
        # Final cleanup of multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _create_segments(self, tokens: List[Dict[str, Any]], file_id: str) -> List[Dict[str, Any]]:
        """Create segments based on speaker changes."""
        if not tokens:
            return []
        
        segments = []
        current_segment_tokens = []
        current_speaker_id = tokens[0].get('speaker', 'unknown_speaker_0') if tokens else 'unknown_speaker_0'

        for token in tokens:
            token_speaker_id = token.get('speaker', current_speaker_id)
            
            # Check for speaker change
            if token_speaker_id != current_speaker_id and current_segment_tokens:
                # Finalize current segment
                segment_text = self._reconstruct_text(current_segment_tokens)
                if segment_text.strip():
                    segments.append({
                        'tokens': current_segment_tokens,
                        'text': segment_text,
                        'speaker_id': current_speaker_id,
                        'file_id': file_id,
                    })
                
                # Start new segment
                current_segment_tokens = [token]
                current_speaker_id = token_speaker_id
            else:
                current_segment_tokens.append(token)
        
        # Handle last segment
        if current_segment_tokens:
            segment_text = self._reconstruct_text(current_segment_tokens)
            if segment_text.strip():
                segments.append({
                    'tokens': current_segment_tokens,
                    'text': segment_text,
                    'speaker_id': current_speaker_id,
                    'file_id': file_id,
                })
        
        return segments

    def process_dataset_entry(self, full_audio_manifest_entry: Dict[str, Any]) -> List[DataEntry]:
        """Process a single full audio manifest entry to create segmented entries."""
        file_id = full_audio_manifest_entry['file_id']
        audio_filepath = full_audio_manifest_entry['audio_filepath']

        logger.info(f"Processing file {file_id} for segmentation")

        # Load NLP tokens
        tokens = self._load_nlp_file(file_id)
        if not tokens:
            logger.warning(f"No NLP tokens for {file_id}, cannot create segments.")
            return []

        # Load entity tags if requested
        entity_tags = self._load_entity_tags(file_id) if self.include_tags else {}

        # Create segments
        segments = self._create_segments(tokens, file_id)
        logger.info(f"Created {len(segments)} segments for file {file_id}")

        # Create manifest entries
        output_entries = []
        for idx, segment_dict in enumerate(segments):
            segment_text = segment_dict['text']
            speaker_id = segment_dict['speaker_id']
            
            # Create manifest entry
            manifest_entry_data = {
                "audio_filepath": audio_filepath,  # Point to original audio file
                "duration": 0,  # Set to 0 instead of None to avoid TypeError in base processor
                "text": segment_text,
                "file_id": file_id,
                "segment_id": idx,
                "start_time": None,  # No timing information
                "end_time": None,    # No timing information
            }

            # Add speaker information
            if self.include_speaker_info:
                speaker_name = speaker_id  # Default to ID
                if (self.use_speaker_metadata_csv and 
                    file_id in self.speaker_name_map and 
                    speaker_id in self.speaker_name_map[file_id]):
                    speaker_name = self.speaker_name_map[file_id][speaker_id]
                manifest_entry_data["speaker"] = speaker_name
            
            # Add tags if requested
            if self.include_tags:
                segment_tags = []
                segment_entities = []
                
                # Extract basic tags from tokens
                for token in segment_dict.get('tokens', []):
                    if token.get('tags') and str(token['tags']).strip():
                        tag_val = str(token['tags']).strip()
                        tag_type = tag_val.split(':', 1)[1].strip() if ':' in tag_val else tag_val
                        if tag_type and tag_type not in segment_tags:
                            segment_tags.append(tag_type)
                
                manifest_entry_data["tags"] = segment_tags
                manifest_entry_data["entities"] = segment_entities

            output_entries.append(DataEntry(data=manifest_entry_data))
            
        logger.info(f"Successfully processed {len(output_entries)} segments for file {file_id}")
        return output_entries


# Step 5: Create Sentence-level Segmented Manifest based on CTM files
class CreateSentenceSegmentedManifest(BaseParallelProcessor):
    """
    Step 5: Create sentence-level segments based on CTM files.
    
    This processor reads CTM files generated by the NeMo Forced Aligner and creates
    sentence-level segments based on punctuation patterns. It segments on words ending
    with !, ?, or . (excluding numbers like 42.12) where the next segment starts with
    a capital letter.
    
    Features:
    - Reads word-level CTM files with timing information
    - Creates sentence-level segments based on punctuation
    - Preserves word-level alignments within each segment
    - Calculates accurate segment durations from CTM data
    """

    def __init__(
        self,
        input_manifest_file: str,
        ctm_dir: str,
        output_manifest_file: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_manifest_file = input_manifest_file
        self.ctm_dir = Path(ctm_dir)
        self.output_manifest_file = output_manifest_file

    def _parse_ctm_file(self, ctm_path: str) -> List[Dict[str, Any]]:
        """Parse CTM file to extract word alignments."""
        alignments = []
        try:
            with open(ctm_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # CTM format: utt_id channel start_time duration word
                        utt_id = parts[0]
                        channel = parts[1]
                        start_time = float(parts[2])
                        duration = float(parts[3])
                        word = parts[4]
                        end_time = start_time + duration
                        
                        alignments.append({
                            'word': word,
                            'start': round(start_time, 3),
                            'end': round(end_time, 3),
                            'utt_id': utt_id,
                            'channel': channel
                        })
        except Exception as e:
            logger.error(f"Error parsing CTM file {ctm_path}: {e}")
        
        return alignments

    def _is_sentence_end(self, word: str, next_word: str = None) -> bool:
        """
        Check if a word marks the end of a sentence.
        
        Rules:
        - Word ends with !, ?, or .
        - Exclude numbers like 42.12 (. within numbers)
        - Exclude common abbreviations like Ms., Mr., Dr., etc.
        - Next word should start with capital letter (if available)
        """
        if not word:
            return False
            
        # Check if word ends with sentence-ending punctuation
        if not word.endswith(('.', '!', '?')):
            return False
        
        # Handle exclamation and question marks - these are always sentence endings
        if word.endswith(('!', '?')):
            return True
            
        # For words ending with '.', do additional checks
        if word.endswith('.'):
            # Remove the final '.' and check if what remains is a number
            word_without_dot = word[:-1]
            try:
                # If it's a pure number, it's likely part of a decimal
                float(word_without_dot)
                return False
            except ValueError:
                # Not a number, continue with other checks
                pass
            
            # Check for common abbreviations (case-insensitive)
            common_abbreviations = {
                'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'inc', 'corp', 'ltd', 'co',
                'st', 'ave', 'blvd', 'rd', 'ln', 'ct', 'pl', 'sq', 'ft', 'in', 'cm', 'mm', 'kg', 'lb',
                'oz', 'pt', 'qt', 'gal', 'mph', 'rpm', 'vol', 'no', 'pg', 'pp', 'ch', 'sec', 'min',
                'hr', 'hrs', 'am', 'pm', 'est', 'pst', 'cst', 'mst', 'utc', 'gmt', 'jan', 'feb', 'mar',
                'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'mon', 'tue', 'wed',
                'thu', 'fri', 'sat', 'sun', 'dept', 'div', 'mgr', 'dir', 'pres', 'vp', 'ceo', 'cfo',
                'cto', 'coo', 'evp', 'svp', 'avp'
            }
            
            if word_without_dot.lower() in common_abbreviations:
                return False
        
        # If we have a next word, check if it starts with capital letter
        if next_word:
            return next_word[0].isupper()
        
        # If no next word, assume it's sentence end
        return True

    def _create_sentence_segments(self, alignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create sentence-level segments from word alignments."""
        if not alignments:
            return []
        
        segments = []
        current_segment_words = []
        
        for i, alignment in enumerate(alignments):
            current_segment_words.append(alignment)
            
            # Check if this word ends a sentence
            next_word = alignments[i + 1]['word'] if i + 1 < len(alignments) else None
            if self._is_sentence_end(alignment['word'], next_word):
                # Create segment from current words
                if current_segment_words:
                    segment_text = ' '.join([w['word'] for w in current_segment_words])
                    segment_start = current_segment_words[0]['start']
                    segment_end = current_segment_words[-1]['end']
                    segment_duration = round(segment_end - segment_start, 3)
                    
                    segments.append({
                        'text': segment_text,
                        'start_time': segment_start,
                        'end_time': segment_end,
                        'duration': segment_duration,
                        'alignment': current_segment_words.copy()
                    })
                    
                    current_segment_words = []
        
        # Handle any remaining words
        if current_segment_words:
            segment_text = ' '.join([w['word'] for w in current_segment_words])
            segment_start = current_segment_words[0]['start']
            segment_end = current_segment_words[-1]['end']
            segment_duration = round(segment_end - segment_start, 3)
            
            segments.append({
                'text': segment_text,
                'start_time': segment_start,
                'end_time': segment_end,
                'duration': segment_duration,
                'alignment': current_segment_words.copy()
            })
        
        return segments

    def process_dataset_entry(self, aligned_manifest_entry: Dict[str, Any]) -> List[DataEntry]:
        """Process a single aligned manifest entry to create sentence-level segments."""
        file_id = aligned_manifest_entry['file_id']
        audio_filepath = aligned_manifest_entry['audio_filepath']
        
        # Find corresponding CTM file
        ctm_file = self.ctm_dir / f"{file_id}.ctm"
        if not ctm_file.exists():
            logger.warning(f"CTM file not found: {ctm_file}")
            return []
        
        # Parse CTM file
        alignments = self._parse_ctm_file(str(ctm_file))
        if not alignments:
            logger.warning(f"No alignments found in CTM file: {ctm_file}")
            return []
        
        # Create sentence segments
        segments = self._create_sentence_segments(alignments)
        logger.info(f"Created {len(segments)} sentence segments for file {file_id}")
        
        # Create manifest entries
        output_entries = []
        for idx, segment in enumerate(segments):
            manifest_entry_data = {
                "audio_filepath": audio_filepath,
                "duration": segment['duration'],
                "text": segment['text'],
                "file_id": file_id,
                "segment_id": idx,
                "offset": segment['start_time'],  # Use offset instead of start_time
                "end_time": segment['end_time'],
                "alignment": segment['alignment']
            }
            
            output_entries.append(DataEntry(data=manifest_entry_data))
        
        logger.info(f"Successfully processed {len(output_entries)} sentence segments for file {file_id}")
        return output_entries


class NeMoForcedAligner(BaseProcessor):
    """
    Step 4: Apply NeMo Forced Aligner to get word-level timestamps.
    
    This processor wraps the NeMo Forced Aligner (NFA) script to generate
    word-level alignments for the earnings21 segments. It uses the ground
    truth text from the earnings21 dataset and aligns it with the audio
    to produce precise timing information.
    
    Features:
    - Uses NeMo's dedicated forced alignment script
    - Preserves ground truth text from earnings21
    - Generates word-level timestamps
    - Outputs CTM files with alignment information
    """

    def __init__(
        self,
        input_manifest_file: str,
        output_manifest_file: str,
        output_dir: str,
        pretrained_name: str = "/disk7/projects/models/small-parakeet/oci-N-1_G-8_config-parakeet-wav2vec-600m-am-fl-mc-mm-yt-yo_En-d0.5-rnnt_ctc-quality_LR-1e-4_wup-0_ts-2500.nemo",
        device: str = "cuda",
        nemo_path: str = None,
        **kwargs,
    ):
        super().__init__(output_manifest_file=output_manifest_file, input_manifest_file=input_manifest_file, **kwargs)
        self.output_dir = Path(output_dir)
        self.pretrained_name = pretrained_name
        self.device = device
        self.nemo_path = nemo_path
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self):
        """Process the manifest using NeMo Forced Aligner script."""
        import subprocess
        import json
        
        try:
            # Find NeMo forced aligner script
            if self.nemo_path:
                align_script = Path(self.nemo_path) / "tools" / "nemo_forced_aligner" / "align.py"
            else:
                # Try to find NeMo installation
                try:
                    import nemo
                    nemo_dir = Path(nemo.__file__).parent.parent
                    align_script = nemo_dir / "tools" / "nemo_forced_aligner" / "align.py"
                except ImportError:
                    raise ImportError("NeMo not found. Please install NeMo or specify nemo_path.")
            
            if not align_script.exists():
                raise FileNotFoundError(f"NeMo Forced Aligner script not found at {align_script}")
            
            logger.info(f"Using NeMo Forced Aligner script at: {align_script}")
            
            # Prepare manifest for forced alignment
            input_manifest = []
            with open(self.input_manifest_file, 'r') as f:
                for line in f:
                    if line.strip():
                        input_manifest.append(json.loads(line))
            
            # Create temporary manifest with absolute paths
            temp_manifest_path = self.output_dir / "temp_manifest_for_alignment.json"
            with open(temp_manifest_path, 'w') as f:
                for entry in input_manifest:
                    if entry.get('text', '').strip():  # Only process entries with text
                        # Ensure absolute path
                        audio_path = Path(entry['audio_filepath'])
                        if not audio_path.is_absolute():
                            audio_path = audio_path.resolve()
                        
                        alignment_entry = {
                            "audio_filepath": str(audio_path),
                            "text": entry['text'].strip()
                        }
                        f.write(json.dumps(alignment_entry) + '\n')
            
            # Run NeMo Forced Aligner
            # Determine if we should use pretrained_name or model_path
            if self.pretrained_name.endswith('.nemo'):
                # Local model file path - use model_path
                model_param = f"model_path={self.pretrained_name}"
            else:
                # Pretrained model name - use pretrained_name
                model_param = f"pretrained_name={self.pretrained_name}"
            
            cmd = [
                "python", str(align_script),
                model_param,
                f"manifest_filepath={temp_manifest_path}",
                f"output_dir={self.output_dir}",
                f"transcribe_device={self.device}",
                f"viterbi_device={self.device}",
                "batch_size=1",
                'save_output_file_formats=["ctm"]'
            ]
            
            logger.info(f"Running NeMo Forced Aligner: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("NeMo Forced Aligner completed successfully")
            
            # Process the output and merge with original manifest
            output_manifest_path = self.output_dir / f"{temp_manifest_path.stem}_with_output_file_paths.json"
            
            if output_manifest_path.exists():
                # Load alignment results
                alignment_results = []
                with open(output_manifest_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            alignment_results.append(json.loads(line))
                
                # Create mapping from audio filepath to alignment results
                alignment_map = {}
                for result in alignment_results:
                    audio_path = result['audio_filepath']
                    alignment_map[audio_path] = result
                
                # Merge alignments with original manifest
                output_entries = []
                for entry in input_manifest:
                    output_entry = entry.copy()
                    
                    if entry.get('text', '').strip():
                        # Find corresponding alignment
                        audio_path = str(Path(entry['audio_filepath']).resolve())
                        if audio_path in alignment_map:
                            alignment_result = alignment_map[audio_path]
                            
                            # Load word-level CTM file if available
                            if 'word_level_ctm_filepath' in alignment_result:
                                ctm_path = alignment_result['word_level_ctm_filepath']
                                word_alignments = self._parse_ctm_file(ctm_path)
                                output_entry['alignment'] = word_alignments
                                
                                # Calculate duration from alignments
                                if word_alignments:
                                    output_entry['duration'] = round(
                                        word_alignments[-1]['end'] - word_alignments[0]['start'], 3
                                    )
                                else:
                                    output_entry['duration'] = 0.0
                            else:
                                output_entry['alignment'] = []
                                output_entry['duration'] = 0.0
                        else:
                            output_entry['alignment'] = []
                            output_entry['duration'] = 0.0
                    else:
                        output_entry['alignment'] = []
                        output_entry['duration'] = 0.0
                    
                    output_entries.append(output_entry)
                
                # Save final manifest
                with open(self.output_manifest_file, 'w') as f:
                    for entry in output_entries:
                        f.write(json.dumps(entry) + '\n')
                
                logger.info(f"Saved aligned manifest to {self.output_manifest_file}")
                
                # Clean up temporary files
                temp_manifest_path.unlink(missing_ok=True)
                
            else:
                logger.error(f"Expected output file not found: {output_manifest_path}")
                raise FileNotFoundError(f"NeMo Forced Aligner did not produce expected output")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"NeMo Forced Aligner failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error in forced alignment: {e}")
            raise

    def _parse_ctm_file(self, ctm_path: str) -> List[Dict[str, Any]]:
        """Parse CTM file to extract word alignments."""
        alignments = []
        try:
            with open(ctm_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # CTM format: utt_id channel start_time duration word
                        start_time = float(parts[2])
                        duration = float(parts[3])
                        word = parts[4]
                        end_time = start_time + duration
                        
                        alignments.append({
                            'word': word,
                            'start': round(start_time, 3),
                            'end': round(end_time, 3)
                        })
        except Exception as e:
            logger.error(f"Error parsing CTM file {ctm_path}: {e}")
        
        return alignments