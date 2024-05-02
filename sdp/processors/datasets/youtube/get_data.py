from typing import List
import wget
import os
import json
import shutil
import urllib
from glob import glob
from pydub import AudioSegment
from pathlib import Path
import subprocess
from tqdm import tqdm
import boto3

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor, DataEntry
from sdp.utils.common import ffmpeg_convert
from sdp.processors.datasets.youtube.utils import read_ctm, Sentence, Sample


class DownloadData(BaseProcessor):
    def __init__(
        self,
        url_list: list, 
        output_dir: str, 
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.urls = url_list
        self.output_dir = output_dir

    def process(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)

        with open(self.output_manifest_file, 'w') as manifest:
            for url in self.urls:
                try:
                    subset_path = wget.download(url, out=self.output_dir)
                except (urllib.error.HTTPError, urllib.error.ContentTooShortError):
                    subset_path = None
                sample = {"subset_path" : subset_path}
                manifest_line = json.dumps(sample)
                manifest.writelines(f'{manifest_line}\n')


class ExtractData(BaseParallelProcessor):
    def __init__(
        self,
        remove_archive: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.remove_archive = remove_archive
    
    def process_dataset_entry(self, data_entry):
        archieve_filepath = data_entry['subset_path']
        output_dir = os.path.splitext(archieve_filepath)[0]
        os.makedirs(output_dir, exist_ok=True)
        shutil.unpack_archive(archieve_filepath, output_dir)

        if self.remove_archive:
            os.remove(archieve_filepath)
        
        data_entry["subset_path"] = os.path.join(output_dir, 'ssl')
        return [DataEntry(data=data_entry)]


class GetSourceAudioFilepaths(BaseParallelProcessor):
    def __init__(
        self,
        extension: str = "opus",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.extension = extension
    
    def process_dataset_entry(self, data_entry):
        opus_filepaths = glob(f"{data_entry['subset_path']}/*.{self.extension}")
        samples = [{"source_audio_path" : os.path.abspath(opus_filepath)} for opus_filepath in opus_filepaths]
        data_entries = [DataEntry(data = sample) for sample in samples]
        return data_entries


class ConvertToWav(BaseParallelProcessor):
    def __init__(
        self,
        output_audio_dir: str,
        audio_file_extenstion: str = "opus",
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_audio_dir = output_audio_dir
        self.audio_file_extenstion = audio_file_extenstion
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels
    
    def prepare(self):
        os.makedirs(self.output_audio_dir, exist_ok=True)
    
    def process_dataset_entry(self, data_entry):
        output_audio_filepath = os.path.abspath(
            os.path.join(self.output_audio_dir, 
                         os.path.basename(data_entry['source_audio_path']).replace(f".{self.audio_file_extenstion}", ".wav")))
        
        if os.path.exists(output_audio_filepath):
            logger.warning(f"{output_audio_filepath} is already exists. Skipping.")
            return []
        
        ffmpeg_convert(input_file=data_entry['source_audio_path'], 
                       output_wav=output_audio_filepath, 
                       sample_rate=self.target_samplerate, 
                       num_channels=self.target_nchannels)

        if not os.path.exists(output_audio_filepath):
            logger.warning(f"Conversion error: {data_entry['source_audio_path']}. File {output_audio_filepath} has not been created.")
            return []

        data_entry['audio_filepath'] = output_audio_filepath

        return [DataEntry(data=data_entry)]


class GetAudioDuration(BaseParallelProcessor):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
    
    def process_dataset_entry(self, data_entry):
        audio_file = AudioSegment.from_file(data_entry['audio_filepath'])
        data_entry['duration'] = round(audio_file.duration_seconds, 2)
        return [DataEntry(data=data_entry)]


class RunNFA(BaseProcessor):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
    
    def process(self):
        pass


class GetSampleID(BaseParallelProcessor):
    def __init__(
        self,
        audio_filepath_field: str = "source_audio_path",
        depth: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_filepath_field = audio_filepath_field
        self.depth = depth
    
    def process_dataset_entry(self, data_entry):
        audio_filepath = Path(data_entry[self.audio_filepath_field])
        data_entry['sample_id'] = os.path.splitext('_'.join(list(audio_filepath.parts[-self.depth : ])))[0]
        return [DataEntry(data = data_entry)]


class GetSentencesFromNFA(BaseParallelProcessor):
    def __init__(
        self,
        end_of_sentence_punctuation_marks: list[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.end_of_sentence_punctuation_marks = end_of_sentence_punctuation_marks
    
    def prepare(self):
        self.output_dir = self.output_manifest_file.replace(".json", "")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_dataset_entry(self, data_entry):
        if 'words_level_ctm_filepath' not in data_entry:
            logger.warning(f"{data_entry['audio_filepath']} does not have related .ctm file.")
            return []

        if not os.path.exists(data_entry['words_level_ctm_filepath']):
            logger.warning(f"{data_entry['words_level_ctm_filepath']} does not exist.")
            return []

        words = read_ctm(data_entry['words_level_ctm_filepath'])
        sentences = []
    
        sentence = Sentence()
        for word in words:
            sentence.add_word(word)

            for pm in self.end_of_sentence_punctuation_marks:
                if pm in word.text:
                    sentence.process()
                    sentence = sentence.to_dict()
                    sentences.append(sentence)
                    sentence = Sentence()
                    break

        if sentence.words is not None and len(sentence.words) > 0:
            sentence.process()
            sentence = sentence.to_dict()
            sentences.append(sentence)
            sentence = Sentence()
        
        output_sample_filepath = os.path.abspath(os.path.join(self.output_dir, f"{sentences[0]['sample_id']}.json"))
        with open(output_sample_filepath, 'w') as output_sample_manifest:
            for sentence in sentences:
                line = json.dumps(sentence)
                output_sample_manifest.writelines(f"{line}\n")
        
        data = {'sample_id' :  sentences[0]['sample_id'], 'sample_sentences_filepath' : output_sample_filepath}
        return [DataEntry(data = data)]
    

class MergeSegmentsToSamplesByDuration(BaseParallelProcessor):
    def __init__(
        self,
        max_duration: float = 40.0,
        max_gap_duration: float = 1.0, 
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.max_duration = max_duration
        self.max_gap_duration = max_gap_duration
    
    def prepare(self):
        self.output_dir = self.output_manifest_file.replace(".json", "")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_dataset_entry(self, data_entry):
        if not os.path.exists(data_entry['sample_sentences_filepath']):
            logger.warning(f"{data_entry['sample_sentences_filepath']} does not exist.")
            return []
        
        with open(data_entry['sample_sentences_filepath'], 'r') as sample_sentences_file:
            lines = sample_sentences_file.readlines()
            sentences = [Sentence.from_dict(json.loads(line)) for line in lines]
        
        samples = []

        sample = Sample()
        sample.add_segment(sentences[0])
        
        for sentence in sentences[1 : ]:
            if (# if current sample duration more than max_duration
                sample.duration >= self.max_duration or
                # if current sample duration with sentence duration more than max_duration
                sentence.start_time + sentence.duration - sample.start_time >= self.max_duration or
                # if gap duration between sample and sentence more than max_gap_duration
                sentence.start_time - (sample.start_time + sample.duration) >= self.max_gap_duration):
                    
                    samples.append(sample.to_dict())
                    sample = Sample()
            
            sample.add_segment(sentence)
        
        else:
            if sample.text is not None:
                samples.append(sample.to_dict())
                sample = Sample()
        
        
        samples_output_manifest_filepath = os.path.abspath(os.path.join(self.output_dir, f"{samples[0]['sample_id']}.json"))
        with open(samples_output_manifest_filepath, 'w') as sampls_output_manifest:
            for sample in samples:
                del sample['sample_id']
                line = json.dumps(sample)
                sampls_output_manifest.writelines(f"{line}\n")

        data_entry['samples_filepath'] = samples_output_manifest_filepath
        return [DataEntry(data = data_entry)]

class CropAudios(BaseParallelProcessor):
    def __init__(self, 
                 output_audio_dir: str, 
                 **kwargs,):

        super().__init__(**kwargs)
        self.output_audio_dir = output_audio_dir
    
    def prepare(self):
        os.makedirs(self.output_audio_dir, exist_ok=True)
    
    def process_dataset_entry(self, data_entry):
        data_entries = []
        
        with open(data_entry['samples_filepath'], 'r') as manifest:
            lines = manifest.readlines()
            samples = [json.loads(line) for line in lines]

            audio = AudioSegment.from_file(data_entry['audio_filepath'])
            for i, sample in enumerate(samples):
                start_time = sample['start_time'] * 1000
                end_time = start_time + sample['duration'] * 1000
                audio_segment = audio[start_time : end_time]

                output_audio_filepath = os.path.abspath(os.path.join(self.output_audio_dir, f"{data_entry['sample_id']}_{str(i+1).zfill(4)}.wav"))
                audio_segment.export(output_audio_filepath, format="wav")

                sample['audio_filepath'] = output_audio_filepath
                data_entries.append(DataEntry(data = sample))
        
        return data_entries

class TarDataset(BaseProcessor):
    def __init__(self, 
                 nemo_dir: str,
                 output_dir: str = None,
                 num_shards: int = -1, 
                 max_duration: float = None, 
                 min_duration: float = None,
                 shuffle: bool = False,
                 keep_files_together: bool = False, 
                 sort_in_shards: bool = False,
                 buckets_num: int = 1, 
                 write_metadata: bool = False, 
                 no_shard_manifests: bool = False,
                 shuffle_seed: int = None,
                 force_codec: str = None,
                 workers: int = 1,
                 **kwargs):

        super().__init__(**kwargs)
        self.nemo_dir = nemo_dir
        self.output_dir=output_dir
        self.num_shards=num_shards
        self.max_duration=max_duration
        self.min_duration=min_duration
        self.shuffle=shuffle
        self.keep_files_together=keep_files_together
        self.sort_in_shards=sort_in_shards
        self.buckets_num=buckets_num
        self.write_metadata=write_metadata
        self.no_shard_manifests=no_shard_manifests
        self.shuffle_seed=shuffle_seed
        self.force_codec=force_codec
        self.workers=workers

    def process(self):
        script_path = os.path.join(self.nemo_dir, "scripts", "speech_recognition", "convert_to_tarred_audio_dataset.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"{script_path} not found.")
        
        cmd = ["python", script_path, 
               "--manifest_path", self.input_manifest_file,
               "--num_shards", f"{self.num_shards}",
               "--buckets_num", f"{self.buckets_num}",
               "--workers", f"{self.workers}"]
        
        if self.output_dir is None:
            self.output_dir = self.output_manifest_file.replace(".json", "")
            logger.info(f"Output results will be saved here: f{self.output_dir}")

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        
        os.makedirs(self.output_dir)
        cmd.extend(["--target_dir", self.output_dir])
        
        if self.max_duration:
            cmd.extend(["--max_duration", f"{self.max_duration}"])
        
        if self.min_duration:
            cmd.extend(["--min_duration", f"{self.min_duration}"])
        
        if self.shuffle:
            cmd.append("--shuffle")
        
        if self.keep_files_together:
            cmd.append("--keep_files_together")
        
        if self.sort_in_shards:
            cmd.append(f"--sort_in_shards")
        
        if self.write_metadata:
            cmd.append("--write_metadata")
        
        if self.no_shard_manifests:
            cmd.append("--no_shard_manifests")
        
        if self.shuffle_seed:
            cmd.extend(["--shuffle_seed", f"{self.shuffle_seed}"])
        
        if self.force_codec:
            cmd.extend(["--force_codec", f"{self.force_codec}"])
        
        subprocess.run(cmd)

        self.finalize()
    
    def finalize(self):
        with open(self.output_manifest_file, 'w') as manifest:
            log = {"tarred_data_dir" : self.output_dir}
            line = json.dumps(log)
            manifest.writelines(f"{line}\n")


class UploadToSwiftStack(BaseProcessor):
    def __init__(self, 
                 bucket_name: str,
                 pbss_credentials_path: str, 
                 **kwargs):
        
        super().__init__(**kwargs)
        self.bucket_name = bucket_name
        self.pbss_credentials_path = pbss_credentials_path
    
    def _find_tar_files(self, directory):
        tar_files = []
        for root, dirs, files in os.walk(directory):
            tar_files.extend(glob(os.path.join(root, '*.tar')))
        return tar_files
    
    def process(self):
        with open(self.input_manifest_file, 'r') as manifest:
            lines = manifest.readlines()
            samples = [json.loads(line) for line in lines]
            paths_to_tarred_dirs = [os.path.abspath(sample['tarred_data_dir']) for sample in samples]
        
        s3_config = json.load(open(self.pbss_credentials_path))
        s3_client = boto3.client('s3', **s3_config)

        with  open(self.output_manifest_file, 'w') as output_manifest:
            for i, tarred_dir_path in enumerate(paths_to_tarred_dirs):
                logger.info(f"Processing files from {i + 1}/{len(paths_to_tarred_dirs)} tarred_dir: {tarred_dir_path}")
                
                tar_filepaths = self._find_tar_files(tarred_dir_path)
                logger.info(f"Found {len(tar_filepaths)} tarred files. Starting uploading")

                for tar_filepath in tqdm(tar_filepaths, desc = "Uploading.."):
                    tar_filepath = os.path.abspath(tar_filepath)
                    key = tar_filepath.replace(f"{tarred_dir_path}/", "")
                    logger.info(f"Uploading: {tar_filepath} to {key}")
                    
                    with open(tar_filepath, 'rb') as f:
                        s3_client.upload_fileobj(f, self.bucket_name, key)
                    
                    line = json.dumps({"bucket": self.bucket_name, "key" : key})
                    output_manifest.writelines(f"{line}\n")
