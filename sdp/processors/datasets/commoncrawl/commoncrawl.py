import os
import json
import subprocess
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Union
from pathlib import Path
from operator import lt, le, eq, ne, ge, gt
import soundfile as sf
from sacrebleu import BLEU

from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor, DataEntry
from sdp.logging import logger
from sdp.processors.datasets.commoncrawl.harv_utils import ffmpeg_convert, txt2vtt, make_trans_list, get_vtt_text, text2lid, load_manifest, read_jsonl, write_jsonl, split_by_vtt_new
from scipy.spatial import distance

class UseSonar(BaseProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        input_text_field: str,
        input_audio_field: str,
        output_field: str,
        speech_encoder_model: str,
        text_encoder_lang: str,
        text_encoder_model: str,
        batch_size: int = 64,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        import torch  # importing after nemo to make sure users first install nemo, instead of torch, then nemo
        from torch.nn import PairwiseDistance
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        
        from sonar.models.sonar_speech.loader import load_sonar_speech_model
        from sonar.models.sonar_text import (
            load_sonar_text_decoder_model,
            load_sonar_text_encoder_model,
            load_sonar_tokenizer,
        )
        self.output_field = output_field
        self.input_text_field = input_text_field
        self.input_audio_field = input_audio_field
        self.batch_size = batch_size
        self.device = device
        self.text_encoder_lang = text_encoder_lang
        self.text_encoder_model = load_sonar_text_encoder_model(text_encoder_model, device=self.device).eval()
        self.text_tokenizer = load_sonar_tokenizer(text_encoder_model)
        self.speech_encoder_model = load_sonar_speech_model(speech_encoder_model, device=self.device).eval()
        self.pdist = PairwiseDistance(p=2)
        self.s2vec_model = SpeechToEmbeddingModelPipeline(encoder=self.speech_encoder_model)
        self.text_embedding_pipeline = TextToEmbeddingModelPipeline(self.text_encoder_model, self.text_tokenizer)
    
    def process(self):
        manifest = load_manifest(Path(self.input_manifest_file))

        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)
        with Path(self.output_manifest_file).open('w') as f:
            for item in tqdm(manifest):
                input_texts = [item[self.input_text_field]]
                input_audios = [item[self.input_audio_field]]
                dist = self.get_pdist(input_texts, input_audios)
                item[self.output_field] = dist
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def get_pdist(self, input_texts, input_audios):
        text_emb = self.text_embedding_pipeline.predict(input = input_texts,
                                            batch_size = 1,
                                            source_lang=self.text_encoder_lang)

        audio_emb = self.s2vec_model.predict(input = input_audios,
                                            batch_size = 1,
                                            n_parallel = 1,
                                            pad_idx = 0,
                                            n_prefetched_batches = 1,)
        # pdist = self.pdist(text_emb, audio_emb).numpy().squeeze().astype(float).tolist()
        pdist = distance.cdist(text_emb.numpy().astype(float), audio_emb.numpy().astype(float), 'sqeuclidean').squeeze().tolist()
        return pdist
    
    def process_batch(self):
        manifest, dict_list = load_manifest(Path(self.input_manifest_file), keys = [self.input_audio_field, self.input_text_field])
        manifest_len = len(manifest)
        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)
        with Path(self.output_manifest_file).open('w') as f:
            for start in tqdm(range(0, manifest_len, self.batch_size)):
                stop = start + self.batch_size
                input_texts = dict_list[self.input_text_field][start:stop]
                input_audios = dict_list[self.input_audio_field][start:stop]
                manifest_batch = manifest[start:stop]

                dists = self.get_pdist(input_texts, input_audios)
                for item, dist in zip(manifest_batch, dists):
                    item[self.output_field] = dist
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

class BLEUScore(BaseParallelProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        ref_field: str,
        hyp_field: str,
        output_field: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ref_field = ref_field
        self.hyp_field = hyp_field
        self.output_field = output_field
        self.scorer = BLEU(effective_order=True)
        
    def process_dataset_entry(self, data_entry):
        ref = data_entry[self.ref_field]
        hyp = data_entry[self.hyp_field]
        
        res = self.scorer.sentence_score(hypothesis=hyp,
                            references=[ref])
        data_entry[self.output_field] = res.score
        return [DataEntry(data=data_entry)]

class Subprocess(BaseProcessor):
    """This processor performs ASR inference on each utterance of the input manifest.

    ASR predictions will be saved in the ``pred_text`` key.

    Args:
        pretrained_model (str): the name of the pretrained NeMo ASR model
            which will be used to do inference.
        batch_size (int): the batch size to use for ASR inference. Defaults to 32.

    Returns:
         The same data as in the input manifest with an additional field
         ``pred_text`` containing ASR model's predictions.
    """

    def __init__(
        self,
        cmd: str,
        input_manifest_arg: str = "",
        output_manifest_arg: str = "",
        arg_separator: str = "=",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_manifest_arg = input_manifest_arg
        self.output_manifest_arg = output_manifest_arg
        self.arg_separator = arg_separator
        self.cmd = cmd

    def process(self):
        """This will add "pred_text" key into the output manifest."""
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        if self.cmd.find(self.input_manifest_file) != -1 or self.cmd.find(self.output_manifest_file) != -1:
            logger.error("input_manifest_file "+self.input_manifest_file+" and output_manifest_file "+self.output_manifest_file+" should be exluded from cmd line!")
            raise ValueError
        process_args = [x for x in self.cmd.split(" ") if x]
        if self.arg_separator == " ":
            if self.input_manifest_arg:
                process_args.extend([self.input_manifest_arg, self.input_manifest_file])
            if self.output_manifest_arg:
                process_args.extend([self.output_manifest_arg, self.output_manifest_file])
        else:
            if self.input_manifest_arg:
                process_args.extend([self.input_manifest_arg + self.arg_separator + self.input_manifest_file])
            if self.output_manifest_arg:
                process_args.extend([self.output_manifest_arg + self.arg_separator + self.output_manifest_file])

        subprocess.run(process_args)

class NmtSubprocess(Subprocess):
    """This processor performs ASR inference on each utterance of the input manifest.

    ASR predictions will be saved in the ``pred_text`` key.

    Args:
        pretrained_model (str): the name of the pretrained NeMo ASR model
            which will be used to do inference.
        batch_size (int): the batch size to use for ASR inference. Defaults to 32.

    Returns:
         The same data as in the input manifest with an additional field
         ``pred_text`` containing ASR model's predictions.
    """

    def __init__(
        self,
        input_field: str,
        output_field: str,
        srctext_file: str,
        tgtout_file: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        self.srctext_file = srctext_file
        self.tgtout_file = tgtout_file
        self.cmd = self.cmd + " --srctext" + self.arg_separator + self.srctext_file + " --tgtout" + self.arg_separator + self.tgtout_file

    def process(self):
        df1 = read_jsonl(self.input_manifest_file)
        with Path(self.srctext_file).open('w') as f:
            for input_field in df1[self.input_field]:
                f.write(input_field + "\n")
        
        super().process()

        with Path(self.tgtout_file).open('r') as f:
            tgtout = [l.strip() for l in f]
        df1[self.output_field] = tgtout
        write_jsonl(df1, self.output_manifest_file)

class PreserveByValue(BaseParallelProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        input_field: str,
        target_value: Union[int, str],
        operator: str = "eq",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.target_value = target_value
        if operator == "lt":
            self.operator = lt
        elif operator == "le":
            self.operator = le
        elif operator == "eq":
            self.operator = eq
        elif operator == "ne":
            self.operator = ne
        elif operator == "ge":
            self.operator = ge
        elif operator == "gt":
            self.operator = gt

    def process_dataset_entry(self, data_entry):
        input_value = data_entry[self.input_field]
        target = self.target_value
        if self.operator(input_value, target):
            return [DataEntry(data=data_entry)]
        else:
            return [DataEntry(data=None)]
    
class Lang2Iso(BaseParallelProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        input_lang_field: str,
        output_lang_field: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_lang_field = input_lang_field
        self.output_lang_field = output_lang_field
        self.iso_m = {'English':'en', 'Spanish':'es', 'Basque':'eu', 'Dutch':'nl', 'Welsh':'cy', 'Italian':'it',
            'Catalan':'ca', 'Maltese':'mt', 'Swedish':'sv', 'French':'fr', 'German':'de', 'Chuvash':'cv',
            'Kinyarwanda':'rw', 'Polish':'pl', 'Kabyle':'kab', 'Interlingua': 'ua', 'Portuguese': 'pt', 'Hakha_Chin': 'cnh', 'Romansh_Sursilvan':'roh', 'Breton':'br', 'Esperanto':'epo', 'Czech':'ces', 'Latvian':'lav',
            'Indonesian':'ind', 'Slovenian':'slv', 'Turkish':'tur', 'Frisian':'frr', 'Tatar':'tat', 'Persian':'fas', 'Estonian':'est', 'Romanian':'rum', 'Chinese_Hongkong':'zh', 'Chinese_Taiwan':'zh',
            'Georgian':'kat', 'Kyrgyz':'kir', 'Dhivehi':'div', 'Sakha':'sah'}
        
    def process_dataset_entry(self, data_entry):
        data_entry[self.output_lang_field] = self.iso_m[data_entry[self.input_lang_field]]
        return [DataEntry(data=data_entry)]

class SplitByVttSentence(BaseParallelProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        splited_audio_dir: str,
        source_audio_field: str,
        target_audio_field: str,
        duration_field: str,
        text_field: str,
        vtt_field: str,
        proxy_fields: List[str] = [],
        duration_threshold: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.splited_audio_dir = splited_audio_dir
        self.source_audio_field = source_audio_field
        self.target_audio_field = target_audio_field
        self.duration_field = duration_field
        self.text_field = text_field
        self.vtt_field = vtt_field
        self.duration_threshold = duration_threshold
        self.proxy_fields = proxy_fields

    def prepare(self):
        os.makedirs(self.splited_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        vtt_file = data_entry[self.vtt_field]
        source_audio = data_entry[self.source_audio_field]
        res_list = []

        if os.path.isfile(source_audio):
            data, samplerate = sf.read(source_audio)
            text_list, start_s, end_s = split_by_vtt_new(vtt_file, samplerate)
            text_c = ''
            start_c, end_c = 0, 0
            if text_list:
                for text, start_sr, end_sr in zip(text_list, start_s, end_s):
                    text_c += " " + text
                    if start_c==0:
                        start_c = start_sr
                    else:
                        pass
                    end_c = end_sr
                    if len(text_c)>0 and (end_c - start_c > self.duration_threshold * samplerate or text_c[-1] == "." or text_c[-1] == "?"):
                        res_list.append(self.makeDataEntry(data_entry, data, vtt_file, samplerate, text_c, start_c, end_c))
                        text_c = ''
                        start_c, end_c = 0, 0
                    else:
                        pass
                if len(text_c)>0 and start_c!=0:
                    res_list.append(self.makeDataEntry(data_entry, data, vtt_file, samplerate, text_c, start_c, end_c))
                
        return res_list

    def makeDataEntry(self, data_entry, data, vtt_file, samplerate, text_c, start_c, end_c):
        data_sample = data[start_c:end_c]
        wav_save_file = os.path.join(self.splited_audio_dir, '/'.join(os.path.splitext(vtt_file)[0].split('/')[-2:]), str(int(start_c/(samplerate/1000)))+"-"+str(int(end_c/(samplerate/1000)))+".wav")
        if not os.path.isfile(wav_save_file):
            os.makedirs(os.path.split(wav_save_file)[0], exist_ok=True)
            sf.write(wav_save_file, data_sample, samplerate)
        
        data = {self.target_audio_field: wav_save_file,
                    self.duration_field: data_sample.shape[0]/samplerate,
                    self.text_field: text_c.strip(),
                    }
        for proxy_field in self.proxy_fields:
            data[proxy_field] = data_entry[proxy_field]
        return DataEntry(data = data)


class SplitByVtt(BaseParallelProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        splited_audio_dir: str,
        source_audio_field: str,
        text_lang_field: str,
        audio_lang_field: str,
        key_field: str,
        target_audio_field: str,
        duration_field: str,
        text_field: str,
        vtt_field: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.splited_audio_dir = splited_audio_dir
        self.source_audio_field = source_audio_field
        self.text_lang_field = text_lang_field
        self.audio_lang_field = audio_lang_field
        self.key_field = key_field
        self.target_audio_field = target_audio_field
        self.duration_field = duration_field
        self.text_field = text_field
        self.vtt_field = vtt_field

    def prepare(self):
        os.makedirs(self.splited_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        key = data_entry[self.key_field]
        vtt_file = data_entry[self.vtt_field]
        source_audio = data_entry[self.source_audio_field]
        res_list = []

        if os.path.isfile(source_audio):
            wav_list, text_list, dur_list = split_by_vtt(vtt_file, source_audio, self.splited_audio_dir)
            if wav_list:
                for wav, text, dur in zip(wav_list, text_list, dur_list):
                    res_list.append(DataEntry(data = {self.target_audio_field: wav,
                                        self.duration_field: dur,
                                        self.text_field: text,
                                        self.audio_lang_field: data_entry[self.audio_lang_field],
                                        self.text_lang_field: data_entry[self.text_lang_field],
                                        self.key_field: key}))
        return res_list

class AudioLid(BaseProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        input_audio_field: str,
        pretrained_model: str,
        output_lang_field: str,
        device: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_audio_field = input_audio_field
        self.pretrained_model = pretrained_model
        self.output_lang_field = output_lang_field
        self.device = device
    
    def process(self):
        import torch  # importing after nemo to make sure users first install nemo, instead of torch, then nemo
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.pretrained_model)

        if self.device is None:
            if torch.cuda.is_available():
                model = model.cuda()
            else:
                model = model.cpu()
        else:
            model = model.to(self.device)

        manifest = load_manifest(Path(self.input_manifest_file))

        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)
        with Path(self.output_manifest_file).open('w') as f:
            for item in tqdm(manifest):
                audio_file = item[self.input_audio_field]

                try:
                    lang = model.get_label(audio_file, 60*5)
                except Exception as e:
                    logger.warning("AudioLid " + audio_file+ " " + str(e))
                    lang = None

                if lang:
                    item[self.output_lang_field] = lang
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')


class TextLid(BaseProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        input_text_field: str,
        pretrained_model: str,
        output_lang_field: str,
        device: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_text_field = input_text_field
        self.pretrained_model = pretrained_model
        self.output_lang_field = output_lang_field
        self.device = device
    
    def process(self):
        import torch  # importing after nemo to make sure users first install nemo, instead of torch, then nemo
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        text_model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model)

        if self.device is None:
            if torch.cuda.is_available():
                text_model = text_model.cuda()
            else:
                text_model = text_model.cpu()
        else:
            text_model = text_model.to(self.device)

        manifest = load_manifest(Path(self.input_manifest_file))

        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)

        with Path(self.output_manifest_file).open('w') as f:
            for item in tqdm(manifest):
                text = item[self.input_text_field]
                if text:
                    lid = text2lid(text_model, tokenizer, text)
                else:
                    lid = None
            
                if lid:
                    item[self.output_lang_field] = lid
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

class AllVttText(BaseParallelProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        output_text_field: str,
        input_filepath_field: str = "vtt_filepath",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_text_field = output_text_field
        self.input_filepath_field = input_filepath_field
        
    def process_dataset_entry(self, data_entry):
        vtt_file = data_entry[self.input_filepath_field]
        res_list = [DataEntry(data=None)]
        if os.path.isfile(vtt_file):
            try:
                data_entry[self.output_text_field] = get_vtt_text(vtt_file)
                res_list = [DataEntry(data=data_entry)]
            except Exception as e:
                logger.warning("AllVttText " + vtt_file + " " + str(e))
        return res_list


class TxtToVtt(BaseParallelProcessor):
    """
        Args:
        raw_data_dir (str): where to put raw downloaded data.
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
        target_samplerate (int): sample rate to resample to. Defaults to 16000.
        target_nchannels (int): target number of channels. Defaults to 1.
    """
    def __init__(
        self,
        vtt_files_dir: str,
        key_field: str,
        text_field: str,
        vtt_field: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vtt_files_dir = vtt_files_dir
        self.key_field = key_field
        self.text_field = text_field
        self.vtt_field = vtt_field
        
        self.trans_list = make_trans_list()

    def prepare(self):
        os.makedirs(self.vtt_files_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        key = data_entry[self.key_field]
        text_file = data_entry[self.text_field]
        os.makedirs(os.path.join(self.vtt_files_dir, key.split("/")[0]), exist_ok=True)

        vtt_file = os.path.join(self.vtt_files_dir, key) + ".vtt"
        
        txt2vtt(text_file, vtt_file, self.trans_list)

        data_entry[self.vtt_field] = vtt_file

        return [DataEntry(data=data_entry)]

class ReadParquet(BaseParallelProcessor):
    """
        Args:
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
    """
    def __init__(
        self,
        output_video_field: str,
        output_vtt_field: str,
        key_field: str,
        raw_data_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_video_field = output_video_field
        self.output_vtt_field = output_vtt_field
        self.key_field = key_field
        self.raw_data_dir = Path(raw_data_dir)

    def prepare(self):
        parquets = [str(self.raw_data_dir / p) for p in self.raw_data_dir.rglob('*.parquet')]
        self.urls = None
        for parquet in tqdm(parquets):
            try:
                df1 = pd.read_parquet(parquet, engine='fastparquet').sort_values("key").set_index("key")
                if self.urls is None:
                    self.urls = df1
                else:
                    self.urls = pd.concat([self.urls, df1])
            except Exception as e:
                logger.warning(str(e) + ", file: " + parquet)
            
    def process_dataset_entry(self, data_entry):
        key = data_entry[self.key_field]
        key = key.split("/")[1]
        try:
            data_entry[self.output_video_field] = self.urls.loc[key]['url']
            data_entry[self.output_vtt_field] = self.urls.loc[key]['caption']
        except:
            data_entry[self.output_video_field] = "NN"
            data_entry[self.output_vtt_field] = "NN"
            logger.warning("Key without URL or caption: " + key)
        return [DataEntry(data=data_entry)]

class CreateInitialManifestCC(BaseParallelProcessor):
    """
        Args:
        raw_data_dir (str): where to put raw downloaded data.
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
        target_samplerate (int): sample rate to resample to. Defaults to 16000.
        target_nchannels (int): target number of channels. Defaults to 1.
    """
    def __init__(
        self,
        raw_data_dir: str,
        resampled_audio_dir: str,
        audio_field: str,
        video_field: str,
        key_field: str,
        text_field: str,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.audio_field = audio_field
        self.video_field = video_field
        self.key_field = key_field
        self.text_field = text_field
        self.resampled_audio_dir = resampled_audio_dir
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def prepare(self):
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def read_manifest(self):
        videos = [str(self.raw_data_dir / video) for video in self.raw_data_dir.rglob('*.jpg')]
        texts = [str(self.raw_data_dir / text) for text in self.raw_data_dir.rglob('*.txt')]
        v_df = pd.DataFrame({self.video_field: videos})
        t_df = pd.DataFrame({self.text_field: texts })
        v_df[self.key_field] = v_df[self.video_field].apply(lambda x: os.path.splitext(x)[0][-13:])
        t_df[self.key_field] = t_df[self.text_field].apply(lambda x: os.path.splitext(x)[0][-13:])
        v_df = v_df.drop_duplicates(self.key_field)
        t_df = t_df.drop_duplicates(self.key_field)
        vt_df = v_df.merge(t_df, on=self.key_field, how="left")
        return vt_df.values

    def process_dataset_entry(self, data_entry):
        (video,	key, text) = data_entry
        os.makedirs(os.path.join(self.resampled_audio_dir, key.split("/")[0]), exist_ok=True)
        audio = os.path.join(self.resampled_audio_dir, key) + ".wav"
        if not os.path.isfile(audio):
            ffmpeg_convert(video, audio, self.target_samplerate, self.target_nchannels)

        data = {self.audio_field: audio,
                self.video_field: video,
                self.key_field: key,
                self.text_field: text}
        return [DataEntry(data=data)]

class FfmpegConvert(BaseParallelProcessor):
    """
        Args:
        video_field (str): field with path to video file in the input manifest
        audio_field (str): field with path to audio file in the output manifest
        key_field (str): field with key value
        resampled_audio_dir (str): where to put re-sampled and trimmed wav files.
        target_samplerate (int): sample rate to resample to. Defaults to 16000.
        target_nchannels (int): target number of channels. Defaults to 1.
    """
    def __init__(
        self,
        resampled_audio_dir: str,
        video_field: str,
        audio_field: str,
        key_field: str,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_field = audio_field
        self.video_field = video_field
        self.key_field = key_field
        self.resampled_audio_dir = resampled_audio_dir
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def process_dataset_entry(self, data_entry):
        video = data_entry[self.video_field]
        key = os.path.splitext(data_entry[self.video_field])[0][-13:]
        os.makedirs(os.path.join(self.resampled_audio_dir, key.split("/")[0]), exist_ok=True)
        audio = os.path.join(self.resampled_audio_dir, key) + ".wav"

        if not os.path.isfile(audio):
            ffmpeg_convert(video, audio, self.target_samplerate, self.target_nchannels)

        data_entry[self.audio_field]= audio
        data_entry[self.key_field] = key
        return [DataEntry(data=data_entry)]