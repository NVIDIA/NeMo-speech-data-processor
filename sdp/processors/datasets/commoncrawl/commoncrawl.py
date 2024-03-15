import json
import math
import os
import re
import shutil
import subprocess
from operator import eq, ge, gt, le, lt, ne
from pathlib import Path
from typing import Dict, List, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sacrebleu import BLEU
from scipy.spatial import distance
from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)
from sdp.processors.datasets.commoncrawl.harv_utils import (
    audio_duration,
    ffmpeg_convert,
    get_vtt_text,
    load_manifest,
    make_trans_list,
    read_jsonl,
    split_by_vtt_new,
    text2lid,
    txt2vtt,
    write_jsonl,
)


class ManifestToUtf8(BaseProcessor):
    """
    Processor to convert manifest file to UTF-8 encoding.
    """

    def process(self):
        with open(self.output_manifest_file, "w") as wout, open(self.input_manifest_file) as win:
            for line in win:
                print(json.dumps(json.loads(line), ensure_ascii=False), file=wout)


class DropAbsPath(BaseParallelProcessor):
    """
    Drop absolute path

    Args:
        path_key (str): where to get path to wav file.
        abs_path_to_drop (str): string to drop from the bigining of path to wav file.
    """

    def __init__(
        self,
        path_key: str,
        abs_path_to_drop: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path_key = path_key
        self.abs_path_to_drop = abs_path_to_drop

    def process_dataset_entry(self, data_entry):
        audio_filepath = data_entry[self.path_key]
        data_entry[self.path_key] = audio_filepath[len(self.abs_path_to_drop) :]
        return [DataEntry(data=data_entry)]


class CopyFiles(BaseParallelProcessor):
    def __init__(
        self,
        file_field: str,
        path_to_copy: str,
        path_levels: str = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.file_field = file_field
        self.path_to_copy = path_to_copy
        self.path_levels = path_levels

    def prepare(self):
        os.makedirs(self.path_to_copy, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        rel_file_path = "/".join(data_entry[self.file_field].split("/")[-self.path_levels :])
        new_file_path = os.path.join(self.path_to_copy, rel_file_path)

        if not os.path.isfile(new_file_path):
            os.makedirs(os.path.split(new_file_path)[0], exist_ok=True)
            shutil.copyfile(data_entry[self.file_field], new_file_path)
        data_entry[self.file_field] = new_file_path
        return [DataEntry(data=data_entry)]


class GetSpecificFiles(BaseParallelProcessor):
    def __init__(
        self,
        file_field: str,
        path_to_copy: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.file_field = file_field
        self.path_to_copy = path_to_copy

        self.split_map = set(
            [
                '0634236',
                '0693626',
                '0029743',
                '0881322',
                '0357427',
                '0455788',
                '0198472',
                '0496259',
                '0812890',
                '0142281',
                '0076612',
                '0629004',
                '0931592',
                '0577447',
                '0768107',
                '0907768',
                '0963898',
                '0671754',
                '0851569',
                '0896715',
                '0366790',
                '0837221',
                '0733702',
                '0278253',
                '0738313',
                '0437256',
                '0558223',
                '0292533',
                '0777911',
                '0826607',
                '0544257',
                '0744206',
                '0576248',
                '0307575',
                '0307577',
                '0879895',
                '0006783',
                '0006755',
                '0125649',
                '0896701',
            ]
        )

    def prepare(self):
        os.makedirs(self.path_to_copy, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        file_id = os.path.splitext(data_entry[self.file_field])[0].split("/")[-1]
        if file_id in self.split_map:
            shutil.copyfile(data_entry[self.file_field], os.path.join(self.path_to_copy, file_id + ".wav"))
            return [DataEntry(data=data_entry)]
        else:
            return []


class TrainDevTestSplitCC(BaseParallelProcessor):
    """Custom train-dev-test split for CORAAL dataset.

    Split is done speaker-wise, so the same speakers don't appear in different
    splits.

    Args:
        data_split (str): train, dev or test.
        lang (str): language to process.

    Returns:
        All the same fields as in the input manifest, but only a subset of
        the data is retained.
    """

    def __init__(
        self,
        data_split: str,
        lang: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if data_split not in ["train", "dev", "test"]:
            raise ValueError("data_split has to be either train, dev or test")
        self.data_split = data_split
        self.lang = lang

        self.split_map = {}
        self.split_map["en"] = {}
        self.split_map["en"]["dev"] = set(
            [
                '0634236',
                '0693626',
                '0029743',
                '0881322',
                '0357427',
                '0455788',
                '0198472',
                '0496259',
                '0812890',
                '0142281',
                '0076612',
                '0629004',
                '0931592',
                '0577447',
                '0768107',
                '0907768',
                '0963898',
                '0671754',
                '0851569',
                '0896715',
            ]
        )
        self.split_map["en"]["test"] = set(
            [
                '0366790',
                '0837221',
                '0733702',
                '0278253',
                '0738313',
                '0437256',
                '0558223',
                '0292533',
                '0777911',
                '0826607',
                '0544257',
                '0744206',
                '0576248',
                '0307575',
                '0307577',
                '0879895',
                '0006783',
                '0006755',
                '0125649',
                '0896701',
            ]
        )
        self.split_map["de"] = {}
        self.split_map["de"]["dev"] = set(
            [
                '0383522',
                '0327835',
                '0327898',
                '0619871',
                '0387103',
                '0854766',
                '0738911',
                '0739038',
                '0854558',
                '0505561',
                '0735963',
                '0086041',
                '0967593',
                '0114210',
                '0098270',
                '0387140',
                '0917035',
                '0327745',
                '0914212',
                '0739071',
            ]
        )
        self.split_map["de"]["test"] = set(
            [
                '0076939',
                '0589098',
                '0916988',
                '0268959',
                '0085896',
                '0327813',
                '0085897',
                '0739103',
                '0502188',
                '0034822',
                '0327729',
                '0572412',
                '0327680',
                '0027277',
                '0324720',
                '0209876',
                '0027226',
                '0268926',
                '0209776',
                '0738970',
            ]
        )
        self.split_map["pl"] = {}
        self.split_map["pl"]["dev"] = set(
            [
                '0977373',
                '0949141',
                '0455759',
                '0357429',
                '0401864',
                '0714974',
                '0422716',
                '0363476',
                '0714976',
                '0927100',
            ]
        )
        self.split_map["pl"]["test"] = set(
            [
                '0157903',
                '0115644',
                '0774572',
                '0688432',
                '0258376',
                '0396163',
                '0456013',
                '0571489',
                '0157653',
                '0062567',
            ]
        )
        self.split_map["fr"] = {}
        self.split_map["fr"]["dev"] = set(
            [
                '0588135',
                '0706751',
                '0533213',
                '0920924',
                '0355413',
                '0985711',
                '0113477',
                '0533044',
                '0089551',
                '0944509',
                '0944576',
                '0766533',
                '0263084',
                '0113490',
                '0647104',
                '0273918',
                '0473607',
                '0706753',
                '0800223',
                '0300105',
                '0944416',
                '0566712',
                '0533102',
                '0177064',
                '0029651',
                '0215767',
                '0054412',
                '0236920',
                '0885068',
                '0296098',
                '0113592',
                '0706610',
                '0473383',
                '0330163',
                '0681542',
                '0272523',
                '0985709',
                '0564446',
                '0944481',
                '0587986',
                '0804060',
                '0236908',
                '0969694',
                '0054058',
                '0800671',
                '0236923',
                '0986025',
                '0770086',
                '0825692',
                '0968870',
                '0152315',
                '0533147',
                '0647027',
                '0029342',
                '0272698',
                '0153863',
                '0355323',
                '0988779',
                '0985959',
                '0237013',
                '0338134',
                '0885097',
                '0507678',
                '0507687',
                '0944485',
                '0825768',
                '0742440',
                '0969664',
                '0885089',
                '0117211',
                '0296044',
                '0985958',
                '0214384',
                '0021267',
                '0565392',
                '0388467',
                '0151715',
                '0861950',
                '0112768',
                '0113596',
                '0621657',
                '0236860',
                '0647128',
                '0058479',
                '0803614',
                '0177501',
                '0533110',
                '0566787',
                '0944496',
                '0859701',
                '0885165',
                '0212639',
                '0054532',
                '0919263',
                '0740701',
            ]
        )
        self.split_map["fr"]["test"] = set(
            [
                '0473649',
                '0390470',
                '0296024',
                '0355365',
                '0314592',
                '0682498',
                '0534637',
                '0270580',
                '0532999',
                '0373977',
                '0622032',
                '0825761',
                '0923303',
                '0113485',
                '0825868',
                '0473710',
                '0511698',
                '0844353',
                '0801733',
                '0091695',
                '0452351',
                '0825872',
                '0969173',
                '0986055',
                '0970208',
                '0141266',
                '0149629',
                '0296117',
                '0153112',
                '0801752',
                '0030816',
                '0508766',
                '0029390',
                '0825877',
                '0271152',
                '0388655',
                '0743376',
                '0177466',
                '0153032',
                '0329945',
                '0473606',
                '0986015',
                '0096178',
                '0089561',
                '0440564',
                '0741466',
                '0499703',
                '0272514',
                '0944571',
                '0919512',
                '0646950',
                '0533215',
                '0760703',
                '0733028',
                '0113488',
                '0825739',
                '0492402',
                '0214463',
                '0154278',
                '0801877',
                '0825675',
                '0675029',
                '0801729',
                '0414446',
                '0054425',
                '0279176',
                '0296100',
                '0355317',
                '0733026',
                '0089548',
                '0177502',
                '0851638',
                '0851640',
                '0448606',
                '0803096',
                '0766603',
                '0507914',
                '0092173',
                '0647061',
                '0473564',
                '0706765',
                '0766538',
                '0295994',
                '0851630',
                '0029358',
                '0647062',
                '0825838',
                '0153786',
                '0944526',
                '0944484',
                '0588046',
                '0706820',
                '0177465',
                '0622092',
                '0332657',
                '0944480',
            ]
        )

    def process_dataset_entry(self, data_entry):
        file_id = os.path.splitext(data_entry["audio_filepath"])[0].split("/")[-2]
        if self.data_split == "train":
            if file_id not in self.split_map[self.lang]["dev"] and file_id not in self.split_map[self.lang]["test"]:
                return [DataEntry(data=data_entry)]
        else:
            if file_id in self.split_map[self.lang][self.data_split]:
                return [DataEntry(data=data_entry)]
        return []


class JoinBy(BaseProcessor):
    """
    This processor join several lines into one using key input_field

    Args:
        input_field (str): where to get path to wav file.
        text_field (str): where to put resulted text.
        audio_field (str): where to put resulted wav file.

    Returns:
        All the same fields as in the input manifest plus audio_field
    """

    def __init__(
        self,
        input_field: str,
        text_field: str = "text",
        audio_field: str = 'audio_filepath',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.text_field = text_field
        self.audio_field = audio_field

    def process(self):
        df1 = read_jsonl(self.input_manifest_file)
        pattern = re.compile("\s{2,}")
        df1[self.text_field] = df1[self.text_field].apply(lambda x: pattern.sub(" ", x).strip())
        # df1["source"] = df1["audio_filepath"].apply(lambda x: x.split("/")[-2])

        df2 = pd.DataFrame(
            df1.groupby(self.input_field).apply(lambda in_df: " ".join(in_df[self.text_field].tolist())),
            columns=[self.text_field],
        ).reset_index()
        df2[self.audio_field] = df2[self.input_field]
        write_jsonl(df2[[self.audio_field, self.text_field]], self.output_manifest_file)


class AudioDuration(BaseParallelProcessor):
    """
    Count audio duration using audio file path from input_field

    Args:
        input_field (str): where to get path to wav file.
        output_field (str): where to put to audio duration.
    Returns:
        All the same fields as in the input manifest plus output_field
    """

    def __init__(
        self,
        input_field: str,
        output_field: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field

    def process_dataset_entry(self, data_entry):
        audio_filepath = data_entry[self.input_field]
        try:
            data_entry[self.output_field] = audio_duration(audio_filepath)
        except Exception as e:
            logger.warning(str(e) + " file: " + audio_filepath)
            data_entry[self.output_field] = -1.0
        return [DataEntry(data=data_entry)]


class EvalBandwidth(BaseParallelProcessor):
    """
    Count audio bandwidth using audio file path from input_field

    Args:
        input_field (str): where to get path to wav file.
        output_field (str): where to put to frequency bandwidth.
        threshold (str): power threshold (in dB relative to peak power in spectrum bin) to estimate frequency bandwidth.

    Returns:
        All the same fields as in the input manifest plus output_field.
    """

    def __init__(
        self,
        input_field: str,
        output_field: str,
        threshold: int = -50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        self.threshold = threshold

    def process_dataset_entry(self, data_entry):
        audio_filepath = data_entry[self.input_field]
        data, samplerate = sf.read(audio_filepath)
        freqband = self.eval_bandwidth(data, samplerate, threshold=self.threshold)
        data_entry[self.output_field] = freqband
        return [DataEntry(data=data_entry)]

    def eval_bandwidth(self, signal, sr, threshold=-50):
        time_stride = 0.01
        hop_length = int(sr * time_stride)
        n_fft = 512
        spectrogram = np.mean(
            np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length, window='blackmanharris')) ** 2, axis=1
        )
        power_spectrum = librosa.power_to_db(S=spectrogram, ref=np.max, top_db=100)
        freqband = 0
        for idx in range(len(power_spectrum) - 1, -1, -1):
            if power_spectrum[idx] > threshold:
                freqband = idx / n_fft * sr
                break
        return freqband


class SplitByAligner(BaseParallelProcessor):
    """
    Split wav file using NFA aligner fields: nfa_start, nfa_duration

    Args:
        input_field (str): field to get source wav file names.
        output_field: (str): field to put splited wav file names.
        splited_audio_dir (str): where to save splited wav files.
    Returns:
        All the same fields as in the input manifest plus output_field.
    """

    def __init__(
        self,
        input_field: str,
        output_field: str,
        splited_audio_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        self.splited_audio_dir = splited_audio_dir

    def prepare(self):
        os.makedirs(self.splited_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        audio_filepath = data_entry[self.input_field]

        # print(data_entry)
        data, samplerate = sf.read(audio_filepath)
        nfa_start = data_entry["nfa_start"]
        nfa_duration = data_entry["nfa_duration"]

        if math.isnan(nfa_start) or math.isnan(nfa_duration) or math.isnan(samplerate):
            print(audio_filepath, nfa_start, nfa_duration)
            data_entry[self.output_field] = data_entry['audio_filepath']
        else:
            start = int(nfa_start * samplerate)
            duration = int(nfa_duration * samplerate)

            data_sample = data[start : start + duration]

            wav_save_file = os.path.join(
                self.splited_audio_dir,
                '/'.join(os.path.splitext(audio_filepath)[0].split('/')[-2:]),
                str(int(start * 1000 / samplerate)) + "-" + str(int((start + duration) * 1000 / samplerate)) + ".wav",
            )
            if not os.path.isfile(wav_save_file):
                os.makedirs(os.path.split(wav_save_file)[0], exist_ok=True)
                sf.write(wav_save_file, data_sample, samplerate)
            data_entry[self.output_field] = wav_save_file
        return [DataEntry(data=data_entry)]


class ASR_HF(BaseProcessor):
    """
    Transcribe usinf ASR model from HuggingFace.

    Args:
        pretrained_model (str): name of pretrained model on HuggingFace.
        output_text_field (str): field to save transcription result.
        device (str): Inference device.
        batch_size (str): Inference batch size.
    Returns:
        All the same fields as in the input manifest plus output_text_field.
    """

    def __init__(
        self,
        pretrained_model: str,
        output_text_field: str,
        device: str = None,
        batch_size: str = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_model = pretrained_model
        self.output_text_field = output_text_field
        self.device = device
        self.batch_size = batch_size

    def process(self):
        import torch
        from huggingsound import SpeechRecognitionModel

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        model = SpeechRecognitionModel(self.pretrained_model, device=self.device, letter_case=None)

        manifest, key_dict = load_manifest(Path(self.input_manifest_file), keys=["audio_filepath"])
        audio_paths = key_dict["audio_filepath"]

        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)

        transcriptions = model.transcribe(paths=audio_paths, batch_size=self.batch_size, decoder=None)

        with Path(self.output_manifest_file).open('w') as f:
            for item, transcription in tqdm(zip(manifest, transcriptions)):
                item[self.output_text_field] = transcription["transcription"]
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


class UseSonar(BaseProcessor):
    """
    Count vector distance using Sonar library.

    Args:
        input_text_field (str): field with text to process.
        input_audio_field (str): field with audio file path to process.
        output_field (str): field to save distance.
        speech_encoder_model (str): name of pretrained speech encoder model.
        text_encoder_lang (str): language of text.
        text_encoder_model (str): name of pretrained text encoder model.
        batch_size (int): batch size for inference.
        device (str): device to inference on it.
    Returns:
        All the same fields as in the input manifest plus output_field.
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
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        from sonar.models.sonar_speech.loader import load_sonar_speech_model
        from sonar.models.sonar_text import (
            load_sonar_text_decoder_model,
            load_sonar_text_encoder_model,
            load_sonar_tokenizer,
        )
        from torch.nn import PairwiseDistance

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
        text_emb = self.text_embedding_pipeline.predict(
            input=input_texts, batch_size=1, source_lang=self.text_encoder_lang
        )

        audio_emb = self.s2vec_model.predict(
            input=input_audios,
            batch_size=1,
            n_parallel=1,
            pad_idx=0,
            n_prefetched_batches=1,
        )
        # pdist = self.pdist(text_emb, audio_emb).numpy().squeeze().astype(float).tolist()
        pdist = (
            distance.cdist(text_emb.numpy().astype(float), audio_emb.numpy().astype(float), 'sqeuclidean')
            .squeeze()
            .tolist()
        )
        return pdist

    def process_batch(self):
        manifest, dict_list = load_manifest(
            Path(self.input_manifest_file), keys=[self.input_audio_field, self.input_text_field]
        )
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
    Count BLEU Score.

    Args:
        ref_field (str): field with reference texts
        hyp_field (str): field with hypotheses
        output_field (str): field to save BLEU Score
    Returns:
        All the same fields as in the input manifest plus output_field.
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

        res = self.scorer.sentence_score(hypothesis=hyp, references=[ref])
        data_entry[self.output_field] = res.score
        return [DataEntry(data=data_entry)]


class Subprocess(BaseProcessor):
    """
    Processor for handling subprocess execution with additional features for managing input and output manifests.

    Args:
        cmd (str): The command to be executed as a subprocess.
        input_manifest_arg (str, optional): The argument specifying the input manifest. Defaults to an empty string.
        output_manifest_arg (str, optional): The argument specifying the output manifest. Defaults to an empty string.
        arg_separator (str, optional): The separator used between argument and value. Defaults to "=".
        **kwargs: Additional keyword arguments to be passed to the base class.

    Example:
        
        _target_: sdp.processors.datasets.commoncrawl.Subprocess
        output_manifest_file: /workspace/manifest.json
        input_manifest_arg: "--manifest"
        output_manifest_arg: "--output_filename"
        arg_separator: "="
        cmd: "python /workspace/NeMo-text-processing/nemo_text_processing/text_normalization/normalize_with_audio.py \
            --language=en --n_jobs=-1 --batch_size=600 --manifest_text_field=text --cache_dir=${workspace_dir}/cache --overwrite_cache \
            --whitelist=/workspace/NeMo-text-processing/nemo_text_processing/text_normalization/en/data/whitelist/asr_with_pc.tsv"

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
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        if self.cmd.find(self.input_manifest_file) != -1 or self.cmd.find(self.output_manifest_file) != -1:
            logger.error(
                "input_manifest_file "
                + self.input_manifest_file
                + " and output_manifest_file "
                + self.output_manifest_file
                + " should be exluded from cmd line!"
            )
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
    """
    A class for executing Neural Machine Translation (NMT) subprocess with enhanced functionality for managing input and output fields.

    Parameters:
        input_field (str): The field in the input manifest containing the source text for translation.
        output_field (str): The field to store the translated output in the output manifest.
        srctext_file (str): The file path to store the source text for translation.
        tgtout_file (str): The file path to store the translated output.
        **kwargs: Additional keyword arguments to be passed to the base class `Subprocess`.

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
        self.cmd = (
            self.cmd
            + " --srctext"
            + self.arg_separator
            + self.srctext_file
            + " --tgtout"
            + self.arg_separator
            + self.tgtout_file
        )

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


class AlignerSubprocess(Subprocess):
    """
    A class for aligning audio transcripts using an aligner subprocess with additional features for managing output fields.

    Parameters:
        output_field (str): The field in the output manifest to store the aligned transcripts.
        duration_threshold (int, optional): The maximum duration threshold for audio files in seconds. Files exceeding this threshold are excluded from alignment. Defaults to 5000.
        **kwargs: Additional keyword arguments to be passed to the base class `Subprocess`.

    """

    def __init__(
        self,
        output_field: str,
        duration_threshold: int = 5000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_field = output_field
        self.duration_threshold = duration_threshold

    def process(self):
        df1 = read_jsonl(self.input_manifest_file)
        pattern = re.compile("\s{2,}")
        df1["text"] = df1["text"].apply(lambda x: pattern.sub(" ", x).strip())
        df1["source"] = df1["audio_filepath"].apply(lambda x: x.split("/")[-2])

        df2 = pd.DataFrame(
            df1.groupby("source_audio").apply(lambda in_df: "|".join(in_df["text"].tolist())), columns=["text"]
        ).reset_index()
        df2['audio_filepath'] = df2['source_audio']
        df2['duration'] = df2['audio_filepath'].apply(audio_duration)
        df2 = df2[df2['duration'] < self.duration_threshold]

        self.input_manifest_file = os.path.join(os.path.split(self.input_manifest_file)[0], 'tmp.json')
        write_jsonl(df2[['audio_filepath', 'text']], self.input_manifest_file)

        super().process()
        manifest_path, manifest_name = os.path.split(self.input_manifest_file)
        manifest_name = os.path.splitext(manifest_name)[0]
        aligner_path = os.path.join(manifest_path, manifest_name + "_with_output_file_paths.json")
        df3 = read_jsonl(aligner_path)
        pattern = re.compile("<space>")
        df4 = pd.DataFrame()

        for ctm_filepath in tqdm(df3["segments_level_ctm_filepath"]):
            source = os.path.splitext(ctm_filepath)[0].split('/')[-1]
            df6 = df1[df1["source"] == source].reset_index()
            df5 = pd.read_csv(ctm_filepath, sep=' ', header=None, dtype={0: str})
            df5["text"] = df5[4].apply(lambda x: pattern.sub(" ", x))
            df5["nfa_start"] = df5[2]
            df5["nfa_duration"] = df5[3]
            if df5.shape[0] == df6.shape[0]:
                df7 = df5[["nfa_start", "nfa_duration", "text"]].merge(df6, how="right")
            else:
                raise ValueError(ctm_filepath)

            df4 = pd.concat([df4, df7])

        write_jsonl(df4, self.output_manifest_file)


class PreserveByValue(BaseParallelProcessor):
    """
    A class for preserving dataset entries based on a specified condition involving a target value and an input field.

    Parameters:
        input_field (str): The field in the dataset entries to be evaluated.
        target_value (Union[int, str]): The value to compare with the input field.
        operator (str, optional): The operator to apply for comparison. Options: "lt" (less than), "le" (less than or equal to),
      "eq" (equal to), "ne" (not equal to), "ge" (greater than or equal to), "gt" (greater than). Defaults to "eq".
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

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
    A class for converting language names to ISO language codes in a dataset.

    Parameters:
    - input_lang_field (str): The field in the dataset containing language names to be converted.
    - output_lang_field (str): The field to store the corresponding ISO language codes.
    - **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Attributes:
    - input_lang_field (str): The field in the dataset containing language names to be converted.
    - output_lang_field (str): The field to store the corresponding ISO language codes.
    - iso_m (dict): A mapping of language names to ISO language codes.

    Methods:
    - process_dataset_entry(data_entry): Processes a single dataset entry, converting language names to ISO language codes.

    Note:
    - This class inherits from the `BaseParallelProcessor` class and extends its functionality to perform language name to ISO code conversion.
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
        self.iso_m = {
            'English': 'en',
            'Spanish': 'es',
            'Basque': 'eu',
            'Dutch': 'nl',
            'Welsh': 'cy',
            'Italian': 'it',
            'Catalan': 'ca',
            'Maltese': 'mt',
            'Swedish': 'sv',
            'French': 'fr',
            'German': 'de',
            'Chuvash': 'cv',
            'Kinyarwanda': 'rw',
            'Polish': 'pl',
            'Kabyle': 'kab',
            'Interlingua': 'ua',
            'Portuguese': 'pt',
            'Hakha_Chin': 'cnh',
            'Romansh_Sursilvan': 'roh',
            'Breton': 'br',
            'Esperanto': 'epo',
            'Czech': 'ces',
            'Latvian': 'lav',
            'Indonesian': 'ind',
            'Slovenian': 'slv',
            'Turkish': 'tur',
            'Frisian': 'frr',
            'Tatar': 'tat',
            'Persian': 'fas',
            'Estonian': 'est',
            'Romanian': 'rum',
            'Chinese_Hongkong': 'zh',
            'Chinese_Taiwan': 'zh',
            'Chinese_China': 'zh',
            'Georgian': 'kat',
            'Kyrgyz': 'kir',
            'Dhivehi': 'div',
            'Sakha': 'sah',
            'Arabic': 'ar',
            'Japanese': 'ja',
            'Russian': 'ru',
        }

    def process_dataset_entry(self, data_entry):
        data_entry[self.output_lang_field] = self.iso_m[data_entry[self.input_lang_field]]
        return [DataEntry(data=data_entry)]


class SplitByVttSentence(BaseParallelProcessor):
    """
    A class for splitting audio files based on VTT (WebVTT) sentence-level segmentation in a dataset.

    Parameters:
    - splited_audio_dir (str): The directory to store the split audio files.
    - source_audio_field (str): The field in the dataset containing the path to the source audio files.
    - target_audio_field (str): The field to store the paths of the split audio files.
    - duration_field (str): The field to store the duration of each split audio segment.
    - text_field (str): The field to store the transcriptions corresponding to each split audio segment.
    - vtt_field (str): The field in the dataset containing the path to the VTT (WebVTT) files for segmentation.
    - proxy_fields (List[str], optional): List of additional fields to proxy from the original data entry to the split entries. Defaults to an empty list.
    - duration_threshold (float, optional): The duration threshold in seconds for each split audio segment. Defaults to 10.0.
    - **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.


    Methods:
    - prepare(): Creates the directory to store the split audio files.
    - process_dataset_entry(data_entry): Processes a single dataset entry, splitting audio based on VTT sentence-level segmentation.

    Note:
    - This class inherits from the `BaseParallelProcessor` class and extends its functionality to split audio files based on VTT segmentation.
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
                    if start_c == 0:
                        start_c = start_sr
                    else:
                        pass
                    end_c = end_sr
                    if len(text_c) > 0 and (
                        end_c - start_c > self.duration_threshold * samplerate
                        or text_c[-1] == "."
                        or text_c[-1] == "?"
                    ):
                        res_list.append(
                            self.makeDataEntry(data_entry, data, vtt_file, samplerate, text_c, start_c, end_c)
                        )
                        text_c = ''
                        start_c, end_c = 0, 0
                    else:
                        pass
                if len(text_c) > 0 and start_c != 0:
                    res_list.append(self.makeDataEntry(data_entry, data, vtt_file, samplerate, text_c, start_c, end_c))

        return res_list

    def makeDataEntry(self, data_entry, data, vtt_file, samplerate, text_c, start_c, end_c):
        data_sample = data[start_c:end_c]
        wav_save_file = os.path.join(
            self.splited_audio_dir,
            '/'.join(os.path.splitext(vtt_file)[0].split('/')[-2:]),
            str(int(start_c / (samplerate / 1000))) + "-" + str(int(end_c / (samplerate / 1000))) + ".wav",
        )
        if not os.path.isfile(wav_save_file):
            os.makedirs(os.path.split(wav_save_file)[0], exist_ok=True)
            sf.write(wav_save_file, data_sample, samplerate)

        data = {
            self.target_audio_field: wav_save_file,
            self.duration_field: data_sample.shape[0] / samplerate,
            self.text_field: text_c.strip(),
        }
        for proxy_field in self.proxy_fields:
            data[proxy_field] = data_entry[proxy_field]
        return DataEntry(data=data)


class SplitByVtt(BaseParallelProcessor):
    """
    A class for splitting audio files based on VTT (WebVTT) segmentation in a dataset.

    Parameters:
    - splited_audio_dir (str): The directory to store the split audio files.
    - source_audio_field (str): The field in the dataset containing the path to the source audio files.
    - text_lang_field (str): The field in the dataset containing the language information of the text.
    - audio_lang_field (str): The field in the dataset containing the language information of the audio.
    - key_field (str): The field in the dataset containing a unique key for each entry.
    - target_audio_field (str): The field to store the paths of the split audio files.
    - duration_field (str): The field to store the duration of each split audio segment.
    - text_field (str): The field to store the transcriptions corresponding to each split audio segment.
    - vtt_field (str): The field in the dataset containing the path to the VTT (WebVTT) files for segmentation.
    - **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Methods:
    - prepare(): Creates the directory to store the split audio files.
    - process_dataset_entry(data_entry): Processes a single dataset entry, splitting audio based on VTT segmentation.

    Note:
    - This class inherits from the `BaseParallelProcessor` class and extends its functionality to split audio files based on VTT segmentation.
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
                    res_list.append(
                        DataEntry(
                            data={
                                self.target_audio_field: wav,
                                self.duration_field: dur,
                                self.text_field: text,
                                self.audio_lang_field: data_entry[self.audio_lang_field],
                                self.text_lang_field: data_entry[self.text_lang_field],
                                self.key_field: key,
                            }
                        )
                    )
        return res_list


class AudioLid(BaseProcessor):
    """
    A class for language identification (LID) of audio files using a pre-trained LID model.

    Args:
        input_audio_field (str): The field in the dataset containing the path to the audio files for language identification.
        pretrained_model (str): The name of the pre-trained ASR model for language identification.
        output_lang_field (str): The field to store the identified language for each audio file.
        device (str): The device to run the ASR model on (e.g., 'cuda', 'cpu'). If None, it automatically selects the available GPU if present; otherwise, it uses the CPU.
        segment_duration (float): Random sample duration in seconds. Delault is np.inf.
        num_segments (int): Number of segments of file to use for majority vote. Delault is 1.
        random_seed (int): Seed for generating the starting position of the segment. Delault is None.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseProcessor`.

    """

    def __init__(
        self,
        input_audio_field: str,
        pretrained_model: str,
        output_lang_field: str,
        device: str,
        segment_duration: float = np.inf,
        num_segments: int = 1,
        random_seed: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_audio_field = input_audio_field
        self.pretrained_model = pretrained_model
        self.output_lang_field = output_lang_field
        self.segment_duration = segment_duration
        self.num_segments = num_segments
        self.random_seed = random_seed
        self.device = device

    def process(self):
        import nemo.collections.asr as nemo_asr
        import torch  # importing after nemo to make sure users first install nemo, instead of torch, then nemo

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
                    lang = model.get_label(audio_file, self.segment_duration, self.num_segments)
                except Exception as e:
                    logger.warning("AudioLid " + audio_file + " " + str(e))
                    lang = None

                if lang:
                    item[self.output_lang_field] = lang
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')


class TextLid(BaseProcessor):
    """
    A class for language identification (LID) of text using a pre-trained text classification model.

    Args:
        input_text_field (str): The field in the dataset containing the text for language identification.
        pretrained_model (str): The name or path of the pre-trained text classification model for language identification.
        output_lang_field (str): The field to store the identified language for each text.
        device (str): The device to run the text classification model on (e.g., 'cuda', 'cpu'). If None, it automatically selects the available GPU if present; otherwise, it uses the CPU.
        drop_text_duplicates (bool, optional): If True, drops duplicate texts from the output manifest. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseProcessor`.

    Methods:
    - process(): Processes the language identification for each text in the dataset and saves the results in a new manifest file.

    """

    def __init__(
        self,
        input_text_field: str,
        pretrained_model: str,
        output_lang_field: str,
        device: str,
        drop_text_duplicates: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_text_field = input_text_field
        self.pretrained_model = pretrained_model
        self.output_lang_field = output_lang_field
        self.device = device
        self.drop_duplicates = drop_text_duplicates

    def process(self):
        import torch  # importing after nemo to make sure users first install nemo, instead of torch, then nemo
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        text_set = set()
        with Path(self.output_manifest_file).open('w') as f:
            for item in tqdm(manifest):
                text = item[self.input_text_field]
                if self.drop_duplicates and text not in text_set:
                    text_set.add(text)
                    if text:
                        lid = text2lid(text_model, tokenizer, text)
                    else:
                        lid = None

                    if lid:
                        item[self.output_lang_field] = lid
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')


class AllVttText(BaseParallelProcessor):
    """
    A class for extracting text content from VTT (WebVTT) files and updating the manifest.

    Args:
        output_text_field (str): The field to store the extracted text content in the manifest.
        input_filepath_field (str, optional): The field in the manifest containing the path to VTT files. Defaults to "vtt_filepath".
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Methods:
        process_dataset_entry(data_entry): Processes a single dataset entry, extracts text content from the specified VTT file, and updates the manifest.

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
    A class for converting text files to WebVTT (VTT) format and updating the manifest.

    Args:
        vtt_files_dir (str): The directory where the generated VTT files will be saved.
        key_field (str): The field in the manifest representing the unique key or identifier for each entry.
        text_field (str): The field in the manifest containing the text content to be converted to VTT format.
        vtt_field (str): The field to store the generated VTT file paths in the manifest.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Methods:
        prepare(): Creates the directory for saving the generated VTT files.
        process_dataset_entry(data_entry): Processes a single dataset entry, converts the text content to VTT format, and updates the manifest.

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
    A class for reading information from Parquet files and updating the manifest with video URLs and captions.

    Args:
        output_video_field (str): The field to store the extracted video URLs in the manifest.
        output_caption_field (str): The field to store the extracted captions in the manifest.
        key_field (str): The field in the manifest representing the unique key or identifier for each entry.
        raw_data_dir (str): The directory containing Parquet files with information to be read.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Methods:
    - prepare(): Reads and prepares information from Parquet files, storing it in the `urls` DataFrame.
    - process_dataset_entry(data_entry): Processes a single dataset entry, extracts video URLs and captions based on the key, and updates the manifest.

    """

    def __init__(
        self,
        output_video_field: str,
        output_caption_field: str,
        key_field: str,
        raw_data_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_video_field = output_video_field
        self.output_caption_field = output_caption_field
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
            data_entry[self.output_caption_field] = self.urls.loc[key]['caption']
        except:
            data_entry[self.output_video_field] = "NN"
            data_entry[self.output_caption_field] = "NN"
            logger.warning("Key without URL or caption: " + key)
        return [DataEntry(data=data_entry)]


def get_key(x):
    key = "/".join(os.path.splitext(x)[0].split("/")[-2:])
    return key


class CreateInitialManifestCC(BaseParallelProcessor):
    """
    A class for creating an initial dataset manifest from image and text files with common keys.

    Args:
        raw_data_dir (str): The directory containing image and text files to include in the initial dataset manifest.
        video_field (str): The field to store the paths to the image files in the dataset.
        key_field (str): The field to represent the common key or identifier for each entry.
        text_field (str): The field to store the paths to the text files in the dataset.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Methods:
        prepare(): Creates the directory for saving the initial dataset manifest.
        read_manifest(): Reads the image and text files, extracts common keys, and creates a DataFrame with video, key, and text fields.
        process_dataset_entry(data_entry): Processes a single dataset entry, creating a DataEntry object with video, key, and text fields, and updates the dataset.

    """

    def __init__(
        self,
        raw_data_dir: str,
        video_field: str,
        key_field: str,
        text_field: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.video_field = video_field
        self.key_field = key_field
        self.text_field = text_field

    def prepare(self):
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def read_manifest(self):
        videos = [str(self.raw_data_dir / video) for video in self.raw_data_dir.rglob('*.jpg')]
        texts = [str(self.raw_data_dir / text) for text in self.raw_data_dir.rglob('*.txt')]
        v_df = pd.DataFrame({self.video_field: videos})
        t_df = pd.DataFrame({self.text_field: texts})

        v_df[self.key_field] = v_df[self.video_field].apply(get_key)
        t_df[self.key_field] = t_df[self.text_field].apply(get_key)
        v_df = v_df.drop_duplicates(self.key_field)
        t_df = t_df.drop_duplicates(self.key_field)
        vt_df = v_df.merge(t_df, on=self.key_field, how="left")
        return vt_df.values

    def process_dataset_entry(self, data_entry):
        (video, key, text) = data_entry

        data = {self.video_field: video, self.key_field: key, self.text_field: text}
        return [DataEntry(data=data)]


class FfmpegConvert(BaseParallelProcessor):
    """
    A class for converting video files to audio using FFmpeg and updating the dataset with the path to the resampled audio.

    Args:
        resampled_audio_dir (str): The directory to store the resampled audio files.
        input_field (str): The field in the dataset representing the path to the input video files.
        output_field (str): The field to store the path to the resampled audio files in the dataset.
        key_field (str): The field in the dataset representing the unique key or identifier for each entry.
        target_samplerate (int, optional): The target sampling rate for the resampled audio. Defaults to 16000.
        target_nchannels (int, optional): The target number of channels for the resampled audio. Defaults to 1.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Methods:
        process_dataset_entry(data_entry): Processes a single dataset entry, converts the input video to resampled audio, and updates the dataset.

    """

    def __init__(
        self,
        resampled_audio_dir: str,
        input_field: str,
        output_field: str,
        key_field: str = None,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        self.key_field = key_field
        self.resampled_audio_dir = resampled_audio_dir
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def prepare(self):
        os.makedirs(self.resampled_audio_dir, exist_ok=True)
        return super().prepare()

    def process_dataset_entry(self, data_entry):
        input_file = data_entry[self.input_field]
        if self.key_field:
            key = data_entry[self.key_field]
            os.makedirs(os.path.join(self.resampled_audio_dir, key.split("/")[0]), exist_ok=True)
        else:
            key = os.path.splitext(input_file)[0].split("/")[-1]
        audio = os.path.join(self.resampled_audio_dir, key) + ".wav"

        if not os.path.isfile(audio):
            ffmpeg_convert(input_file, audio, self.target_samplerate, self.target_nchannels)

        data_entry[self.output_field] = audio
        if self.key_field:
            data_entry[self.key_field] = key
        return [DataEntry(data=data_entry)]


class CreateInitialManifestExt(BaseParallelProcessor):
    """
    A class for creating an initial dataset manifest from audio files with a specified extension.

    Args:
        raw_data_dir (str): The directory containing audio files to include in the initial dataset manifest.
        output_field (str, optional): The field to store the audio file paths in the dataset. Defaults to "audio_filepath".
        extention (str, optional): The file extension of the audio files to include in the manifest. Defaults to "mp3".
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Methods:
        prepare(): Creates the directory for saving the initial dataset manifest.
        read_manifest(): Reads the audio files with the specified extension and creates a DataFrame with the specified output field.
        process_dataset_entry(data_entry): Processes a single dataset entry, creating a DataEntry object with the audio file path, and updates the dataset.

    """

    def __init__(
        self,
        raw_data_dir: str,
        output_field: str = "audio_filepath",
        extention: str = "mp3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.output_field = output_field
        self.extention = extention

    def prepare(self):
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def read_manifest(self):
        input_files = [str(self.raw_data_dir / video) for video in self.raw_data_dir.rglob('*.' + self.extention)]
        v_df = pd.DataFrame({self.output_field: input_files})
        return v_df.values

    def process_dataset_entry(self, data_entry):
        (inputf) = data_entry

        data = {self.output_field: inputf[0]}
        return [DataEntry(data=data)]
