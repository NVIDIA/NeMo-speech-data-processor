import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import webvtt  # pip install webvtt-py

from sdp.logging import logger


def read_jsonl(manifest_file):
    rec = []
    with open(manifest_file, 'r') as the_file:
        for l in the_file:
            rec.append(json.loads(l))
    return pd.DataFrame.from_records(rec)


def write_jsonl(df_in, manifest_filename):
    with open(manifest_filename, 'w') as the_file:
        for i, x in enumerate(df_in.itertuples()):
            r_dict = {}
            for column in df_in.columns:
                r_dict[column] = getattr(x, column)
            l1 = json.dumps(r_dict)
            the_file.write(l1 + '\n')


def load_manifest(manifest: Path, keys: List[str] = []) -> List[Dict[str, Union[str, float]]]:
    result = []
    r_dict = dict()
    for key in keys:
        r_dict[key] = list()

    with manifest.open() as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            result.append(data)
            for key in keys:
                r_dict[key].append(data[key])
    if keys:
        return result, r_dict
    else:
        return result


def get_vtt_text(vtt_file):
    text_all = []
    if os.path.splitext(vtt_file)[1] == '.vtt':
        webvtt_i = webvtt.read
    elif os.path.splitext(vtt_file)[1] == '.srt':
        webvtt_i = webvtt.from_srt
    else:
        raise ValueError("Unsupported extention of file " + vtt_file)

    for caption in webvtt_i(vtt_file):
        text = caption.text
        if text.find("thumbnails") != -1:
            pass
        else:
            text_all.append(' '.join(text.split('\n')))
    return ' '.join(text_all)


def text2lid(text_model, tokenizer, text):
    text_langs = "Arabic, Basque, Breton, Catalan, Chinese_China, Chinese_Hongkong, Chinese_Taiwan, Chuvash, Czech, Dhivehi, Dutch, English, Esperanto, Estonian, French, Frisian, Georgian, German, Greek, Hakha_Chin, Indonesian, Interlingua, Italian, Japanese, Kabyle, Kinyarwanda, Kyrgyz, Latvian, Maltese, Mongolian, Persian, Polish, Portuguese, Romanian, Romansh_Sursilvan, Russian, Sakha, Slovenian, Spanish, Swedish, Tamil, Tatar, Turkish, Ukranian, Welsh".split(
        ", "
    )
    inputs = tokenizer(text[:512], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        text_logits = text_model(**inputs).logits
    lang_id = text_logits.argmax(1).cpu()[0].numpy()
    return text_langs[lang_id]


def parse_hours(inp):
    inp_list = inp.split(":")
    if len(inp_list) == 3 and int(inp_list[0]) >= 24:
        hours = int(inp_list[0]) % 24
        days = int(inp_list[0]) // 24
        if days < 31:
            inp = str(1 + days) + ":" + str(hours) + ":" + ":".join(inp_list[1:])
            return datetime.strptime(inp, '%d:%H:%M:%S.%f')
        else:
            months = days // 31
            days = days % 31
            inp = str(1 + months) + "/" + str(1 + days) + " " + str(hours) + ":" + ":".join(inp_list[1:])
            return datetime.strptime(inp, '%m/%d %H:%M:%S.%f')
    else:
        return datetime.strptime(inp, '%H:%M:%S.%f')


def split_by_vtt(vtt_file, wav_file, wav_save_path):
    try:
        data, samplerate = sf.read(wav_file)
        target_sr = samplerate
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        _begin = datetime.strptime('00:00:00.000', '%H:%M:%S.%f')
        rel_vtt_file = '/'.join(os.path.splitext(vtt_file)[0].split('/')[-2:])
        wav_list, text_list, dur_list = [], [], []
        for caption in webvtt.read(vtt_file):
            _start = parse_hours(caption.start)
            start = (_start - _begin).total_seconds()
            start_sr = int(start * samplerate)

            _end = parse_hours(caption.end)
            end = (_end - _begin).total_seconds()
            end_sr = int(end * samplerate)

            text = ' '.join(caption.text.split('\n'))

            wav_save_file = os.path.join(
                wav_save_path, rel_vtt_file, str(int(start * 1000)) + "-" + str(int(end * 1000)) + ".wav"
            )
            os.makedirs(os.path.split(wav_save_file)[0], exist_ok=True)

            # number_of_samples = round(len(data[start_sr:end_sr]) * float(target_sr) / samplerate)
            # if number_of_samples > 0:
            # if not os.path.exists(wav_save_file):
            # data_sample = sps.resample(data[start_sr:end_sr], number_of_samples)
            data_sample = data[start_sr:end_sr]
            sf.write(wav_save_file, data_sample, target_sr)
            text_list.append(text)
            wav_list.append(wav_save_file)
            dur_list.append(data_sample.shape[0] / samplerate)  # (_end-_start).total_seconds()
        return wav_list, text_list, dur_list
    except Exception as e:
        logger.warning(str(e) + vtt_file)
        return None, None, None


def split_by_vtt_new(vtt_file, samplerate):
    try:
        _begin = datetime.strptime('00:00:00.000', '%H:%M:%S.%f')
        text_list, start_s, end_s = [], [], []
        if os.path.splitext(vtt_file)[1] == '.vtt':
            webvtt_i = webvtt.read
        elif os.path.splitext(vtt_file)[1] == '.srt':
            webvtt_i = webvtt.from_srt
        else:
            raise ValueError("Unsupporte extention of file " + vtt_file)

        for caption in webvtt_i(vtt_file):
            text = ' '.join(caption.text.split('\n'))

            _start = parse_hours(caption.start)
            start = (_start - _begin).total_seconds()
            start_sr = int(start * samplerate)

            _end = parse_hours(caption.end)
            end = (_end - _begin).total_seconds()
            end_sr = int(end * samplerate)

            text_list.append(text.strip())
            start_s.append(start_sr)
            end_s.append(end_sr)
        return text_list, start_s, end_s
    except Exception as e:
        logger.warning(str(e) + vtt_file)
        return None, None, None


def audio_duration(fname):
    data, samplerate = sf.read(fname)
    return data.shape[0] / samplerate


def ffmpeg_convert(jpg: str, wav: str, ar: int = 0, ac: int = 1):
    process_args = ["ffmpeg", "-i", jpg, '-ac', str(ac), "-map", "0:a", "-c:a", "pcm_s16le", "-y", wav]
    # '-filter_complex', '"[0:a]amerge=inputs=4[a]"',
    if ar:
        process_args = process_args[:-1]
        process_args.extend(["-ar", str(ar), wav])
    return subprocess.run(process_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_txt(txt_file):
    with open(txt_file, "r") as f:
        text = f.read()
        return text[2:-1].replace("\\n", "\n").replace("\\r", "\r")


def translate(txt, trans_list):
    for trans in trans_list:
        txt = txt.replace(trans[0], trans[1])
    return txt


def txt2vtt(txt_file: str, vtt_file: str, trans_list: List):
    txt = read_txt(txt_file)
    if txt:
        if txt[:6] == "WEBVTT":
            pass
        else:
            txt = "WEBVTT" + txt
        #                 print(f"'{txt[:7]}''")
        vtt = translate(txt, trans_list)
        with open(vtt_file, "w") as f:
            f.write(vtt)


def make_trans_list():
    t1 = """U+0000	 	&nbsp;
    U+0001	\'	\\'
    U+0080	 	\\xc2\\x80
    U+0081	 	\\xc2\\x81
    U+0082	 	\\xc2\\x82
    U+0083	 	\\xc2\\x83
    U+0084	 	\\xc2\\x84
    U+0085	 	\\xc2\\x85
    U+0086	 	\\xc2\\x86
    U+0087	 	\\xc2\\x87
    U+0088	 	\\xc2\\x88
    U+0089	 	\\xc2\\x89
    U+008A	 	\\xc2\\x8a
    U+008B	 	\\xc2\\x8b
    U+008C	 	\\xc2\\x8c
    U+008D	 	\\xc2\\x8d
    U+008E	 	\\xc2\\x8e
    U+008F	 	\\xc2\\x8f
    U+0090	 	\\xc2\\x90
    U+0091	 	\\xc2\\x91
    U+0092	 	\\xc2\\x92
    U+0093	 	\\xc2\\x93
    U+0094	 	\\xc2\\x94
    U+0095	 	\\xc2\\x95
    U+0096	 	\\xc2\\x96
    U+0097	 	\\xc2\\x97
    U+0098	 	\\xc2\\x98
    U+0099	 	\\xc2\\x99
    U+009A	 	\\xc2\\x9a
    U+009B	 	\\xc2\\x9b
    U+009C	 	\\xc2\\x9c
    U+009D	 	\\xc2\\x9d
    U+009E	 	\\xc2\\x9e
    U+009F	 	\\xc2\\x9f
    U+00A0	 	\\xc2\\xa0
    U+00A1	¡	\\xc2\\xa1
    U+00A2	¢	\\xc2\\xa2
    U+00A3	£	\\xc2\\xa3
    U+00A4	¤	\\xc2\\xa4
    U+00A5	¥	\\xc2\\xa5
    U+00A6	¦	\\xc2\\xa6
    U+00A7	§	\\xc2\\xa7
    U+00A8	¨	\\xc2\\xa8
    U+00A9	©	\\xc2\\xa9
    U+00AA	ª	\\xc2\\xaa
    U+00AB	«	\\xc2\\xab
    U+00AC	¬	\\xc2\\xac
    U+00AD	­	\\xc2\\xad
    U+00AE	®	\\xc2\\xae
    U+00AF	¯	\\xc2\\xaf
    U+00B0	°	\\xc2\\xb0
    U+00B1	±	\\xc2\\xb1
    U+00B2	²	\\xc2\\xb2
    U+00B3	³	\\xc2\\xb3
    U+00B4	´	\\xc2\\xb4
    U+00B5	µ	\\xc2\\xb5
    U+00B6	¶	\\xc2\\xb6
    U+00B7	·	\\xc2\\xb7
    U+00B8	¸	\\xc2\\xb8
    U+00B9	¹	\\xc2\\xb9
    U+00BA	º	\\xc2\\xba
    U+00BB	»	\\xc2\\xbb
    U+00BC	¼	\\xc2\\xbc
    U+00BD	½	\\xc2\\xbd
    U+00BE	¾	\\xc2\\xbe
    U+00BF	¿	\\xc2\\xbf
    U+00C0	À	\\xc3\\x80
    U+00C1	Á	\\xc3\\x81
    U+00C2	Â	\\xc3\\x82
    U+00C3	Ã	\\xc3\\x83
    U+00C4	Ä	\\xc3\\x84
    U+00C5	Å	\\xc3\\x85
    U+00C6	Æ	\\xc3\\x86
    U+00C7	Ç	\\xc3\\x87
    U+00C8	È	\\xc3\\x88
    U+00C9	É	\\xc3\\x89
    U+00CA	Ê	\\xc3\\x8a
    U+00CB	Ë	\\xc3\\x8b
    U+00CC	Ì	\\xc3\\x8c
    U+00CD	Í	\\xc3\\x8d
    U+00CE	Î	\\xc3\\x8e
    U+00CF	Ï	\\xc3\\x8f
    U+00D0	Ð	\\xc3\\x90
    U+00D1	Ñ	\\xc3\\x91
    U+00D2	Ò	\\xc3\\x92
    U+00D3	Ó	\\xc3\\x93
    U+00D4	Ô	\\xc3\\x94
    U+00D5	Õ	\\xc3\\x95
    U+00D6	Ö	\\xc3\\x96
    U+00D7	×	\\xc3\\x97
    U+00D8	Ø	\\xc3\\x98
    U+00D9	Ù	\\xc3\\x99
    U+00DA	Ú	\\xc3\\x9a
    U+00DB	Û	\\xc3\\x9b
    U+00DC	Ü	\\xc3\\x9c
    U+00DD	Ý	\\xc3\\x9d
    U+00DE	Þ	\\xc3\\x9e
    U+00DF	ß	\\xc3\\x9f
    U+00E0	à	\\xc3\\xa0
    U+00E1	á	\\xc3\\xa1
    U+00E2	â	\\xc3\\xa2
    U+00E3	ã	\\xc3\\xa3
    U+00E4	ä	\\xc3\\xa4
    U+00E5	å	\\xc3\\xa5
    U+00E6	æ	\\xc3\\xa6
    U+00E7	ç	\\xc3\\xa7
    U+00E8	è	\\xc3\\xa8
    U+00E9	é	\\xc3\\xa9
    U+00EA	ê	\\xc3\\xaa
    U+00EB	ë	\\xc3\\xab
    U+00EC	ì	\\xc3\\xac
    U+00ED	í	\\xc3\\xad
    U+00EE	î	\\xc3\\xae
    U+00EF	ï	\\xc3\\xaf
    U+00F0	ð	\\xc3\\xb0
    U+00F1	ñ	\\xc3\\xb1
    U+00F2	ò	\\xc3\\xb2
    U+00F3	ó	\\xc3\\xb3
    U+00F4	ô	\\xc3\\xb4
    U+00F5	õ	\\xc3\\xb5
    U+00F6	ö	\\xc3\\xb6
    U+00F7	÷	\\xc3\\xb7
    U+00F8	ø	\\xc3\\xb8
    U+00F9	ù	\\xc3\\xb9
    U+00FA	ú	\\xc3\\xba
    U+00FB	û	\\xc3\\xbb
    U+00FC	ü	\\xc3\\xbc
    U+00FD	ý	\\xc3\\xbd
    U+00FE	þ	\\xc3\\xbe
    U+00FF	ÿ	\\xc3\\xbf
    U+0100	Ā	\\xc4\\x80
    U+0101	ā	\\xc4\\x81
    U+0102	Ă	\\xc4\\x82
    U+0103	ă	\\xc4\\x83
    U+0104	Ą	\\xc4\\x84
    U+0105	ą	\\xc4\\x85
    U+0106	Ć	\\xc4\\x86
    U+0107	ć	\\xc4\\x87
    U+0108	Ĉ	\\xc4\\x88
    U+0109	ĉ	\\xc4\\x89
    U+010A	Ċ	\\xc4\\x8a
    U+010B	ċ	\\xc4\\x8b
    U+010C	Č	\\xc4\\x8c
    U+010D	č	\\xc4\\x8d
    U+010E	Ď	\\xc4\\x8e
    U+010F	ď	\\xc4\\x8f
    U+0110	Đ	\\xc4\\x90
    U+0111	đ	\\xc4\\x91
    U+0112	Ē	\\xc4\\x92
    U+0113	ē	\\xc4\\x93
    U+0114	Ĕ	\\xc4\\x94
    U+0115	ĕ	\\xc4\\x95
    U+0116	Ė	\\xc4\\x96
    U+0117	ė	\\xc4\\x97
    U+0118	Ę	\\xc4\\x98
    U+0119	ę	\\xc4\\x99
    U+011A	Ě	\\xc4\\x9a
    U+011B	ě	\\xc4\\x9b
    U+011C	Ĝ	\\xc4\\x9c
    U+011D	ĝ	\\xc4\\x9d
    U+011E	Ğ	\\xc4\\x9e
    U+011F	ğ	\\xc4\\x9f
    U+0120	Ġ	\\xc4\\xa0
    U+0121	ġ	\\xc4\\xa1
    U+0122	Ģ	\\xc4\\xa2
    U+0123	ģ	\\xc4\\xa3
    U+0124	Ĥ	\\xc4\\xa4
    U+0125	ĥ	\\xc4\\xa5
    U+0126	Ħ	\\xc4\\xa6
    U+0127	ħ	\\xc4\\xa7
    U+0128	Ĩ	\\xc4\\xa8
    U+0129	ĩ	\\xc4\\xa9
    U+012A	Ī	\\xc4\\xaa
    U+012B	ī	\\xc4\\xab
    U+012C	Ĭ	\\xc4\\xac
    U+012D	ĭ	\\xc4\\xad
    U+012E	Į	\\xc4\\xae
    U+012F	į	\\xc4\\xaf
    U+0130	İ	\\xc4\\xb0
    U+0131	ı	\\xc4\\xb1
    U+0132	Ĳ	\\xc4\\xb2
    U+0133	ĳ	\\xc4\\xb3
    U+0134	Ĵ	\\xc4\\xb4
    U+0135	ĵ	\\xc4\\xb5
    U+0136	Ķ	\\xc4\\xb6
    U+0137	ķ	\\xc4\\xb7
    U+0138	ĸ	\\xc4\\xb8
    U+0139	Ĺ	\\xc4\\xb9
    U+013A	ĺ	\\xc4\\xba
    U+013B	Ļ	\\xc4\\xbb
    U+013C	ļ	\\xc4\\xbc
    U+013D	Ľ	\\xc4\\xbd
    U+013E	ľ	\\xc4\\xbe
    U+013F	Ŀ	\\xc4\\xbf
    U+0140	ŀ	\\xc5\\x80
    U+0141	Ł	\\xc5\\x81
    U+0142	ł	\\xc5\\x82
    U+0143	Ń	\\xc5\\x83
    U+0144	ń	\\xc5\\x84
    U+0145	Ņ	\\xc5\\x85
    U+0146	ņ	\\xc5\\x86
    U+0147	Ň	\\xc5\\x87
    U+0148	ň	\\xc5\\x88
    U+0149	ŉ	\\xc5\\x89
    U+014A	Ŋ	\\xc5\\x8a
    U+014B	ŋ	\\xc5\\x8b
    U+014C	Ō	\\xc5\\x8c
    U+014D	ō	\\xc5\\x8d
    U+014E	Ŏ	\\xc5\\x8e
    U+014F	ŏ	\\xc5\\x8f
    U+0150	Ő	\\xc5\\x90
    U+0151	ő	\\xc5\\x91
    U+0152	Œ	\\xc5\\x92
    U+0153	œ	\\xc5\\x93
    U+0154	Ŕ	\\xc5\\x94
    U+0155	ŕ	\\xc5\\x95
    U+0156	Ŗ	\\xc5\\x96
    U+0157	ŗ	\\xc5\\x97
    U+0158	Ř	\\xc5\\x98
    U+0159	ř	\\xc5\\x99
    U+015A	Ś	\\xc5\\x9a
    U+015B	ś	\\xc5\\x9b
    U+015C	Ŝ	\\xc5\\x9c
    U+015D	ŝ	\\xc5\\x9d
    U+015E	Ş	\\xc5\\x9e
    U+015F	ş	\\xc5\\x9f
    U+0160	Š	\\xc5\\xa0
    U+0161	š	\\xc5\\xa1
    U+0162	Ţ	\\xc5\\xa2
    U+0163	ţ	\\xc5\\xa3
    U+0164	Ť	\\xc5\\xa4
    U+0165	ť	\\xc5\\xa5
    U+0166	Ŧ	\\xc5\\xa6
    U+0167	ŧ	\\xc5\\xa7
    U+0168	Ũ	\\xc5\\xa8
    U+0169	ũ	\\xc5\\xa9
    U+016A	Ū	\\xc5\\xaa
    U+016B	ū	\\xc5\\xab
    U+016C	Ŭ	\\xc5\\xac
    U+016D	ŭ	\\xc5\\xad
    U+016E	Ů	\\xc5\\xae
    U+016F	ů	\\xc5\\xaf
    U+0170	Ű	\\xc5\\xb0
    U+0171	ű	\\xc5\\xb1
    U+0172	Ų	\\xc5\\xb2
    U+0173	ų	\\xc5\\xb3
    U+0174	Ŵ	\\xc5\\xb4
    U+0175	ŵ	\\xc5\\xb5
    U+0176	Ŷ	\\xc5\\xb6
    U+0177	ŷ	\\xc5\\xb7
    U+0178	Ÿ	\\xc5\\xb8
    U+0179	Ź	\\xc5\\xb9
    U+017A	ź	\\xc5\\xba
    U+017B	Ż	\\xc5\\xbb
    U+017C	ż	\\xc5\\xbc
    U+017D	Ž	\\xc5\\xbd
    U+017E	ž	\\xc5\\xbe
    U+017F	ſ	\\xc5\\xbf
    U+0180	ƀ	\\xc6\\x80
    U+0181	Ɓ	\\xc6\\x81
    U+0182	Ƃ	\\xc6\\x82
    U+0183	ƃ	\\xc6\\x83
    U+0184	Ƅ	\\xc6\\x84
    U+0185	ƅ	\\xc6\\x85
    U+0186	Ɔ	\\xc6\\x86
    U+0187	Ƈ	\\xc6\\x87
    U+0188	ƈ	\\xc6\\x88
    U+0189	Ɖ	\\xc6\\x89
    U+018A	Ɗ	\\xc6\\x8a
    U+018B	Ƌ	\\xc6\\x8b
    U+018C	ƌ	\\xc6\\x8c
    U+018D	ƍ	\\xc6\\x8d
    U+018E	Ǝ	\\xc6\\x8e
    U+018F	Ə	\\xc6\\x8f
    U+0190	Ɛ	\\xc6\\x90
    U+0191	Ƒ	\\xc6\\x91
    U+0192	ƒ	\\xc6\\x92
    U+0193	Ɠ	\\xc6\\x93
    U+0194	Ɣ	\\xc6\\x94
    U+0195	ƕ	\\xc6\\x95
    U+0196	Ɩ	\\xc6\\x96
    U+0197	Ɨ	\\xc6\\x97
    U+0198	Ƙ	\\xc6\\x98
    U+0199	ƙ	\\xc6\\x99
    U+019A	ƚ	\\xc6\\x9a
    U+019B	ƛ	\\xc6\\x9b
    U+019C	Ɯ	\\xc6\\x9c
    U+019D	Ɲ	\\xc6\\x9d
    U+019E	ƞ	\\xc6\\x9e
    U+019F	Ɵ	\\xc6\\x9f
    U+01A0	Ơ	\\xc6\\xa0
    U+01A1	ơ	\\xc6\\xa1
    U+01A2	Ƣ	\\xc6\\xa2
    U+01A3	ƣ	\\xc6\\xa3
    U+01A4	Ƥ	\\xc6\\xa4
    U+01A5	ƥ	\\xc6\\xa5
    U+01A6	Ʀ	\\xc6\\xa6
    U+01A7	Ƨ	\\xc6\\xa7
    U+01A8	ƨ	\\xc6\\xa8
    U+01A9	Ʃ	\\xc6\\xa9
    U+01AA	ƪ	\\xc6\\xaa
    U+01AB	ƫ	\\xc6\\xab
    U+01AC	Ƭ	\\xc6\\xac
    U+01AD	ƭ	\\xc6\\xad
    U+01AE	Ʈ	\\xc6\\xae
    U+01AF	Ư	\\xc6\\xaf
    U+01B0	ư	\\xc6\\xb0
    U+01B1	Ʊ	\\xc6\\xb1
    U+01B2	Ʋ	\\xc6\\xb2
    U+01B3	Ƴ	\\xc6\\xb3
    U+01B4	ƴ	\\xc6\\xb4
    U+01B5	Ƶ	\\xc6\\xb5
    U+01B6	ƶ	\\xc6\\xb6
    U+01B7	Ʒ	\\xc6\\xb7
    U+01B8	Ƹ	\\xc6\\xb8
    U+01B9	ƹ	\\xc6\\xb9
    U+01BA	ƺ	\\xc6\\xba
    U+01BB	ƻ	\\xc6\\xbb
    U+01BC	Ƽ	\\xc6\\xbc
    U+01BD	ƽ	\\xc6\\xbd
    U+01BE	ƾ	\\xc6\\xbe
    U+01BF	ƿ	\\xc6\\xbf
    U+01C0	ǀ	\\xc7\\x80
    U+01C1	ǁ	\\xc7\\x81
    U+01C2	ǂ	\\xc7\\x82
    U+01C3	ǃ	\\xc7\\x83
    U+01C4	Ǆ	\\xc7\\x84
    U+01C5	ǅ	\\xc7\\x85
    U+01C6	ǆ	\\xc7\\x86
    U+01C7	Ǉ	\\xc7\\x87
    U+01C8	ǈ	\\xc7\\x88
    U+01C9	ǉ	\\xc7\\x89
    U+01CA	Ǌ	\\xc7\\x8a
    U+01CB	ǋ	\\xc7\\x8b
    U+01CC	ǌ	\\xc7\\x8c
    U+01CD	Ǎ	\\xc7\\x8d
    U+01CE	ǎ	\\xc7\\x8e
    U+01CF	Ǐ	\\xc7\\x8f
    U+01D0	ǐ	\\xc7\\x90
    U+01D1	Ǒ	\\xc7\\x91
    U+01D2	ǒ	\\xc7\\x92
    U+01D3	Ǔ	\\xc7\\x93
    U+01D4	ǔ	\\xc7\\x94
    U+01D5	Ǖ	\\xc7\\x95
    U+01D6	ǖ	\\xc7\\x96
    U+01D7	Ǘ	\\xc7\\x97
    U+01D8	ǘ	\\xc7\\x98
    U+01D9	Ǚ	\\xc7\\x99
    U+01DA	ǚ	\\xc7\\x9a
    U+01DB	Ǜ	\\xc7\\x9b
    U+01DC	ǜ	\\xc7\\x9c
    U+01DD	ǝ	\\xc7\\x9d
    U+01DE	Ǟ	\\xc7\\x9e
    U+01DF	ǟ	\\xc7\\x9f
    U+01E0	Ǡ	\\xc7\\xa0
    U+01E1	ǡ	\\xc7\\xa1
    U+01E2	Ǣ	\\xc7\\xa2
    U+01E3	ǣ	\\xc7\\xa3
    U+01E4	Ǥ	\\xc7\\xa4
    U+01E5	ǥ	\\xc7\\xa5
    U+01E6	Ǧ	\\xc7\\xa6
    U+01E7	ǧ	\\xc7\\xa7
    U+01E8	Ǩ	\\xc7\\xa8
    U+01E9	ǩ	\\xc7\\xa9
    U+01EA	Ǫ	\\xc7\\xaa
    U+01EB	ǫ	\\xc7\\xab
    U+01EC	Ǭ	\\xc7\\xac
    U+01ED	ǭ	\\xc7\\xad
    U+01EE	Ǯ	\\xc7\\xae
    U+01EF	ǯ	\\xc7\\xaf
    U+01F0	ǰ	\\xc7\\xb0
    U+01F1	Ǳ	\\xc7\\xb1
    U+01F2	ǲ	\\xc7\\xb2
    U+01F3	ǳ	\\xc7\\xb3
    U+01F4	Ǵ	\\xc7\\xb4
    U+01F5	ǵ	\\xc7\\xb5
    U+01F6	Ƕ	\\xc7\\xb6
    U+01F7	Ƿ	\\xc7\\xb7
    U+01F8	Ǹ	\\xc7\\xb8
    U+01F9	ǹ	\\xc7\\xb9
    U+01FA	Ǻ	\\xc7\\xba
    U+01FB	ǻ	\\xc7\\xbb
    U+01FC	Ǽ	\\xc7\\xbc
    U+01FD	ǽ	\\xc7\\xbd
    U+01FE	Ǿ	\\xc7\\xbe
    U+01FF	ǿ	\\xc7\\xbf
    U+2000	 	\\xe2\\x80\\x80	EN QUAD
    U+2001	 	\\xe2\\x80\\x81	EM QUAD
    U+2002	 	\\xe2\\x80\\x82	EN SPACE
    U+2003	 	\\xe2\\x80\\x83	EM SPACE
    U+2004	 	\\xe2\\x80\\x84	THREE-PER-EM SPACE
    U+2005	 	\\xe2\\x80\\x85	FOUR-PER-EM SPACE
    U+2006	 	\\xe2\\x80\\x86	SIX-PER-EM SPACE
    U+2007	 	\\xe2\\x80\\x87	FIGURE SPACE
    U+2008	 	\\xe2\\x80\\x88	PUNCTUATION SPACE
    U+2009	 	\\xe2\\x80\\x89	THIN SPACE
    U+200A	 	\\xe2\\x80\\x8a	HAIR SPACE
    U+200B	​	\\xe2\\x80\\x8b	ZERO WIDTH SPACE
    U+200C	‌	\\xe2\\x80\\x8c	ZERO WIDTH NON-JOINER
    U+200D	‍	\\xe2\\x80\\x8d	ZERO WIDTH JOINER
    U+200E	‎	\\xe2\\x80\\x8e	LEFT-TO-RIGHT MARK
    U+200F	‏	\\xe2\\x80\\x8f	RIGHT-TO-LEFT MARK
    U+2010	‐	\\xe2\\x80\\x90	HYPHEN
    U+2011	‑	\\xe2\\x80\\x91	NON-BREAKING HYPHEN
    U+2012	‒	\\xe2\\x80\\x92	FIGURE DASH
    U+2013	–	\\xe2\\x80\\x93	EN DASH
    U+2014	—	\\xe2\\x80\\x94	EM DASH
    U+2015	―	\\xe2\\x80\\x95	HORIZONTAL BAR
    U+2016	‖	\\xe2\\x80\\x96	DOUBLE VERTICAL LINE
    U+2017	‗	\\xe2\\x80\\x97	DOUBLE LOW LINE
    U+2018	‘	\\xe2\\x80\\x98	LEFT SINGLE QUOTATION MARK
    U+2019	’	\\xe2\\x80\\x99	RIGHT SINGLE QUOTATION MARK
    U+201A	‚	\\xe2\\x80\\x9a	SINGLE LOW-9 QUOTATION MARK
    U+201B	‛	\\xe2\\x80\\x9b	SINGLE HIGH-REVERSED-9 QUOTATION MARK
    U+201C	“	\\xe2\\x80\\x9c	LEFT DOUBLE QUOTATION MARK
    U+201D	”	\\xe2\\x80\\x9d	RIGHT DOUBLE QUOTATION MARK
    U+201E	„	\\xe2\\x80\\x9e	DOUBLE LOW-9 QUOTATION MARK
    U+201F	‟	\\xe2\\x80\\x9f	DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    U+2020	†	\\xe2\\x80\\xa0	DAGGER
    U+2021	‡	\\xe2\\x80\\xa1	DOUBLE DAGGER
    U+2022	•	\\xe2\\x80\\xa2	BULLET
    U+2023	‣	\\xe2\\x80\\xa3	TRIANGULAR BULLET
    U+2024	․	\\xe2\\x80\\xa4	ONE DOT LEADER
    U+2025	‥	\\xe2\\x80\\xa5	TWO DOT LEADER
    U+2026	…	\\xe2\\x80\\xa6	HORIZONTAL ELLIPSIS
    U+2027	‧	\\xe2\\x80\\xa7	HYPHENATION POINT
    U+2028	 	\\xe2\\x80\\xa8	LINE SEPARATOR
    U+2029	 	\\xe2\\x80\\xa9	PARAGRAPH SEPARATOR
    U+202A	‪	\\xe2\\x80\\xaa	LEFT-TO-RIGHT EMBEDDING
    U+202B	‫	\\xe2\\x80\\xab	RIGHT-TO-LEFT EMBEDDING
    U+202C	‬	\\xe2\\x80\\xac	POP DIRECTIONAL FORMATTING
    U+202D	‭	\\xe2\\x80\\xad	LEFT-TO-RIGHT OVERRIDE
    U+202E	‮	\\xe2\\x80\\xae	RIGHT-TO-LEFT OVERRIDE
    U+202F	 	\\xe2\\x80\\xaf	NARROW NO-BREAK SPACE
    U+2030	‰	\\xe2\\x80\\xb0	PER MILLE SIGN
    U+2031	‱	\\xe2\\x80\\xb1	PER TEN THOUSAND SIGN
    U+2032	′	\\xe2\\x80\\xb2	PRIME
    U+2033	″	\\xe2\\x80\\xb3	DOUBLE PRIME
    U+2034	‴	\\xe2\\x80\\xb4	TRIPLE PRIME
    U+2035	‵	\\xe2\\x80\\xb5	REVERSED PRIME
    U+2036	‶	\\xe2\\x80\\xb6	REVERSED DOUBLE PRIME
    U+2037	‷	\\xe2\\x80\\xb7	REVERSED TRIPLE PRIME
    U+2038	‸	\\xe2\\x80\\xb8	CARET
    U+2039	‹	\\xe2\\x80\\xb9	SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    U+203A	›	\\xe2\\x80\\xba	SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    U+203B	※	\\xe2\\x80\\xbb	REFERENCE MARK
    U+203C	‼	\\xe2\\x80\\xbc	DOUBLE EXCLAMATION MARK
    U+203D	‽	\\xe2\\x80\\xbd	INTERROBANG
    U+203E	‾	\\xe2\\x80\\xbe	OVERLINE
    U+203F	‿	\\xe2\\x80\\xbf	UNDERTIE
    U+2040	⁀	\\xe2\\x81\\x80	CHARACTER TIE
    U+2041	⁁	\\xe2\\x81\\x81	CARET INSERTION POINT
    U+2042	⁂	\\xe2\\x81\\x82	ASTERISM
    U+2043	⁃	\\xe2\\x81\\x83	HYPHEN BULLET
    U+2044	⁄	\\xe2\\x81\\x84	FRACTION SLASH
    U+2045	⁅	\\xe2\\x81\\x85	LEFT SQUARE BRACKET WITH QUILL
    U+2046	⁆	\\xe2\\x81\\x86	RIGHT SQUARE BRACKET WITH QUILL
    U+2047	⁇	\\xe2\\x81\\x87	DOUBLE QUESTION MARK
    U+2048	⁈	\\xe2\\x81\\x88	QUESTION EXCLAMATION MARK
    U+2049	⁉	\\xe2\\x81\\x89	EXCLAMATION QUESTION MARK
    U+204A	⁊	\\xe2\\x81\\x8a	TIRONIAN SIGN ET
    U+204B	⁋	\\xe2\\x81\\x8b	REVERSED PILCROW SIGN
    U+204C	⁌	\\xe2\\x81\\x8c	BLACK LEFTWARDS BULLET
    U+204D	⁍	\\xe2\\x81\\x8d	BLACK RIGHTWARDS BULLET
    U+204E	⁎	\\xe2\\x81\\x8e	LOW ASTERISK
    U+204F	⁏	\\xe2\\x81\\x8f	REVERSED SEMICOLON
    U+2050	⁐	\\xe2\\x81\\x90	CLOSE UP
    U+2051	⁑	\\xe2\\x81\\x91	TWO ASTERISKS ALIGNED VERTICALLY
    U+2052	⁒	\\xe2\\x81\\x92	COMMERCIAL MINUS SIGN
    U+2053	⁓	\\xe2\\x81\\x93	SWUNG DASH
    U+2054	⁔	\\xe2\\x81\\x94	INVERTED UNDERTIE
    U+2055	⁕	\\xe2\\x81\\x95	FLOWER PUNCTUATION MARK
    U+2056	⁖	\\xe2\\x81\\x96	THREE DOT PUNCTUATION
    U+2057	⁗	\\xe2\\x81\\x97	QUADRUPLE PRIME
    U+2058	⁘	\\xe2\\x81\\x98	FOUR DOT PUNCTUATION
    U+2059	⁙	\\xe2\\x81\\x99	FIVE DOT PUNCTUATION
    U+205A	⁚	\\xe2\\x81\\x9a	TWO DOT PUNCTUATION
    U+205B	⁛	\\xe2\\x81\\x9b	FOUR DOT MARK
    U+205C	⁜	\\xe2\\x81\\x9c	DOTTED CROSS
    U+205D	⁝	\\xe2\\x81\\x9d	TRICOLON
    U+205E	⁞	\\xe2\\x81\\x9e	VERTICAL FOUR DOTS
    U+205F	 	\\xe2\\x81\\x9f	MEDIUM MATHEMATICAL SPACE
    U+2060	⁠	\\xe2\\x81\\xa0	WORD JOINER
    U+2061	⁡	\\xe2\\x81\\xa1	FUNCTION APPLICATION
    U+2062	⁢	\\xe2\\x81\\xa2	INVISIBLE TIMES
    U+2063	⁣	\\xe2\\x81\\xa3	INVISIBLE SEPARATOR
    U+2064	⁤	\\xe2\\x81\\xa4	INVISIBLE PLUS
    U+2065	⁥	\\xe2\\x81\\xa5	
    U+2066	⁦	\\xe2\\x81\\xa6	LEFT-TO-RIGHT ISOLATE
    U+2067	⁧	\\xe2\\x81\\xa7	RIGHT-TO-LEFT ISOLATE
    U+2068	⁨	\\xe2\\x81\\xa8	FIRST STRONG ISOLATE
    U+2069	⁩	\\xe2\\x81\\xa9	POP DIRECTIONAL ISOLATE
    U+206A	⁪	\\xe2\\x81\\xaa	INHIBIT SYMMETRIC SWAPPING
    U+206B	⁫	\\xe2\\x81\\xab	ACTIVATE SYMMETRIC SWAPPING
    U+206C	⁬	\\xe2\\x81\\xac	INHIBIT ARABIC FORM SHAPING
    U+206D	⁭	\\xe2\\x81\\xad	ACTIVATE ARABIC FORM SHAPING
    U+206E	⁮	\\xe2\\x81\\xae	NATIONAL DIGIT SHAPES
    U+206F	⁯	\\xe2\\x81\\xaf	NOMINAL DIGIT SHAPES
    U+2070	⁰	\\xe2\\x81\\xb0	SUPERSCRIPT ZERO
    U+2071	ⁱ	\\xe2\\x81\\xb1	SUPERSCRIPT LATIN SMALL LETTER I
    U+2072	⁲	\\xe2\\x81\\xb2	
    U+2073	⁳	\\xe2\\x81\\xb3	
    U+2074	⁴	\\xe2\\x81\\xb4	SUPERSCRIPT FOUR
    U+2075	⁵	\\xe2\\x81\\xb5	SUPERSCRIPT FIVE
    U+2076	⁶	\\xe2\\x81\\xb6	SUPERSCRIPT SIX
    U+2077	⁷	\\xe2\\x81\\xb7	SUPERSCRIPT SEVEN
    U+2078	⁸	\\xe2\\x81\\xb8	SUPERSCRIPT EIGHT
    U+2079	⁹	\\xe2\\x81\\xb9	SUPERSCRIPT NINE
    U+207A	⁺	\\xe2\\x81\\xba	SUPERSCRIPT PLUS SIGN
    U+207B	⁻	\\xe2\\x81\\xbb	SUPERSCRIPT MINUS
    U+207C	⁼	\\xe2\\x81\\xbc	SUPERSCRIPT EQUALS SIGN
    U+207D	⁽	\\xe2\\x81\\xbd	SUPERSCRIPT LEFT PARENTHESIS
    U+207E	⁾	\\xe2\\x81\\xbe	SUPERSCRIPT RIGHT PARENTHESIS
    U+207F	ⁿ	\\xe2\\x81\\xbf	SUPERSCRIPT LATIN SMALL LETTER N
    U+2580	▀	\\xe2\\x96\\x80
    U+2581	▁	\\xe2\\x96\\x81
    U+2582	▂	\\xe2\\x96\\x82
    U+2583	▃	\\xe2\\x96\\x83
    U+2584	▄	\\xe2\\x96\\x84
    U+2585	▅	\\xe2\\x96\\x85
    U+2586	▆	\\xe2\\x96\\x86
    U+2587	▇	\\xe2\\x96\\x87
    U+2588	█	\\xe2\\x96\\x88
    U+2589	▉	\\xe2\\x96\\x89
    U+258A	▊	\\xe2\\x96\\x8a
    U+258B	▋	\\xe2\\x96\\x8b
    U+258C	▌	\\xe2\\x96\\x8c
    U+258D	▍	\\xe2\\x96\\x8d
    U+258E	▎	\\xe2\\x96\\x8e
    U+258F	▏	\\xe2\\x96\\x8f
    U+2590	▐	\\xe2\\x96\\x90
    U+2591	░	\\xe2\\x96\\x91
    U+2592	▒	\\xe2\\x96\\x92
    U+2593	▓	\\xe2\\x96\\x93
    U+2594	▔	\\xe2\\x96\\x94
    U+2595	▕	\\xe2\\x96\\x95
    U+2596	▖	\\xe2\\x96\\x96
    U+2597	▗	\\xe2\\x96\\x97
    U+2598	▘	\\xe2\\x96\\x98
    U+2599	▙	\\xe2\\x96\\x99
    U+259A	▚	\\xe2\\x96\\x9a
    U+259B	▛	\\xe2\\x96\\x9b
    U+259C	▜	\\xe2\\x96\\x9c
    U+259D	▝	\\xe2\\x96\\x9d
    U+259E	▞	\\xe2\\x96\\x9e
    U+259F	▟	\\xe2\\x96\\x9f
    U+25A0	■	\\xe2\\x96\\xa0
    U+25A1	□	\\xe2\\x96\\xa1
    U+25A2	▢	\\xe2\\x96\\xa2
    U+25A3	▣	\\xe2\\x96\\xa3
    U+25A4	▤	\\xe2\\x96\\xa4
    U+25A5	▥	\\xe2\\x96\\xa5
    U+25A6	▦	\\xe2\\x96\\xa6
    U+25A7	▧	\\xe2\\x96\\xa7
    U+25A8	▨	\\xe2\\x96\\xa8
    U+25A9	▩	\\xe2\\x96\\xa9
    U+25AA	▪	\\xe2\\x96\\xaa
    U+25AB	▫	\\xe2\\x96\\xab
    U+25AC	▬	\\xe2\\x96\\xac
    U+25AD	▭	\\xe2\\x96\\xad
    U+25AE	▮	\\xe2\\x96\\xae
    U+25AF	▯	\\xe2\\x96\\xaf
    U+25B0	▰	\\xe2\\x96\\xb0
    U+25B1	▱	\\xe2\\x96\\xb1
    U+25B2	▲	\\xe2\\x96\\xb2
    U+25B3	△	\\xe2\\x96\\xb3
    U+25B4	▴	\\xe2\\x96\\xb4
    U+25B5	▵	\\xe2\\x96\\xb5
    U+25B6	▶	\\xe2\\x96\\xb6
    U+25B7	▷	\\xe2\\x96\\xb7
    U+25B8	▸	\\xe2\\x96\\xb8
    U+25B9	▹	\\xe2\\x96\\xb9
    U+25BA	►	\\xe2\\x96\\xba
    U+25BB	▻	\\xe2\\x96\\xbb
    U+25BC	▼	\\xe2\\x96\\xbc
    U+25BD	▽	\\xe2\\x96\\xbd
    U+25BE	▾	\\xe2\\x96\\xbe
    U+25BF	▿	\\xe2\\x96\\xbf
    U+25C0	◀	\\xe2\\x97\\x80
    U+25C1	◁	\\xe2\\x97\\x81
    U+25C2	◂	\\xe2\\x97\\x82
    U+25C3	◃	\\xe2\\x97\\x83
    U+25C4	◄	\\xe2\\x97\\x84
    U+25C5	◅	\\xe2\\x97\\x85
    U+25C6	◆	\\xe2\\x97\\x86
    U+25C7	◇	\\xe2\\x97\\x87
    U+25C8	◈	\\xe2\\x97\\x88
    U+25C9	◉	\\xe2\\x97\\x89
    U+25CA	◊	\\xe2\\x97\\x8a
    U+25CB	○	\\xe2\\x97\\x8b
    U+25CC	◌	\\xe2\\x97\\x8c
    U+25CD	◍	\\xe2\\x97\\x8d
    U+25CE	◎	\\xe2\\x97\\x8e
    U+25CF	●	\\xe2\\x97\\x8f
    U+25D0	◐	\\xe2\\x97\\x90
    U+25D1	◑	\\xe2\\x97\\x91
    U+25D2	◒	\\xe2\\x97\\x92
    U+25D3	◓	\\xe2\\x97\\x93
    U+25D4	◔	\\xe2\\x97\\x94
    U+25D5	◕	\\xe2\\x97\\x95
    U+25D6	◖	\\xe2\\x97\\x96
    U+25D7	◗	\\xe2\\x97\\x97
    U+25D8	◘	\\xe2\\x97\\x98
    U+25D9	◙	\\xe2\\x97\\x99
    U+25DA	◚	\\xe2\\x97\\x9a
    U+25DB	◛	\\xe2\\x97\\x9b
    U+25DC	◜	\\xe2\\x97\\x9c
    U+25DD	◝	\\xe2\\x97\\x9d
    U+25DE	◞	\\xe2\\x97\\x9e
    U+25DF	◟	\\xe2\\x97\\x9f
    U+25E0	◠	\\xe2\\x97\\xa0
    U+25E1	◡	\\xe2\\x97\\xa1
    U+25E2	◢	\\xe2\\x97\\xa2
    U+25E3	◣	\\xe2\\x97\\xa3
    U+25E4	◤	\\xe2\\x97\\xa4
    U+25E5	◥	\\xe2\\x97\\xa5
    U+25E6	◦	\\xe2\\x97\\xa6
    U+25E7	◧	\\xe2\\x97\\xa7
    U+25E8	◨	\\xe2\\x97\\xa8
    U+25E9	◩	\\xe2\\x97\\xa9
    U+25EA	◪	\\xe2\\x97\\xaa
    U+25EB	◫	\\xe2\\x97\\xab
    U+25EC	◬	\\xe2\\x97\\xac
    U+25ED	◭	\\xe2\\x97\\xad
    U+25EE	◮	\\xe2\\x97\\xae
    U+25EF	◯	\\xe2\\x97\\xaf
    U+25F0	◰	\\xe2\\x97\\xb0
    U+25F1	◱	\\xe2\\x97\\xb1
    U+25F2	◲	\\xe2\\x97\\xb2
    U+25F3	◳	\\xe2\\x97\\xb3
    U+25F4	◴	\\xe2\\x97\\xb4
    U+25F5	◵	\\xe2\\x97\\xb5
    U+25F6	◶	\\xe2\\x97\\xb6
    U+25F7	◷	\\xe2\\x97\\xb7
    U+25F8	◸	\\xe2\\x97\\xb8
    U+25F9	◹	\\xe2\\x97\\xb9
    U+25FA	◺	\\xe2\\x97\\xba
    U+25FB	◻	\\xe2\\x97\\xbb
    U+25FC	◼	\\xe2\\x97\\xbc
    U+25FD	◽	\\xe2\\x97\\xbd
    U+25FE	◾	\\xe2\\x97\\xbe
    U+25FF	◿	\\xe2\\x97\\xbf"""
    trans_list = []
    for a in t1.split('\n'):
        b = a.split("\t")
        trans_list.append((b[2], b[1]))
    return trans_list
