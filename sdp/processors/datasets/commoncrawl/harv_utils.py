import os
import torch
# import ffmpeg # pip install ffmpeg-python
import webvtt  # pip install webvtt-py
import subprocess, sys
import json, os
import soundfile as sf
from typing import Dict, List, Union
from datetime import datetime
import numpy as np
from pathlib import Path
import pandas as pd
from sdp.logging import logger


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


def split_by_vtt(vtt_file, samplerate):
    try:
        _begin = datetime.strptime('00:00:00.000', '%H:%M:%S.%f')
        text_list, start_s, end_s = [], [], []
        for caption in webvtt.read(vtt_file):
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

