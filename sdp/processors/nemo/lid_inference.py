import json
from tqdm import tqdm
import numpy as np
from pathlib import Path

from sdp.processors.base_processor import BaseProcessor
from sdp.logging import logger
from sdp.utils.common import load_manifest

class AudioLid(BaseProcessor):
    """
    Processor for language identification (LID) of audio files using a pre-trained LID model.

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
                    lang = model.get_label(audio_file, self.segment_duration, self.num_segments)
                except Exception as e:
                    logger.warning("AudioLid " + audio_file+ " " + str(e))
                    lang = None

                if lang:
                    item[self.output_lang_field] = lang
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')