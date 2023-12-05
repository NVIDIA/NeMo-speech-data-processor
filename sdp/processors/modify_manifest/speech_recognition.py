import json
from tqdm import tqdm
from pathlib import Path
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest

class ASRWhisper(BaseProcessor):
    """
    Processor to transcribe using ASR Whisper model from HuggingFace.
    
    Args:
        pretrained_model (str): name of pretrained model on HuggingFace.
        output_text_field (str): field to save transcription result.
        device (str): Inference device.
        batch_size (str): Inference batch size.
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
        import torch
        import whisper # pip install -U openai-whisper

        self.pretrained_model = pretrained_model
        self.output_text_field = output_text_field
        self.device = device
        self.batch_size = batch_size
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        self.model = whisper.load_model(self.pretrained_model)
    
    def process(self):
        json_list = load_manifest(Path(self.input_manifest_file))
        
        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)
        
        with Path(self.output_manifest_file).open('w') as f:
            for item in tqdm(json_list):
                pred_text, pred_lang = self.whisper_infer(item["audio_filepath"])

                item[self.output_text_field] = pred_text
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def whisper_infer(self, audio_path):
        audio = whisper.load_audio(audio_path)

        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        mel = mel.to(self.device)

        _, probs = self.model.detect_language(mel)
        lang = max(probs, key=probs.get)
        
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)
        return result.text, lang
    
class ASRTransformer(BaseProcessor):
    """
    Processor to transcribe using ASR Transformer model from HuggingFace.
    
    Args:
        pretrained_model (str): name of pretrained model on HuggingFace.
        output_text_field (str): field to save transcription result.
        device (str): Inference device.
        batch_size (str): Inference batch size.
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
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        self.pretrained_model = pretrained_model
        self.output_text_field = output_text_field
        self.device = device
        self.batch_size = batch_size
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.pretrained_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
        self.model.to(self.device)
        
        processor = AutoProcessor.from_pretrained(self.pretrained_model)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def process(self):
        
        json_list = load_manifest(Path(self.input_manifest_file))
        
        Path(self.output_manifest_file).parent.mkdir(exist_ok=True, parents=True)
        
        with Path(self.output_manifest_file).open('w') as f:
            for item in tqdm(json_list):
                pred_text = self.pipe(item["audio_filepath"])["text"]

                item[self.output_text_field] = pred_text
                f.write(json.dumps(item, ensure_ascii=False) + '\n')