import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

from sdp.logging import logger
from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)
from sdp.utils.common import load_manifest


class ApplyLlama3(BaseProcessor):
    """
    Processor to prompt llm model from HuggingFace.

    Args:
        input_example_manifest (str): Assistent example manifest file.
        example_query_key (str): Field name that contains examples queries.
        example_response_key (str): Field name that contains examples ground truth responses.
        pretrained_model (str): Pretrained model name.
        input_text_key (str): Field name that contains input text.
        message (str): LLM command text.
        torch_dtype (str): Tensor data type. Default to "float16" (as llama3 is trained so).
        output_text_key (str): Key to save result.
    """

    def __init__(
        self,
        input_example_manifest: str = None,
        example_query_key: str = "text",
        example_response_key: str = "text_pc",
        pretrained_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        input_text_key: str = "text",
        main_promt: List[str] = [
            "Add missing punctuation marks. Don't change the words of the text. Keep the text as it is."
        ],
        torch_dtype: str = "float16",
        output_text_key: str = "text_pc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            import torch
            import transformers
        except:
            raise ImportError("Need to install transformers: pip install accelerate transformers")

        logger.warning("This is an example processor, for demonstration only. Do not use it for production purposes.")
        self.pretrained_model = pretrained_model
        self.example_query_key = example_query_key
        self.example_response_key = example_response_key
        self.input_example_manifest = input_example_manifest
        self.input_text_key = input_text_key
        self.output_text_key = output_text_key
        self.message = " ".join(main_promt)
        if torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            raise NotImplementedError(torch_dtype + " is not implemented!")

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.pretrained_model,
            model_kwargs={"torch_dtype": self.torch_dtype},
            device="cuda",
        )

        self.messages = [{"role": "system", "content": self.message}]
        if self.input_example_manifest:
            example_manifest = load_manifest(Path(self.input_example_manifest))
            for data_entry in example_manifest:
                self.messages.append({"role": "user", "content": data_entry[self.example_query_key]})
                self.messages.append({"role": "assistant", "content": data_entry[self.example_response_key]})

    def process(self):
        data_entries = load_manifest(Path(self.input_manifest_file))

        with Path(self.output_manifest_file).open("w") as f:
            for data_entry in data_entries:
                messages = self.messages.copy()
                messages.append({"role": "user", "content": data_entry[self.input_text_key]})

                prompt = self.pipeline.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                terminators = [
                    self.pipeline.tokenizer.eos_token_id,
                    self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]

                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=2 * len(data_entry[self.input_text_key]),
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )

                data_entry[self.output_text_key] = outputs[0]["generated_text"][len(prompt) :]
                f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")


class Subprocess(BaseProcessor):
    """
    Processor for handling subprocess execution with additional features for managing input and output manifests.

    Args:
        cmd (str): The command to be executed as a subprocess.
        input_manifest_arg (str, optional): The argument specifying the input manifest. Defaults to an empty string.
        output_manifest_arg (str, optional): The argument specifying the output manifest. Defaults to an empty string.
        arg_separator (str, optional): The separator used between argument and value. Defaults to "=".
        shell (bool, optional): The argument specifies whether to use shell for subprocess.run(). Defaults to False.
        dont_wait (bool, optional): The argument specifies whether to wait while the subprocess finishes. . Defaults to False.
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
        input_manifest_arg: str | None = None,
        output_manifest_arg: str | None = None,
        arg_separator: str = "=",
        shell: bool = False,
        dont_wait: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_manifest_arg = input_manifest_arg
        self.output_manifest_arg = output_manifest_arg
        self.arg_separator = arg_separator
        self.cmd = cmd
        self.shell = shell
        self.dont_wait = dont_wait

    def process(self):
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        if (
            self.input_manifest_arg is not None
            and self.cmd.find(self.input_manifest_file) != -1
            or self.output_manifest_arg is not None
            and self.cmd.find(self.output_manifest_file) != -1
        ):
            raise ValueError(
                "input_manifest_file "
                + self.input_manifest_file
                + " and output_manifest_file "
                + self.output_manifest_file
                + " should be exluded from cmd line: "
                + self.cmd
            )
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
        if self.shell:
            process_args = " ".join(process_args)
            logger.info("subprocess shell: " + process_args)

        if self.dont_wait:
            logger.warning("dont_wait flag is True, no logs captures!")
            subprocess.Popen(process_args, shell=self.shell, stdin=None, stdout=None, stderr=None, close_fds=True)
        else:
            subprocess.run(process_args, shell=self.shell)


class WriteTxtFiles(BaseParallelProcessor):
    """ """

    def __init__(
        self,
        text_key: Dict,
        audio_key: Dict,
        output_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_key = audio_key
        self.text_key = text_key
        self.output_dir = output_dir

    def prepare(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry: Dict):
        text = data_entry[self.text_key]
        audiofile_path = data_entry[self.audio_key]
        base_name = os.path.splitext(os.path.split(audiofile_path)[1])[0]
        output_name = os.path.join(self.output_dir, base_name + ".txt")
        with open(output_name, 'w') as file:
            file.write(text)
        return [DataEntry(data=data_entry)]
