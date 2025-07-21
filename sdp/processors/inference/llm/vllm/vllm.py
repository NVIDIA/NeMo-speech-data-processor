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

import yaml
import json
from tqdm import tqdm

from sdp.processors.base_processor import BaseProcessor


class vLLMInference(BaseProcessor):
    """
    A processor that performs inference using a vLLM model on entries from an input manifest.

    This class supports three prompt configuration modes:
    - a static prompt template (`prompt`)
    - a field in each entry containing the prompt (`prompt_field`)
    - a YAML file containing the prompt structure (`prompt_file`)

    The prompts are converted into chat-style input using a tokenizer chat template,
    passed to the vLLM engine for generation, and the results are written to an output manifest.

    Args:
        prompt (str, optional): A fixed prompt used for all entries.
        prompt_field (str, optional): The key in each entry that holds the prompt template.
        prompt_file (str, optional): Path to a YAML file containing the prompt structure.
        generation_field (str): Name of the output field to store generated text. Default is 'generation'.
        model (dict): Parameters to initialize the vLLM model.
        inference (dict): Sampling parameters passed to vLLM.SamplingParams.
        apply_chat_template (dict): Arguments passed to the tokenizer's `apply_chat_template` method.
        **kwargs: Passed to the BaseProcessor (includes `input_manifest_file` and `output_manifest_file`).

    Raises:
        ValueError: If zero or more than one prompt configuration methods are used simultaneously.

    Returns:
        A line-delimited JSON manifest where each entry includes the original fields
        plus a field with the generated output.

    .. note::
        For detailed parameter options, refer to the following documentation:

        - model: https://docs.vllm.ai/en/latest/api/vllm/index.html#vllm.LLM
        - inference: https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
        - apply_chat_template: https://huggingface.co/docs/transformers/main/en/chat_templating#applychattemplate

        Make sure to install `optree>=0.13.0` and `vllm` before using this processor:
            pip install "optree>=0.13.0" vllm

    """

    def __init__(self,
                 prompt: str = None,
                 prompt_field: str = None,
                 prompt_file: str = None,
                 generation_field: str = 'generation',
                 model: dict = {},
                 inference: dict = {},
                 apply_chat_template: dict = {},
                 **kwargs):

        from vllm import SamplingParams
        from transformers import AutoTokenizer

        super().__init__(**kwargs)
    
        self.prompt = prompt
        self.prompt_field = prompt_field
        self.generation_field = generation_field

        # Ensure that exactly one prompt method is used
        prompt_args_counter = sum([prompt is not None, prompt_field is not None, prompt_file is not None])
        if prompt_args_counter < 1:
            raise ValueError(f'One of `prompt`, `prompt_field` or `prompt_file` should be provided.')
        elif prompt_args_counter > 1:
            err = []
            if prompt:
                err.append(f'`prompt` ({prompt})')
            if prompt_field:
                err.append(f'`prompt_field` ({prompt_field})')
            if prompt_file:
                err.append(f'`prompt_file` ({prompt_file})')
            raise ValueError(f'Found more than one prompt values: {", ".join(err)}.')

        if prompt_file:
            self.prompt = self._read_prompt_file(prompt_file)

        self.model_params = model
        self.sampling_params = SamplingParams(**inference)
        self.chat_template_params = apply_chat_template

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_params['model'])

    def _read_prompt_file(self, prompt_filepath):
        """Read a YAML file with a chat-style prompt template."""
        with open(prompt_filepath, 'r') as prompt: 
            return yaml.safe_load(prompt)

    def get_entry_prompt(self, data_entry):
        """Format the prompt for a single data entry using the chat template."""
        entry_chat = []
        prompt = self.prompt
        if self.prompt_field:
            prompt = data_entry[self.prompt_field]

        for role in prompt:
            entry_chat.append(dict(
                role=role,
                content=prompt[role].format(**data_entry)
            ))

        entry_prompt = self.tokenizer.apply_chat_template(
            entry_chat,
            **self.chat_template_params
        )

        return entry_prompt

    def process(self):
        """Main processing function: reads entries, builds prompts, runs generation, writes results."""
        from vllm import LLM

        entries = []
        entry_prompts = []

        # Read entries and generate prompts
        with open(self.input_manifest_file, 'r', encoding='utf8') as fin:
            for line in tqdm(fin, desc = "Building prompts: "):
                data_entry = json.loads(line)
                entries.append(data_entry)
                entry_prompt = self.get_entry_prompt(data_entry)
                entry_prompts.append(entry_prompt)

        # Run vLLM inference
        llm = LLM(**self.model_params)
        outputs = llm.generate(entry_prompts, self.sampling_params)

        # Write results to output manifest
        with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
            for data_entry, output in tqdm(zip(entries, outputs), desc="Writing outputs: "):
                data_entry[self.generation_field] = output.outputs[0].text
                line = json.dumps(data_entry)
                fout.writelines(f'{line}\n')