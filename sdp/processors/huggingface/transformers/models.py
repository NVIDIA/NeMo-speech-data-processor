import yaml
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor

class AutoModelForCausalLMProcessor(BaseProcessor):
    def __init__(self, 
                input_manifest_file: str, 
                output_manifest_file: str,
                prompt_file: str,
                output_field: str = 'generation',
                **kwargs):
                super().__init__(
                    input_manifest_file=input_manifest_file,
                    output_manifest_file=output_manifest_file,
                    )
                
                self.prompt_file = prompt_file
                self.prompt = None
                
                self.cfg = kwargs['model']
                self.model_cfg = AutoConfig.from_pretrained(**self.cfg)

                self.output_field = output_field
    
    def read_prompt_file(self):
        with open(self.prompt_file, 'r') as prompt: 
            self.prompt = yaml.safe_load(prompt)

    def build_entry_prompt(self, data_entry):
        entry_prompt = []
        for role in self.prompt:
            entry_prompt.append(dict(
                role=role,
                content=self.prompt[role].format(**data_entry)
            ))
        return entry_prompt

    def process(self):
        logger.info(f'Reading prompt: ')
        self.read_prompt_file()
        logger.info(f'Prompt:\n{yaml.dump(self.prompt, default_flow_style=False)}\n')

        logger.info(f'Loading model:')
        model = AutoModelForCausalLM.from_config(self.model_cfg)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path)
        
        with open(self.input_manifest_file, 'r', encoding='utf8') as fin, open(self.output_manifest_file, 'w', encoding='utf8') as fout:
            for line in tqdm(fin, desc = "Generation: "):
                data_entry = json.loads(line)
                entry_prompt = self.build_entry_prompt(data_entry)
                text = tokenizer.apply_chat_template(
                    entry_prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )

                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512
                )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                data_entry[self.output_field] = response
                line = json.dumps(data_entry)
                fout.writelines(f'{line}\n')