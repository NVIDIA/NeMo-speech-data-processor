from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
import json

class MergeManifests(BaseParallelProcessor):
    def __init__(
        self, input_manifest_file2: str, fields_to_merge: dict, key_field: str = "audio_filepath",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_manifest_file2 = input_manifest_file2
        self.manifest2_dict = {}
        self.fields_to_merge = fields_to_merge
        self.key_field = key_field
    
    def prepare(self):
        with open(self.input_manifest_file2, 'r') as manifest:
            line = manifest.readline()
            while line:
                whole_sample = json.loads(line)
                key_value = whole_sample[self.key_field]
                sample = {}
                for field_names_dict in self.fields_to_merge:
                    curr_field_name = list(field_names_dict.keys())[0]
                    sample[curr_field_name] = whole_sample[curr_field_name]

                self.manifest2_dict[key_value] = sample
                line = manifest.readline()

    def process_dataset_entry(self, data_entry: dict):
        key_value = data_entry[self.key_field]
        for field_names_dict in self.fields_to_merge:
            curr_field_name = list(field_names_dict.keys())[0]
            new_field_name = field_names_dict[curr_field_name]
            data_entry[new_field_name] = self.manifest2_dict[key_value][curr_field_name]
        return [DataEntry(data=data_entry)]