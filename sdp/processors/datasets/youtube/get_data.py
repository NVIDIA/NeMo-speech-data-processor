import wget
import os
import json
import shutil

from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor, DataEntry


class DownloadData(BaseProcessor):
    def __init__(
        self,
        url_list: list, 
        output_dir: str, 
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.urls = url_list
        self.output_dir = output_dir

    def process(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)

        with open(self.output_manifest_file, 'w') as manifest:
            for url in self.urls:
                subset_path = wget.download(url, out=self.output_dir)
                sample = {"subset_path" : subset_path}
                manifest_line = json.dumps(sample)
                manifest.writelines(f'{manifest_line}\n')


class ExtractData(BaseParallelProcessor):
    def __init__(
        self,
        remove_archive: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.remove_archive = remove_archive
    
    def process_dataset_entry(self, data_entry):
        archieve_filepath = data_entry['subset_path']
        output_dir = os.path.splitext(archieve_filepath)[0]
        os.makedirs(output_dir, exist_ok=True)
        shutil.unpack_archive(archieve_filepath, output_dir)

        if self.remove_archive:
            os.remove(archieve_filepath)
        
        data_entry["subset_path"] = os.path.join(output_dir, 'ssl')
        return [DataEntry(data=data_entry)]

