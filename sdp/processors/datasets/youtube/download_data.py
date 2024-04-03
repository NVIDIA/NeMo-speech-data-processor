import wget
import os
import json

from sdp.processors.base_processor import BaseProcessor, DataEntry


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