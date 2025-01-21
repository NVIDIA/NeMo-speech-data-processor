from sdp.processors.base_processor import BaseProcessor
import json


class ManifestToUtf8(BaseProcessor):
    """
    Processor to convert manifest file to UTF-8 encoding.
    """

    def process(self):
        with open(self.output_manifest_file, "w") as wout, open(self.input_manifest_file) as win:
            for line in win:
                print(json.dumps(json.loads(line), ensure_ascii=False), file=wout)