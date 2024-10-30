from sdp.processors.base_processor import BaseProcessor
from sdp.data_units.data_entry import DataEntry
from sdp.data_units.stream import Stream
from sdp.data_units.manifest import Manifest


class ManifestToStream(BaseProcessor):
    def __init__(self, 
                 output: Stream,
                 input: Manifest):
        
        super().__init__(output = output,
                         input = input)

    def process(self):        
        data = [DataEntry(data_entry) for data_entry in self.input.read()]
        self.output.write(data)

    def test(self):
        assert type(self.input) is Manifest, ""
        assert type(self.output) is Stream, ""


class StreamToManifest(BaseProcessor):
    def __init__(self, 
                 output: Manifest,
                 input: Stream):
        
        super().__init__(output = output,
                         input = input)
    
    def process(self):
        data = [DataEntry(data) for data in self.input.read()[-1]]
        self.output.write(data)

    def test(self):
        assert type(self.input) is Stream, ""
        assert type(self.output) is Manifest, ""    