import os

from sdp.processors.base_processor import DataEntry, BaseParallelProcessor
from sdp.processors import ListToEntries

class CreateInitialManifest(ListToEntries):
    def __init__(self, **kwargs):
         super().__init__(**kwargs)
        
    def process_dataset_entry(self, data_entry):
        data_entries = super().process_dataset_entry(data_entry)
        for entry in data_entries:
            entry.data['yodas_id'] = os.path.basename(entry.data['source_audio_filepath']).split('.')[0]
        
        return data_entries


