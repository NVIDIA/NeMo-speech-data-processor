import os

from sdp.processors.base_processor import DataEntry, BaseParallelProcessor
from sdp.processors import ListToEntries
from sdp.logging import logger

class CreateInitialManifest(ListToEntries):
    def __init__(self, **kwargs):
         super().__init__(**kwargs)
        
    def get_samples_durations(self, durations_filepath: str):
        durations = dict()
        with open(durations_filepath, 'r') as durations_txt:
            for line in durations_txt:
                yodas_id, duration = line.strip().split()
                durations[yodas_id] = float(duration)
        return durations
    
    def process_dataset_entry(self, data_entry):
        durations = self.get_samples_durations(data_entry['local_duration'])
        data_entries = super().process_dataset_entry(data_entry)

        yodas_entries = []
        for entry in data_entries:
            yodas_id = os.path.basename(entry.data['source_audio_filepath']).split('.')[0]
            entry.data['yodas_id'] = yodas_id
            if yodas_id in durations:
                entry.data['duration'] = durations[yodas_id]
                yodas_entries.append(entry)
            else:
                logger.warning(f'Skipping `{yodas_id}` because there is no duration info in metadata.')
                
        return yodas_entries


