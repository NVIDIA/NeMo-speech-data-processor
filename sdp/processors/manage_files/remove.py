import os
import shutil

from sdp.logging import logger
from sdp.processors.base_processor import DataEntry, BaseParallelProcessor

class RemoveFiles(BaseParallelProcessor):
    def __init__(self,
                filepath_field: str,
                drop_filepath_field: bool = True,
                **kwargs):

        super().__init__(**kwargs)
        self.filepath_field = filepath_field
        self.drop_filepath_field = drop_filepath_field
    
    def process_dataset_entry(self, data_entry):
        filepath = data_entry[self.filepath_field]
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)
        else:
            os.remove(filepath)
        
        if self.drop_filepath_field:
            data_entry.pop(self.filepath_field)
        
        return [DataEntry(data=data_entry)]