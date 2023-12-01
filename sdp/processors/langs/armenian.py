import os
import pandas as pd
from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor, DataEntry
from sdp.processors.modify_manifest.common import load_manifest


class GetSource(BaseParallelProcessor):
    """
    A class for extracting source information from file paths and updating the dataset.

    Args:
    - input_field (str): The field containing the file path in the dataset.
    - output_field (str): The field to store the extracted source information in the dataset.
    - **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Methods:
    - process_dataset_entry(data_entry): Processes a single dataset entry, extracts source information, and updates the dataset.

    Note:
    - This class inherits from the `BaseParallelProcessor` class and extends its functionality to extract source information from file paths and update the dataset.
    """
    def __init__(
        self,
        input_field: str,
        output_field: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.output_field = output_field

    def process_dataset_entry(self, data_entry):
        input_values = os.path.splitext(data_entry[self.input_field])[0].split("/")
        
        data_entry[self.output_field] = input_values[-1]# + ", " +input_values[-2]
        return [DataEntry(data=data_entry)]


class MakeTsv(BaseProcessor):
    """
    A class for converting a JSON manifest file to a TSV (Tab-Separated Values) file.

    Args:
    - **kwargs: Additional keyword arguments to be passed to the base class `BaseProcessor`.

    Methods:
    - process(): Reads the input JSON manifest file, converts it to a DataFrame, and saves it as a TSV file.

    Note:
    - This class inherits from the `BaseProcessor` class and provides functionality to convert a JSON manifest file to a TSV file.
    """
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def process(self):
        df1 = pd.DataFrame.from_records(load_manifest(self.input_manifest_file))
        df1.to_csv(self.output_manifest_file, index=None, sep='\t')

class RandomTsvPart(BaseProcessor):
    """
    A class for creating a random subset of a TSV (Tab-Separated Values) file based on the specified fraction.

    Args:
    - part (float): The fraction of the dataset to include in the random subset, should be in the range (0.0, 1.0).
    - random_state (int): Seed for reproducibility when generating the random subset.
    - **kwargs: Additional keyword arguments to be passed to the base class `BaseProcessor`.

    Methods:
    - process(): Reads the input TSV manifest file, creates a random subset based on the specified fraction, and saves it as a new TSV file.

    Note:
    - This class inherits from the `BaseProcessor` class and provides functionality to create a random subset of a TSV file.
    """
    def __init__(
        self,
        part: float,
        random_state: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.part = part
        self.random_state = random_state

    def process(self):
        df1 = pd.read_csv(self.input_manifest_file, sep='\t')
        df1.sample(frac=self.part, random_state = self.random_state).to_csv(self.output_manifest_file, index=None, sep='\t')