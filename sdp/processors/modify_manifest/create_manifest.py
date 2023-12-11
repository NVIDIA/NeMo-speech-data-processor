from pathlib import Path

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

class CreateInitialManifestByExt(BaseParallelProcessor):
    """
    Processor for creating an initial dataset manifest by saving filepaths with a common extension to the field specified in output_field.

    Args:
        raw_data_dir (str): The directory containing image and text files to include in the initial dataset manifest.
        output_field (str): The field to store the paths to the files in the dataset.
        extension (str): The field stecify extention of the file in the dataset.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Methods:
        prepare(): Creates the directory for saving the initial dataset manifest.
        read_manifest(): Reads the image and text files, extracts common keys, and creates a DataFrame with video, key, and text fields.
        process_dataset_entry(data_entry): Processes a single dataset entry, creating a DataEntry object with video, key, and text fields, and updates the dataset.

    Note:
        This class inherits from the `BaseParallelProcessor` class and extends its functionality to create an initial dataset manifest from image and text files with common keys.
    """

    def __init__(
        self,
        raw_data_dir: str,
        output_field: str = "audio_filepath",
        extension: str = "mp3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.output_field = output_field
        self.extension = extension

    def read_manifest(self):
        input_files = [str(self.raw_data_dir / video) for video in \
                       self.raw_data_dir.rglob('*.' + self.extension)]
        return input_files
    
    def process_dataset_entry(self, data_entry):
        data = {self.output_field: data_entry}
        return [DataEntry(data=data)]
    