import json
import os
import tarfile

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class ExtractTar(BaseParallelProcessor):
    """
    Processor that computes the duration of the file in ``audio_filepath_key`` (using soundfile)
    and saves the duration in ``duration_key``. If there is an error computing the duration,
    the value at ``duration_key`` will be updated with the value -1.0.

    Args:
        audio_filepath_key (str): Key to get path to wav file.
        duration_key (str): Key to put to audio duration.
    Returns:
        All the same fields as in the input manifest plus duration_key
    """

    def __init__(
        self,
        tar_filepath_key: str,
        dir_to_extract: str,
        output_dir_key: str,
        rel_depth: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tar_filepath_key = tar_filepath_key
        self.output_dir_key = output_dir_key
        self.dir_to_extract = dir_to_extract
        self.rel_depth = rel_depth

    def prepare(self):
        os.makedirs(self.dir_to_extract, exist_ok=True)


    def process_dataset_entry(self, data_entry):
        tar_filepath = data_entry[self.tar_filepath_key]
        rel_folder = os.path.splitext("/".join(tar_filepath.split("/")[-self.rel_depth:]))[0]
        output_folder = os.path.join(self.dir_to_extract, rel_folder)
           
        if os.path.exists(tar_filepath + "_processed"):
            logger.warning(f"File {tar_filepath} processed.")
            data_entry[self.output_dir_key] = output_folder
        else:
            os.makedirs(output_folder, exist_ok=True)
            
            try:
                with tarfile.open(tar_filepath) as file:
                    file.extractall(output_folder)
                    os.rename(tar_filepath, tar_filepath+"_processed")
                    logger.info(f"Completed {output_folder}")
                    data_entry[self.output_dir_key] = output_folder
            except Exception as e:
                logger.warning(f"Error extracting {tar_filepath}: {e}")
                data_entry[self.output_dir_key] = ""

        return [DataEntry(data=data_entry)]
