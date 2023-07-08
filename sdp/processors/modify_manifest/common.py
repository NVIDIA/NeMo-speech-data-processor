import json
import os
from typing import Dict, List

from tqdm import tqdm

from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)


class CombineSources(BaseParallelProcessor):
    """Can be used to create a single field from two alternative sources.

    E.g.::

        _target_: sdp.processors.CombineSources
        source1: text_pc
        source1_origin: original
        source2: text_pc_pred
        source2_origin: synthetic
        target: text

    will populate the ``text`` field with data from ``text_pc`` field if it's
    present and not equal to ``n/a`` (can be customized). Alternatively, it
    will populate ``text`` from ``text_pc_pred`` field. In both cases it will
    specify which source was used in the ``text_origin`` field by using either
    ``original`` or ``synthetic`` labels.

    Args:
        source1 (str): first source field to use by default if it's available.
        source2 (str): second source field to use if first is not available.
            Has to be present in that case.
        target (str): target field that we are populating.
        source1_origin (str): a label that will be written in the ``<target>_origin``
            field to specify that the data was populated from the first source.
            Defaults to "source1".
        source2_origin (str): a label that will be written in the ``<target>_origin``
            field to specify that the data was populated from the second source.
            Defaults to "source2".
        na_indicator (str): if the first source field has text equal to the
            ``na_indicator`` it will be considered as not available.
            Defaults to ``n/a``.

    Returns:
        The same data as in the input manifest enhanced with the following fields::

            <target>: <populated with data from either <source1> or <source2>>
            <target>_origin: <label that marks where the data came from>
    """

    def __init__(
        self,
        source1: str,
        source2: str,
        target: str,
        source1_origin: str = "source1",
        source2_origin: str = "source2",
        na_indicator: str = "n/a",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source1 = source1
        self.source2 = source2
        self.target = target
        self.source1_origin = source1_origin
        self.source2_origin = source2_origin
        self.na_indicator = na_indicator

    def process_dataset_entry(self, data_entry: Dict):
        if data_entry.get(self.source1, self.na_indicator) != self.na_indicator:
            data_entry[self.target] = data_entry[self.source1]
            data_entry[f"{self.target}_origin"] = self.source1_origin
        else:
            data_entry[self.target] = data_entry[self.source2]
            data_entry[f"{self.target}_origin"] = self.source2_origin

        return [DataEntry(data=data_entry)]


class AddConstantFields(BaseParallelProcessor):
    """This processor adds constant fields to all manifest entries.

    E.g., can be useful to add fixes "label: <language>" field for downstream
    lang-id model training.

    Args:
        fields: dictionary with any additional information to add. E.g.::

            fields = {
                "label": "en",
                "metadata": "mcv-11.0-2022-09-21"
            }
    """

    def __init__(
        self,
        fields: Dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fields = fields

    def process_dataset_entry(self, data_entry: Dict):
        data_entry.update(self.fields)
        return [DataEntry(data=data_entry)]


class DuplicateFields(BaseParallelProcessor):
    """
    This processor duplicates fields in all manifest entries.
    It is useful for when you want to do downstream processing of a variant of the entry.
    e.g. make a copy of "text" called "text_no_pc", and remove punctuation from "text_no_pc" in
    downstream processors.

    Args:
        duplicate_fields: dictionary where keys are the original fields to be copied and their values
            are the new names of the duplicate fields.
    """

    def __init__(
        self,
        duplicate_fields: Dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.duplicate_fields = duplicate_fields

    def process_dataset_entry(self, data_entry: Dict):
        for field_src, field_tgt in self.duplicate_fields.items():
            if not field_src in data_entry:
                raise ValueError(f"Expected field {field_src} in data_entry {data_entry} but there isn't one.")

            data_entry[field_tgt] = data_entry[field_src]

        return [DataEntry(data=data_entry)]


class RenameFields(BaseParallelProcessor):
    """
    This processor renames the field in all manifest entries.

    Args:
        rename_fields: dictionary where keys are the fields to be renamed and their values
            are the new names of the fields.
    """

    def __init__(
        self,
        rename_fields: Dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rename_fields = rename_fields

    def process_dataset_entry(self, data_entry: Dict):
        for field_src, field_tgt in self.rename_fields.items():
            if not field_src in data_entry:
                raise ValueError(f"Expected field {field_src} in data_entry {data_entry} but there isn't one.")

            data_entry[field_tgt] = data_entry[field_src]
            del data_entry[field_src]

        return [DataEntry(data=data_entry)]


class SplitOnFixedDuration(BaseParallelProcessor):
    """This processor splits audio into a fixed length segments.

    It does not actually create different audio files, but simply adds
    corresponding "offset" and "duration" fields.

    Args:
        segment_duration: fixed desired duraiton of each segment.
        drop_last: whether to drop the last segment if total duration is not
            divisible by desired segment duration. If False, the last segment
            will be of a different lenth which is ``< segment_duration``.
        drop_text: whether to drop text from entries as it is most likely
            inaccurate after the split on duration.
    """

    def __init__(
        self,
        segment_duration: float,
        drop_last: bool = True,
        drop_text: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.segment_duration = segment_duration
        self.drop_last = drop_last
        self.drop_text = drop_text

    def process_dataset_entry(self, data_entry: Dict):
        total_duration = data_entry["duration"]
        total_segments = int(total_duration // self.segment_duration)
        output = [None] * total_segments
        for segment_idx in range(total_segments):
            modified_entry = data_entry.copy()  # shallow copy should be good enough
            modified_entry["duration"] = self.segment_duration
            modified_entry["offset"] = segment_idx * self.segment_duration
            if self.drop_text:
                modified_entry.pop("text", None)
            output[segment_idx] = DataEntry(data=modified_entry)

        remainder = total_duration - self.segment_duration * total_segments
        if not self.drop_last and remainder > 0:
            modified_entry = data_entry.copy()
            modified_entry["duration"] = remainder
            modified_entry["offset"] = self.segment_duration * total_segments
            if self.drop_text:
                modified_entry.pop("text", None)
            output.append(DataEntry(data=modified_entry))

        return output


class ChangeToRelativePath(BaseParallelProcessor):
    """This processor changes the audio filepaths to be relative.

    Args:
        base_dir: typically a folder where manifest file is going to be
            stored. All passes will be relative to that folder.
    """

    def __init__(
        self,
        base_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_dir = base_dir

    def process_dataset_entry(self, data_entry: Dict):
        data_entry["audio_filepath"] = os.path.relpath(data_entry["audio_filepath"], self.base_dir)

        return [DataEntry(data=data_entry)]


class SortManifest(BaseProcessor):
    """
    Processor which will sort the manifest by some specified attribute.

    Args:
        output_manifest: the path to the output manifest. It will be the same as the
            input manifest, but resorted.
        input_manifest_file: the path to the input manifest which will be resorted.
        attribute_sort_by: the attribute by which the manifest will be sorted.
        descending: if set to False (default), attribute will be in ascending order.
            If True, attribute will be in descending order.

    """

    def __init__(
        self, output_manifest_file: str, input_manifest_file: str, attribute_sort_by: str, descending: bool = True
    ):
        self.output_manifest_file = output_manifest_file
        self.input_manifest_file = input_manifest_file
        self.attribute_sort_by = attribute_sort_by
        self.descending = descending

    def process(self):
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            dataset_entries = [json.loads(line) for line in fin.readlines()]

        dataset_entries = sorted(dataset_entries, key=lambda x: x[self.attribute_sort_by], reverse=self.descending)

        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for line in dataset_entries:
                fout.write(json.dumps(line) + "\n")


class WriteManifest(BaseProcessor):
    """Saves a copy of a manifest but only with a subset of the fields.

    Args:
        fields_to_save (list[str]): list of the fields in the input manifest
            that we want to retain. The output file will only contain these
            fields.
    """

    def __init__(self, fields_to_save: List[str], **kwargs):
        super().__init__(**kwargs)
        self.fields_to_save = fields_to_save

    def process(self):
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin, open(
            self.output_manifest_file, "wt", encoding="utf8"
        ) as fout:
            for line in tqdm(fin):
                line = json.loads(line)
                new_line = {field: line[field] for field in self.fields_to_save}
                fout.write(json.dumps(new_line) + "\n")
