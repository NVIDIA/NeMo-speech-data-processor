# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class TrainDevTestSplitCORAAL(BaseParallelProcessor):
    """Custom train-dev-test split for CORAAL dataset.

    Split is done speaker-wise, so the same speakers don't appear in different
    splits.

    Args:
        data_split (str): train, dev or test.
    """

    def __init__(
        self,
        data_split: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if data_split not in ["train", "dev", "test"]:
            raise ValueError("data_split has to be either train, dev or test")
        self.data_split = data_split
        self.split_map = {}
        self.split_map["train"] = set(
            [
                'ATL_se0_ag1_m',
                'DCA_se1_ag1_f',
                'DCA_se1_ag2_f',
                'DCA_se1_ag2_m',
                'DCA_se1_ag3_f',
                'DCA_se1_ag3_m',
                'DCA_se1_ag4_m',
                'DCA_se2_ag1_f',
                'DCA_se2_ag1_m',
                'DCA_se2_ag2_m',
                'DCB_se1_ag1_m',
                'DCB_se1_ag2_f',
                'DCB_se1_ag2_m',
                'DCB_se1_ag3_f',
                'DCB_se1_ag3_m',
                'DCB_se1_ag4_f',
                'DCB_se1_ag4_m',
                'DCB_se2_ag1_f',
                'DCB_se2_ag1_m',
                'DCB_se2_ag2_f',
                'LES_se0_ag2_f',
                'LES_se0_ag2_m',
                'PRV_se0_ag1_f',
                'PRV_se0_ag2_f',
                'ROC_se0_ag1_m',
                'ROC_se0_ag2_f',
                'VLD_se0_ag2_f',
                'VLD_se0_ag2_m',
            ]
        )
        self.split_map["dev"] = set(
            [
                'ATL_se0_ag1_f',
                'DCA_se1_ag1_m',
                'DCB_se1_ag1_f',
                'LES_se0_ag3_f',
                'PRV_se0_ag1_m',
                'ROC_se0_ag1_f',
                'VLD_se0_ag3_f',
            ]
        )
        self.split_map["test"] = set(
            [
                'ATL_se0_ag2_f',
                'ATL_se0_ag2_m',
                'DCA_se2_ag3_m',
                'DCA_se2_ag4_f',
                'DCA_se2_ag4_m',
                'DCA_se3_ag1_f',
                'DCA_se3_ag1_m',
                'DCA_se3_ag2_f',
                'DCA_se3_ag2_m',
                'DCA_se3_ag3_f',
                'DCA_se3_ag3_m',
                'DCA_se3_ag4_m',
                'DCB_se2_ag2_m',
                'DCB_se2_ag3_f',
                'DCB_se2_ag3_m',
                'DCB_se2_ag4_f',
                'DCB_se2_ag4_m',
                'DCB_se3_ag1_f',
                'DCB_se3_ag1_m',
                'DCB_se3_ag2_f',
                'DCB_se3_ag3_f',
                'DCB_se3_ag3_m',
                'DCB_se3_ag4_f',
                'DCB_se3_ag4_m',
                'LES_se0_ag3_m',
                'LES_se0_ag4_f',
                'LES_se0_ag4_m',
                'PRV_se0_ag2_m',
                'PRV_se0_ag3_f',
                'PRV_se0_ag3_m',
                'ROC_se0_ag2_m',
                'ROC_se0_ag3_f',
                'ROC_se0_ag3_m',
                'VLD_se0_ag3_m',
                'VLD_se0_ag4_f',
                'VLD_se0_ag4_m',
            ]
        )

    def process_dataset_entry(self, data_entry):
        if data_entry["original_file"][:-5] in self.split_map[self.data_split]:
            return [DataEntry(data=data_entry)]
        return []
