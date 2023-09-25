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

"""Run this file to generate documentation for SDP config files.

Will parse all the yaml files and include any built-in documentation in
the expected format.
"""

import yaml
import os
from pathlib import Path

ROOT_LINK = "https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs"

# let's ignore some of the configs we don't (yet) want to be exposed in the documentation
IGNORE_CONFIGS = []


def gen_docs():
    config_dir = str(Path(__file__).absolute().parents[1] / 'dataset_configs')
    config_docs_dir = str(Path(__file__).parents[0] / 'src' / 'sdp' / 'config-docs')

    for root, dirs, files in os.walk(config_dir):
        # Create corresponding directories in the destination directory
        for directory in dirs:
            source_path = os.path.join(root, directory)
            destination_path = source_path.replace(config_dir, config_docs_dir)
            os.makedirs(destination_path, exist_ok=True)

        # Copy files and change the file extensions
        for file in files:
            if file.endswith('.yaml'):
                source_path = os.path.join(root, file)
                config_path = source_path.replace(config_dir, '')[1:]  # removing leading /
                if config_path in IGNORE_CONFIGS:
                    continue
                destination_path = source_path.replace(config_dir, config_docs_dir).replace('.yaml', '.rst')
                with open(source_path, "rt", encoding="utf-8") as fin:
                    docs = yaml.safe_load(fin).get('documentation', "Documentation is not yet available.") + "\n\n"
                link = f"Config link: `dataset_configs/{config_path} <{ROOT_LINK}/{config_path}>`_"
                with open(destination_path, "wt", encoding="utf-8") as fout:
                    fout.write(docs + link)


if __name__ == '__main__':
    gen_docs()
