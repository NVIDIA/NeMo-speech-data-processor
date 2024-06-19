# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from setuptools import find_packages, setup

setup(
    name="sdp",
    version="0.1.0",
    description="NeMo-Speech-Data-Processor is a toolkit for processing speech data and creating speech datasets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    url="https://github.com/NVIDIA/NeMo-speech-data-processor",
    packages=find_packages(include=["sdp*"]),
    python_requires=">=3.10",
    install_requires=[
    'diff_match_patch',
    'editdistance',
    'hydra-core',
    'joblib',
    'librosa>=0.10.0', # specify >=0.10.0 so that librosa.get_duration(path=...) will work
    'numpy',
    'omegaconf',
    'pandas',
    'regex',
    'sox',
    'tqdm',
    'wget',
    'ffmpeg',
    'rarfile',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
