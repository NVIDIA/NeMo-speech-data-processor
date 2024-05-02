# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# let's import all supported processors here to simplify target specification

from sdp.processors.datasets.coraal import (
    CreateInitialManifestCORAAL,
    TrainDevTestSplitCORAAL,
)

from sdp.processors.datasets.fleurs.create_initial_manifest import (
    CreateInitialManifestFleurs,
)
from sdp.processors.datasets.lhotse import LhotseImport

from sdp.processors.datasets.librispeech.create_initial_manifest import (
    CreateInitialManifestLibrispeech,
)
from sdp.processors.datasets.mcv.create_initial_manifest import CreateInitialManifestMCV
from sdp.processors.datasets.mls.create_initial_manifest import CreateInitialManifestMLS
from sdp.processors.datasets.mls.restore_pc import RestorePCForMLS
from sdp.processors.datasets.slr83.create_initial_manifest import (
    CreateInitialManifestSLR83,
    CustomDataSplitSLR83,
)
from sdp.processors.datasets.voxpopuli.create_initial_manifest import (
    CreateInitialManifestVoxpopuli,
)
from sdp.processors.datasets.voxpopuli.normalize_from_non_pc_text import (
    NormalizeFromNonPCTextVoxpopuli,
)
from sdp.processors.huggingface.speech_recognition import ASRTransformers, ASRWhisper
from sdp.processors.modify_manifest.common import (
    AddConstantFields,
    ChangeToRelativePath,
    CombineSources,
    DuplicateFields,
    KeepOnlySpecifiedFields,
    RenameFields,
    SortManifest,
    SplitOnFixedDuration,
)
from sdp.processors.modify_manifest.create_manifest import CreateInitialManifestByExt
from sdp.processors.modify_manifest.data_to_data import (
    CountNumWords,
    FfmpegConvert,
    GetAudioDuration,
    InsIfASRInsertion,
    ReadTxtLines,
    SoxConvert,
    SplitLineBySentence,
    SubIfASRSubstitution,
    SubMakeLowercase,
    SubRegex,
)
from sdp.processors.modify_manifest.data_to_dropbool import (
    DropASRError,
    DropASRErrorBeginningEnd,
    DropHighCER,
    DropHighLowCharrate,
    DropHighLowDuration,
    DropHighLowWordrate,
    DropHighWER,
    DropIfNoneOfRegexMatch,
    DropIfRegexMatch,
    DropIfSubstringInInsertion,
    DropLowWordMatchRate,
    DropNonAlphabet,
    DropOnAttribute,
    PreserveByValue,
)
from sdp.processors.modify_manifest.make_letters_uppercase_after_period import (
    MakeLettersUppercaseAfterPeriod,
)
from sdp.processors.nemo.asr_inference import ASRInference
from sdp.processors.nemo.pc_inference import PCInference
