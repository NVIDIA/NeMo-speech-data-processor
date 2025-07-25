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

from sdp.processors.datasets.coraa.create_initial_manifest import (
    CreateInitialManifestCORAA,
)
from sdp.processors.datasets.coraal import (
    CreateInitialManifestCORAAL,
    TrainDevTestSplitCORAAL,
)
from sdp.processors.datasets.earnings import (
    CreateInitialAudioAndManifest,
    CreateFullAudioManifestEarnings21,
    SpeakerSegmentedManifest,
    CreateSentenceSegmentedManifest,
    ApplyEarnings21Normalizations,
)
from sdp.processors.datasets.fleurs.create_initial_manifest import (
    CreateInitialManifestFleurs,
)
from sdp.processors.datasets.hifitts2.download_dataset import DownloadHiFiTTS2
from sdp.processors.datasets.hifitts2.remove_failed_chapters import (
    RemovedFailedChapters,
)
from sdp.processors.datasets.ksc2.create_initial_manifest import (
    CreateInitialManifestKSC2,
)
from sdp.processors.datasets.lhotse import LhotseImport
from sdp.processors.datasets.librispeech.create_initial_manifest import (
    CreateInitialManifestLibrispeech,
)
from sdp.processors.datasets.masc import (
    AggregateSegments,
    CreateInitialManifestMASC,
    GetCaptionFileSegments,
    RegExpVttEntries,
)
from sdp.processors.datasets.mcv.create_initial_manifest import CreateInitialManifestMCV
from sdp.processors.datasets.mediaspeech.create_initial_manifest import (
    CreateInitialManifestMediaSpeech,
)
from sdp.processors.datasets.mls.create_initial_manifest import CreateInitialManifestMLS
from sdp.processors.datasets.mls.restore_pc import RestorePCForMLS
from sdp.processors.datasets.mtedx.create_initial_manifest import (
    CreateInitialManifestMTEDX,
)
from sdp.processors.datasets.slr83.create_initial_manifest import (
    CreateInitialManifestSLR83,
    CustomDataSplitSLR83,
)
from sdp.processors.datasets.slr102.create_initial_manifest import (
    CreateInitialManifestSLR102,
)
from sdp.processors.datasets.slr140.create_initial_manifest import (
    CreateInitialManifestSLR140,
    CustomDataSplitSLR140,
)
from sdp.processors.datasets.uzbekvoice.create_initial_manifest import (
    CreateInitialManifestUzbekvoice,
)
from sdp.processors.datasets.voxpopuli.create_initial_manifest import (
    CreateInitialManifestVoxpopuli,
)
from sdp.processors.datasets.voxpopuli.normalize_from_non_pc_text import (
    NormalizeFromNonPCTextVoxpopuli,
)
from sdp.processors.datasets.ytc.create_initial_manifest import CreateInitialManifestYTC
from sdp.processors.huggingface.create_initial_manifest import (
    CreateInitialManifestHuggingFace,
)
from sdp.processors.modify_manifest.common import (
    AddConstantFields,
    ApplyInnerJoin,
    ChangeToRelativePath,
    CombineSources,
    DuplicateFields,
    KeepOnlySpecifiedFields,
    RenameFields,
    SortManifest,
    SplitOnFixedDuration,
    Subprocess,
    DropSpecifiedFields,

)
from sdp.processors.modify_manifest.create_manifest import (
    CreateCombinedManifests,
    CreateInitialManifestByExt,
)
from sdp.processors.modify_manifest.data_to_data import (
    ASRFileCheck,
    CopyManifestData,
    CountNumWords,
    ExtractFromBrackets,
    GetAudioDuration,
    GetWER,
    InsIfASRInsertion,
    InverseNormalizeText,
    MakeSentence,
    NormalizeText,
    ReadDocxLines,
    ReadTxtLines,
    SplitLineBySentence,
    SubIfASRSubstitution,
    SubMakeLowercase,
    SubRegex,
    ListToEntries,
    LambdaExpression,
    EstimateBandwidth,
)
from sdp.processors.modify_manifest.data_to_dropbool import (
    DropASRError,
    DropASRErrorBeginningEnd,
    DropDuplicates,
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
    DropRepeatedFields,
    PreserveByValue,
)
from sdp.processors.modify_manifest.make_letters_uppercase_after_period import (
    MakeLettersUppercaseAfterPeriod,
)
from sdp.processors.inference.asr.nemo.asr_inference import ASRInference
from sdp.processors.inference.asr.nemo.lid_inference import AudioLid
from sdp.processors.inference.asr.faster_whisper.faster_whisper_inference import FasterWhisperInference
from sdp.processors.inference.asr.transformers.speech_recognition import ASRTransformers
from sdp.processors.inference.asr.utils.whisper_hallucinations import DetectWhisperHallucinationFeatures
from sdp.processors.inference.asr.utils.rttm import GetRttmSegments, SplitAudioFile
from sdp.processors.inference.nlp.nemo.pc_inference import PCInference
from sdp.processors.inference.llm.vllm.vllm import vLLMInference
from sdp.processors.inference.llm.utils.qwen_cleaning import CleanQwenGeneration

from sdp.processors.manage_files.convert_audio import (
    FfmpegConvert,
    SoxConvert,
)
from sdp.processors.manage_files.extract import (
    ExtractTar,
)
from sdp.processors.manage_files.remove import (
    RemoveFiles,
)

from sdp.processors.toloka.accept_if import AcceptIfWERLess
from sdp.processors.toloka.create_pool import CreateTolokaPool
from sdp.processors.toloka.create_project import CreateTolokaProject
from sdp.processors.toloka.create_sentence_set import CreateSentenceSet
from sdp.processors.toloka.create_task_set import CreateTolokaTaskSet
from sdp.processors.toloka.download_responses import GetTolokaResults
from sdp.processors.toloka.reject_if import RejectIfBanned
