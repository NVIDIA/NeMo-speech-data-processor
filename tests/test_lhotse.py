import json
from pathlib import Path

import pytest

lhotse = pytest.importorskip(
    "lhotse", reason="Lhotse import tests require lhotse to be installed."
)

import torchaudio
from lhotse.testing.dummies import DummyManifest
from lhotse import CutSet

from sdp.processors.datasets.lhotse import LhotseImport


@pytest.fixture
def cuts_path(tmp_path):
    """
    Create tmpdir with audio data referenced by a CutSet
    (two 1s utterances with text, speaker, gender, and language values of 'irrelevant').
    """
    p = tmp_path / "cuts.jsonl.gz"

    def drop_custom(c):
        c.custom = None
        return c

    (
        DummyManifest(CutSet, begin_id=0, end_id=2, with_data=True)
        .map(drop_custom)
        .save_audios(tmp_path / "audios")
        .to_file(p)
    )

    return p


def test_lhotse_import(tmp_path, cuts_path):
    out_path = tmp_path / "nemo_manifest.json"

    processor = LhotseImport(
        input_manifest_file=cuts_path, output_manifest_file=out_path
    )
    processor.process()

    EXPECTED_KEYS = {
        "audio_filepath",
        "text",
        "duration",
        "speaker",
        "gender",
        "language",
    }

    data = [json.loads(line) for line in out_path.open()]
    assert len(data) == 2

    for item in data:
        assert set(item.keys()) == EXPECTED_KEYS
        assert item["duration"] == 1.0
        audio, sr = torchaudio.load(item["audio_filepath"])
        assert audio.shape == (1, 16000)
        for key in ("text", "speaker", "gender", "language"):
            assert item[key] == "irrelevant"
