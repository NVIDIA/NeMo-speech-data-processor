import tempfile
import os
import json
from pathlib import Path
from sdp.utils.import_manager import ImportManager
import pytest
from typing import Dict, List, Union, Optional

# Example YAML content with processors
#Content is right, additional {} is needed because of the format function
TEST_YAML_CONTENT = """ 
use_import_manager: True
processors_to_run: ":"  # Run all processors
workspace_dir: {workspace_dir}
processors:
  - _target_: sdp.processors.modify_manifest.common.DuplicateFields
    input_manifest_file: {workspace_dir}/test1.json
    output_manifest_file: {workspace_dir}/test2.json
    duplicate_fields: {{"text": "answer"}}

  - _target_: sdp.processors.modify_manifest.common.RenameFields
    input_manifest_file: {workspace_dir}/test2.json
    output_manifest_file: {workspace_dir}/test3.json
    rename_fields: {{"text": "text2test"}}
"""

# Example manifest content
EXAMPLE_MANIFEST = [
    {"id": 1, "text": "hello", "duration": 10, "audio_filepath": "path1"},
    {"id": 2, "text": "world", "duration": 12, "audio_filepath": "path2"}
]

def _write_manifest(file_path, content: List[Dict]):
    """json lines to a file."""
    with open(file_path, "w") as f:
        for entry in content:
            f.write(json.dumps(entry) + "\n")

def test_import_manager_with_workspace():
    """
    Test ImportManager's functionality with a workspace directory and example manifests.
    """
    with tempfile.TemporaryDirectory() as tmp_workspace:
        #workspace_dir = Path
        workspace_dir = Path(tmp_workspace)

        # Step 1: example manifest files
        test1_path = workspace_dir / "test1.json"
        test2_path = workspace_dir / "test2.json"
        test3_path = workspace_dir / "test3.json"
        _write_manifest(test1_path, EXAMPLE_MANIFEST)

        # create yaml configuration file
        yaml_content = TEST_YAML_CONTENT.format(workspace_dir=workspace_dir)
        yaml_file = workspace_dir / "config.yaml"
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        # Run ImportManager
        init_file = workspace_dir / "__init__.py"
        manager = ImportManager()
        manager.sync_with_config(yaml_config=str(yaml_file), init_file=str(init_file))

        # Verify that __init__.py contains the expected imports
        assert init_file.exists(), "__init__.py file should be created"

        with open(init_file, "r") as f:
            init_content = f.read()

        expected_imports = [
            "from sdp.processors.modify_manifest.common import DuplicateFields",
            "from sdp.processors.modify_manifest.common import RenameFields",
        ]
        for expected_import in expected_imports:
            assert expected_import in init_content, f"Expected import '{expected_import}' not found"

        # Verify that the manifests is ok
        assert test1_path.exists(), "test1.json should exist"
        assert not test2_path.exists(), "test2.json should not be overwritten yet"
        assert not test3_path.exists(), "test3.json should not be overwritten yet"

        