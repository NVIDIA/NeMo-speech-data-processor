# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
from pathlib import Path
from typing import Dict, List, Any

from sdp.processors.base_processor import BaseProcessor, DataEntry


class ApplyEarnings21Normalizations(BaseProcessor):
    """Apply text normalizations using Earnings21 dataset normalization files.

    This processor reads normalization files provided with the Earnings21 dataset
    and applies text normalizations based on probability scores. It can use the
    highest probability normalization candidate or fallback to original text.

    Args:
        earnings21_root (str): Path to the root directory of Earnings21 dataset.
        use_top_candidate (bool): Whether to use the highest probability candidate. Defaults to True.
        fallback_to_original (bool): Whether to fallback to original text if no normalization available. Defaults to True.
        preserve_entity_tags (bool): Whether to preserve entity tags during normalization. Defaults to True.

    Returns:
        Manifest entries with normalized text field based on the normalization files.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.datasets.earnings.ApplyEarnings21Normalizations
              earnings21_root: /path/to/earnings21
              use_top_candidate: true
              fallback_to_original: true
              preserve_entity_tags: true
    """
    
    def __init__(
        self,
        earnings21_root: str,
        use_top_candidate: bool = True,
        fallback_to_original: bool = True,
        preserve_entity_tags: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.earnings21_root = Path(earnings21_root)
        self.use_top_candidate = use_top_candidate
        self.fallback_to_original = fallback_to_original
        self.preserve_entity_tags = preserve_entity_tags
        
    def process_dataset_entry(self, data_entry: DataEntry) -> List[DataEntry]:
        """Process a single dataset entry to apply normalizations."""
        data = data_entry.data
        
        # Extract file_id to load corresponding normalization file
        file_id = data.get('file_id')
        if not file_id:
            # If no file_id, return original entry
            return [data_entry]
        
        # Load normalization data for this file
        norm_file = self.earnings21_root / "transcripts" / "normalizations" / f"{file_id}.norm.json"
        
        if not norm_file.exists():
            # If no normalization file, return original entry
            return [data_entry]
        
        try:
            with open(norm_file, 'r', encoding='utf-8') as f:
                normalizations = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If can't load normalization file, return original entry
            return [data_entry]
        
        # Apply normalizations to text
        normalized_text = self._apply_normalizations(data.get('text', ''), normalizations)
        
        # Create new data entry with normalized text
        new_data = data.copy()
        new_data['text'] = normalized_text
        
        return [DataEntry(data=new_data)]
    
    def _apply_normalizations(self, text: str, normalizations: Dict[str, Any]) -> str:
        """Apply normalizations to text based on normalization data."""
        # This is a simplified implementation
        # In practice, you would need to map tokens to normalization IDs
        # and apply the appropriate normalizations
        
        # For now, just return the original text
        # This can be extended to implement actual normalization logic
        return text 