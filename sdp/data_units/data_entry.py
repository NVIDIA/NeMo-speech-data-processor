from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Iterable
from abc import ABC, abstractmethod

@dataclass
class DataEntry:
    """A wrapper for data entry + any additional metrics."""

    data: Optional[Dict]  # can be None to drop the entry
    metrics: Any = None

class DataSource(ABC):
    def __init__(self, source: Any):
        self.source = source
        self.number_of_entries =  0
        self.total_duration = 0.0
        self.metrics = []

    @abstractmethod
    def read(self, **kwargs) -> List[Dict]:
        pass
    
    @abstractmethod
    def write(self, data_entries: Iterable):
        for data_entry in data_entries:
            self._add_metrics(data_entry)

    def _add_metrics(self, data_entry: DataEntry):
        self.metrics.append(data_entry.metrics)
        self.number_of_entries += 1
        self.total_duration += data_entry.data.get("duration", 0)
