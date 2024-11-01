from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterable
from sdp.data_units.data_entry import DataEntry


class DataSource(ABC):
    def __init__(self, source: Any):
        self.source = source
        self.number_of_entries =  0
        self.total_duration = 0.0
        self.metrics = []

    @abstractmethod
    def read(self, *args, **kwargs) -> List[Dict]:
        pass
    
    @abstractmethod
    def write(self, data_entries: Iterable):
        for data_entry in data_entries:
            self._add_metrics(data_entry)

    def _add_metrics(self, data_entry: DataEntry):
        if data_entry.metrics is not None:
            self.metrics.append(data_entry.metrics)
        if data_entry.data is not None:
            self.total_duration += data_entry.data.get("duration", 0)
        self.number_of_entries += 1

class DataSetter(ABC):
    def __init__(self, processors_cfgs: List[Dict]):
        self.processors_cfgs = processors_cfgs
    
    def get_resolvable_link(*args):
        return f"${{{'.' + '.'.join(list(map(str, args)))}}}"

