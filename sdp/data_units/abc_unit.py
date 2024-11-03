from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterable
from sdp.data_units.data_entry import DataEntry


class DataSource(ABC):
    def __init__(self, source: Any):
        self.source = source
        self.number_of_entries = 0
        self.total_duration = 0.0
        self.metrics = []

    
    @abstractmethod
    def read_entry(self) -> Dict:
        pass

    @abstractmethod
    def read_entries(self, in_memory_chunksize: int = None) -> List[Dict]:
        pass

    @abstractmethod
    def write_entry(self, data_entry: DataEntry):
        pass

    @abstractmethod
    def write_entries(self, data_entries: List[DataEntry]):
        pass

    def update_metrics(self, data_entry: DataEntry):
        if data_entry.metrics is not None:
            self.metrics.append(data_entry.metrics)
        if data_entry.data is dict:
            self.total_duration += data_entry.data.get("duration", 0)
        self.number_of_entries += 1

class DataSetter(ABC):
    def __init__(self, processors_cfgs: List[Dict]):
        self.processors_cfgs = processors_cfgs
    
    def get_resolvable_link(*args):
        return f"${{{'.' + '.'.join(list(map(str, args)))}}}"