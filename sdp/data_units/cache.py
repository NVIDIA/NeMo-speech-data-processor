import os
from tempfile import TemporaryDirectory
from uuid import uuid4

class CacheDir:
    def __init__(self, cache_dirpath: str = None, prefix: str = None, suffix: str = None):
            if cache_dirpath:
                os.makedirs(cache_dirpath, exist_ok=True)
            self.cache_dir = TemporaryDirectory(dir = cache_dirpath, prefix = prefix, suffix = suffix)
    
    def make_tmp_filepath(self):
        return os.path.join(self.cache_dir.name, str(uuid4()))

    def makedir(self, **kwargs): 
         tmp_dir = CacheDir(cache_dirpath = self.cache_dir.name, **kwargs)
         return tmp_dir
    
    def cleanup(self):
        self.cache_dir.cleanup()


CACHE_DIR = CacheDir()