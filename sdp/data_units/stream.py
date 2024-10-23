from io import BytesIO

class Stream:
    def __init__(self):
        self.stream = BytesIO()
    
    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc_value, traceback):
       self.stream.seek(0)


def set_streams():
    pass