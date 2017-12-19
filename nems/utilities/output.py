class SplitOutput():
    """Custom splitter to output to both sys.stdout and StringIO.

    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)

    def flush(self):
        pass
