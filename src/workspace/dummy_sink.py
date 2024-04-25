from src.workspace.abstract_sink import Sink

class DummySink(Sink):
    def log(self, metrics):
        ...

    def log_artifacts(self, name, type, files):
        ...

    def stop(self):
        ...