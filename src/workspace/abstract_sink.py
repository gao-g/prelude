from abc import ABC, abstractmethod

class Sink(ABC):
    @abstractmethod
    def log(self, metrics):
        ...

    @abstractmethod
    def log_artifacts(self, name, type, files):
        ...

    @abstractmethod
    def stop(self):
        ...

    def __del__(self):
        self.stop()  