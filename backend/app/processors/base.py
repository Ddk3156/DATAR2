from abc import ABC, abstractmethod


class DatasetProcessor(ABC):

    @abstractmethod
    def inspect(self, dataset):
        pass

    @abstractmethod
    def sample(self, dataset, sample_size):
        pass

    @abstractmethod
    def export(self, dataset, output_path):
        pass

    @abstractmethod
    def validate(self, dataset):
        pass
