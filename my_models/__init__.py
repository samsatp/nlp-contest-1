from abc import ABC
from abc import abstractmethod

class my_models(ABC):

    @abstractmethod
    def preprocess(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplemented