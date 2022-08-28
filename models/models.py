from abc import ABC, abstractmethod

class ModelBehavior(ABC):

    @abstractmethod
    def predict(self) -> None:
        pass

class ResNet(ModelBehavior):
    def predict(self,file) -> str:
        return f'ResNet Model predicting {file}'

class VGG16(ModelBehavior):
    def predict(self,file) -> str:
        return f'VGG Model predicting {file}'