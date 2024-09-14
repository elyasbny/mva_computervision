"""Python file to instantite the model and the transform that goes with it."""
from model import Net, ResNet, ResidualBlock, resnet_p, GoogLeNet
from data import data_transforms
from efficientnet_pytorch import EfficientNet
from torchvision import models


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet_p":
            return resnet_p()
        # Not used
        elif self.model_name == "efficientnet-b0":
            model = EfficientNet.from_pretrained('efficientnet-b0')
            return model
        elif self.model_name == "efficientnet-b7":
            model = models.efficientnet_b7(pretrained=True)
            #model = EfficientNet.from_pretrained('efficientnet-b5')
            return model
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        elif self.model_name == "resnet_p":
            return data_transforms
        elif self.model_name == "efficientnet-b0":
            return data_transforms
        elif self.model_name == "efficientnet-b7":
            return data_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
    
    
    

