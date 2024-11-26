"""Python file to instantite the model and the transform that goes with it."""
from data import data_transforms, resnet_transforms, data_aug_transforms
from model import Net
from resnet import ResnetClf
from dinov2_clf import Dinov2CLF

class ModelFactory:
    def __init__(self, model_name: str, weight_path: str = "facebook/dinov2-giant", dropout: float = 0.0, frozen_strategy: str = "all", aug_flag: bool = False, embedding_strategy: str = "cls+seq_emb"):
        self.model_name = model_name
        self.weight_path = weight_path
        self.dropout = dropout
        self.frozen_strategy = frozen_strategy
        self.embedding_strategy = embedding_strategy
        self.aug_flag = aug_flag
        self.model = self.init_model()
        self.train_transform, self.val_transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet":
            return ResnetClf()
        elif self.model_name == "dinov2":
            return Dinov2CLF(weight_path=self.weight_path, dropout=self.dropout, frozen_strategy=self.frozen_strategy, embedding_strategy=self.embedding_strategy)
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms, data_transforms
        if (self.model_name == "resnet" or self.model_name == "dinov2") and not self.aug_flag:
            return resnet_transforms, resnet_transforms
        elif (self.model_name == "restnet" or self.model_name == "dinov2_aug") and self.aug_flag:
            return data_aug_transforms, resnet_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.train_transform, self.val_transform

    def get_all(self):
        return self.model, self.train_transform, self.val_transform
