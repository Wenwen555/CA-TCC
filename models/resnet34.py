import torch

from models.resnet import resnet50, resnet101, resnet34, resnet18
import torch.nn as nn

class base_Model(nn.Module):
    def __init__(self, configs, strides=1):
        super().__init__()
        # self.resnet50 = resnet50()
        # self.resnet101 = resnet101()
        self.resnet34 = resnet34()
        # self.resnet18 = resnet18()


    def forward(self, x):
        return self.resnet34(x)


# def count_parameters(model):
#     count = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print('The model has {} trainable parameters'.format(count))
#
#
# if __name__ == "__main__":
#     model = base_Model(None)
#     count_parameters(model)
#
