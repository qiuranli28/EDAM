import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter


__all__ = ['Classifier', 'NormalizedClassifier']


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.flatten = nn.Flatten().to("cuda:0")
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        # print(self.flatten(x).shape)
        x = self.flatten(x)
        y = self.classifier(x)

        return y
        

class NormalizedClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten().to("cuda:0")
        self.weight = Parameter(torch.Tensor(num_classes, feature_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5) 

    def forward(self, x):
        w = self.weight  

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        x = self.flatten(x)

        return F.linear(x, w)



