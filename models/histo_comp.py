import torch.nn as nn
import torch

class AllCNN96_HistoComp(nn.Module):
    """
    All-CNN-96 from https://arxiv.org/pdf/1611.01353.pdf
    (without softmax layer)
    """
    def __init__(self, config):
        super().__init__()
        # 96x96
        self.conv1 = HistoComp(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ))
        # 48x48
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 96, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        # 24x24
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(96, 192, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        # 12x12
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        # 6x6
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        # 1x1
        self.avg_pool = nn.AvgPool2d(kernel_size=6)
        self.cross_entropy = nn.CrossEntropyLoss()

    def get_loss(self, inputs, labels):
        out = self.forward(inputs)
        return self.cross_entropy(out, labels)

    @staticmethod
    def family() -> str:
        return "All-CNN-96 HistoComp"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(-1, 10)
        return x

class HistoComp(nn.Module):

    log: bool = False

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.outs = []

    def forward(self, input):
        out = self.layer.forward(input)
        if self.log:
            self.outs.append(out.detach())
        return out
