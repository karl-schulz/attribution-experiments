import torch
import torch.nn as nn
from scipy.stats import norm

from deploy.utils import to_np
import numpy as np
import matplotlib.pyplot as plt

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
    """ TODO: resore log after passing thru """
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.outs = []
        self.log = False

    def forward(self, input):
        out = self.layer.forward(input)
        if self.log:
            for i in to_np(out):
                self.outs.append(i)
        return out

    def get_hist(self, model, feed_data, inspect: torch.Tensor):
        # reset first
        self.outs = []
        self.log = True
        # implicitly pass through our layer
        if isinstance(feed_data, tuple):
            feed_data = feed_data[0]
        if isinstance(feed_data, torch.Tensor):
            feed_data = [feed_data]
        for i in feed_data:
            print("feeding ", i.shape)
            model(i)
        # analyse
        print("out", self.outs[0].shape)
        stack = np.stack(self.outs)
        stack = stack[:,0]

        #hists = np.zeros(stack.shape[1:2])
        #print("hists", hists.shape)
        #for i in stack:
        #    stack

        flat = stack.flatten()
        nz = flat.nonzero()
        flat_z = flat[nz]
        flat_nz = flat[np.nonzero(flat)]
        plt.figure(figsize=(20, 10))
        plt.hist(flat.flatten(), bins=100, density=False)



        mu, std = norm.fit(flat)
        xmin, xmax = min(flat), max(flat)
        plt_x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(plt_x, mu, std)
        plt.plot(plt_x, p, 'k', linewidth=2)

        plt.show()

        print("stack", stack.shape)
        print(stack)
        print("stack")
