import torch.nn as nn


class AllCNN32(nn.Module):
    """
    All-CNN-32 from https://arxiv.org/pdf/1611.01353.pdf
    (without softmax layer)
    """

    def __init__(self, config):
        super().__init__()
        # 32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        # 48x48
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        # 6x6
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(192, 10, kernel_size=1),
            nn.ReLU(),
        )
        # 1x1
        self.avg_pool = nn.AvgPool2d(kernel_size=6)
        self.print_shapes = False

    def family(self) -> str:
        return "All-CNN-32"

    def forward(self, x):
        if self.print_shapes: print(x.shape)
        x = self.conv1(x)
        if self.print_shapes: print(x.shape)
        x = self.conv2(x)
        if self.print_shapes: print(x.shape)
        x = self.conv3(x)
        if self.print_shapes: print(x.shape)
        x = self.avg_pool(x)
        if self.print_shapes: print(x.shape)
        x = x.view(-1, 10)
        if self.print_shapes: print(x.shape)
        return x
