from torchvision.transforms import Compose, RandomHorizontalFlip, Normalize

from deploy.dataloader import *
from deploy.misc import *
from methods.bottleneck import *
import mlproject
from mlproject.data import *
from models.lenet import *

from mlproject import *
from mlproject.model import ProxyModel

class MNISTProject(MLProject):
    @staticmethod
    def get_dataset_factory(config):
        train_transform = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return MNISTDatasetFactory(train_transform=train_transform, test_transform=test_transform)

    @staticmethod
    def get_model(config):
        net = LeNet()
        loss = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters())
        return ProxyModel("test_mnist_lenet", net, opt, loss, "gpu")

