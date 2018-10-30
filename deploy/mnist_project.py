from torchvision.transforms import Compose, RandomHorizontalFlip, Normalize

from deploy.misc import *
from mlproject.data import *
from models.allcnn32 import *
from models.allcnn96 import *

from .attr_project import *
from mlproject import *
from mlproject.model import SimpleModel

class MNISTProject(AttrProject):
    @staticmethod
    def get_dataset_factory(config):
        data_resize = config.get("data_resize", (32, 32))
        train_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        fac = MNISTDatasetFactory(
            batch_size=config["batch_size"],
            train_transform=train_transform,
            test_transform=test_transform)
        return fac

    @staticmethod
    def get_model(config):
        net = AllCNN32(config)
        loss = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=config["lr"])
        return SimpleModel("allcnn32", net, opt, loss, "gpu")
