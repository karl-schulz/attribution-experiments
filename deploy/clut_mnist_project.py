from torchvision.transforms import Compose, Normalize

from deploy.misc import *
from mlproject.data import *
from models.allcnn96 import *
from models.holder import *

from .attr_project import *

class ClutteredMNISTProject(AttrProject):
    @staticmethod
    def get_dataset_factory(config):
        data_resize = config.get("data_resize", (96, 96))
        train_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return ClutteredMNISTDatasetFactory(
            batch_size=config["batch_size"],
            shape=(96, 96),
            use_filesys=config.get("data_use_filesys", True),
            n_clutters=config.get("clutters", 21),
            data_dir=config.get("data_dir", None),
            train_transform=train_transform,
            test_transform=test_transform,
            n_samples_train=60000,
            n_samples_test=10000)

    @staticmethod
    def get_model(config):
        net = AllCNN96(config)
        loss = nn.CrossEntropyLoss()
        # opt = torch.optim.Adam(net.parameters(), lr=config["lr"])
        params = net.parameters()
        lr = config["lr"]
        if config["opt"] == "adam":
            opt = torch.optim.Adam(params=params, lr=lr)
        else:
            opt = torch.optim.SGD(params=params, lr=lr, momentum=config["momentum"], nesterov=True)
        return ModelHolder(model=net, optimizer=opt, loss=loss, device=config["device"])
