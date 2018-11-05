from torchvision.transforms import Compose, Normalize

from deploy.misc import *
from mlproject.data import *
from models.info_dropout import *

from .attr_project import *
from models.histo_comp import *

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
            n_clutters=config.get("clutters"),
            data_dir=config.get("data_dir", None),
            train_transform=train_transform,
            test_transform=test_transform,
            n_samples_train=config.get("train_samples"),
            n_samples_test=int(0.15 * config.get("train_samples")))

    def build_setup(self, config: dict):
        setup = ClutMNISTSetup(config)
        setup.loss.proj = self  # hack to make epoch accessible
        return setup

class ClutMNISTSetup(Setup):

    def build_model(self, config):
        return AllCNN96_HistoComp(config)

    def build_loss(self, config):
        return self.model

    def build_optimizer(self, config):
        lr = config["lr"]
        params = self.model.parameters()
        if config["opt"] == "adam":
            return torch.optim.Adam(params=params, lr=lr)
        else:
            return torch.optim.SGD(params=params, lr=lr, momentum=config["momentum"], nesterov=True)
