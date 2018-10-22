

from deploy.dataloader import *
from deploy.misc import *
from methods.bottleneck import *
import mlproject
from mlproject.data import *
from models.lenet import *

from mlproject import *
from sacred import Experiment

class MNISTProject(MLProject):
    @staticmethod
    def get_dataset_factory(config):
        return MNISTDatasetFactory()

    @staticmethod
    def get_model(config):
        net = LeNet()
        opt = torch.optim.Adam(net.parameters())
        loss = nn.CrossEntropyLoss()
        return ClassificationModel(net, opt, loss=loss, name='test_mnist')


def test_mnist(tmpdir):
    ex = Experiment()
    ex.add_config({
        'batch_size': 5,
        'n_epochs':  1,
        'tensorboard_dir': None,
        'device': 'cuda:0',
        'model_dir': str(tmpdir.join('models')),
    })

    @ex.automain
    def main(_run):
        proj = MNISTProject.from_run(_run)
        print(proj.model._device_args, proj.model._device_kwargs)
        proj.test()
        proj.train()
        proj.test()

    ex.run()