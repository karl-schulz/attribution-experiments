import torch
import numpy as np
from mlproject.log import DevNullSummaryWriter

from deploy.setup import Setup
from mlproject import MLProject
from tqdm import tqdm
import os
from deploy.utils import *
from tensorboardX import SummaryWriter
from time import gmtime
from time import strftime

class AttrProject(MLProject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup = self.model
        self._init_writer(self.config)

    def _init_writer(self, config):
        """ set the writer """
        tb_base = os.environ['TENSORBOARD_DIR'] if 'TENSORBOARD_DIR' in os.environ else None
        tb_dir = config.get("tensorboard_dir", "default")
        if tb_base is None or tb_dir is None:
            self.writer = DevNullSummaryWriter()
        else:
            if tb_dir == "default":
                tb_dir = strftime("%m-%d/%H-%M-%S", gmtime())
            self.writer = SummaryWriter(os.path.join(tb_base, tb_dir))

    def get_model(self, config: dict) -> Setup:
        """ really: build_setup """
        return self.build_setup(config)

    def build_setup(self, config) -> Setup:
        raise NotImplementedError

    def get_setup(self) -> Setup:
        """ holder is a nicer name for self.model """
        return self.model

    def train_single_sample(self, iters=100, idx=0):
        print("training on single data...")
        data, label = self.single_datapoint(idx=idx)
        test = self.setup.test_batch((data, label))
        print(test)
        for i in tqdm(range(0,iters)):
            self.setup.optimizer.zero_grad()
            loss = self.setup.model.get_loss(data, label)
            loss.backward()
            self.setup.optimizer.step()
        test = self.setup.test_batch((data, label))
        print(test)

    def info_grad(self, layer: torch.nn.Module, input: torch.Tensor = None):
        if input is None:
            input, _ = next(self.dataset_factory.test_loader().__iter__())
            input = input.to(self.device)
            input = input[0].unsqueeze(0)  # single element

        # forward + backward
        self.setup.model.zero_grad()
        out = self.setup.model(input)
        one_hot = torch.zeros(out.shape).to(self.device)
        one_hot[0, out.argmax(dim=1)] = 1
        out.backward(one_hot)

        # get info
        params = np.concatenate([to_np(p).flatten() for p in layer.parameters()])
        grads = np.concatenate([to_np(p.grad).flatten() for p in layer.parameters()])
        print("param dim", params.shape)
        print("param std", params.std())
        print("param mean", params.mean())
        print("grad mean", grads.mean())
        print("grad std", grads.std())

    def test(self):
        print("testing...")
        return super().test()

    def train(self):
        self.setup.loss.epoch = self.epoch
        return super().train()

    def visualize_info_dropout(self, layers):

        # pass thru anything
        self.data_example()

        for i, l in enumerate(layers):
            alpha = l.last_alpha
            hmap = alpha.mean(axis=(0,1))
            print("alpha: shape={}, max={}, min={}, std={:04f}, mean={}".format(
                hmap.shape, hmap.max(), hmap.min(), hmap.std(), hmap.mean()))
            plt.imshow(hmap, cmap="Greys")
            plt.title("alpha of {}".format(i))
            plt.show()

        for i, l in enumerate(layers):
            kl = l.last_kls.mean(axis=(0,1))
            print(kl.shape)
            hmap = kl
            print("KL: shape={}, max={}, min={}, std={:04f}, mean={}".format(
                hmap.shape, hmap.max(), hmap.min(), hmap.std(), hmap.mean()))
            plt.imshow(hmap, cmap="Greys")
            plt.title("KL of {}".format(i))
            plt.show()

    def single_batch(self, idx: int=0) -> (torch.Tensor, torch.Tensor):
        """ dim: BxCxHxW """
        # TODO unneccessary iterating, jump would be better (however, itertools itslice doesnt work somehow)
        for i, (dat, labels) in enumerate(self.dataset_factory.test_loader().__iter__()):
            if i == idx:
                return dat.to(self.device), labels.to(self.device)
        raise RuntimeError

    def single_datapoint(self, idx: int=0) -> (torch.Tensor, torch.Tensor):
        """ dim: 1xCxHxW """
        data, labels = self.single_batch(idx)
        return data[0].unsqueeze(0), labels[0].unsqueeze(0)

    def data_example(self, idx: int=0):
        data, labels = self.single_datapoint(idx)
        data = data.to(self.device)
        img = data[0].unsqueeze(0)
        show_img(img[0])
        self.setup.print_shapes = True
        out = self.setup.forward(img)  # pass thru
        print("label: ", out.detach().cpu().numpy())
        self.model.model.print_shapes = False

    def data_info(self):
        print("train set", len(self.dataset_factory.train_set()))
        print("train batch", len(self.dataset_factory.train_loader()))
        print("test set", len(self.dataset_factory.test_set()))
        print("test batch", len(self.dataset_factory.test_loader()))

    def model_info(self):
        print("model family", self.model.model.family())
        print("optimizer", self.setup.optimizer)
        print("parameters", len(list(self.setup.model.parameters())))
