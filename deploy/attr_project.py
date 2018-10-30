import itertools

import torch
import numpy as np
from models.holder import ModelHolder
from mlproject import *
from tqdm import tqdm
from abc import ABC
import matplotlib.pyplot as plt

class AttrProject(MLProject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.holder = self.model

    def get_holder(self) -> ModelHolder:
        """ holder is a nicer name for self.model """
        return self.model

    def train_single_datapoint(self, iters=100):
        print("training on single data...")
        data, label = self.single_datapoint()
        test = self.holder.test_batch((data, label))
        print(test)
        for i in tqdm(range(0,iters)):
            self.holder.optimizer.zero_grad()
            output = self.holder.model(data)
            loss = self.holder.loss(output, label)
            loss.backward()
            self.holder.optimizer.step()
        test = self.holder.test_batch((data, label))
        print(test)

    def info_grad(self, layer: torch.nn.Module, input: torch.Tensor = None):
        if input is None:
            input, _ = next(self.dataset_factory.test_loader().__iter__())
            input = input.to(self.device)
            input = input[0].unsqueeze(0)  # single element

        # forward + backward
        self.holder.model.zero_grad()
        out = self.holder.model(input)
        one_hot = torch.zeros(out.shape).to(self.device)
        one_hot[0, out.argmax(dim=1)] = 1
        out.backward(one_hot)

        # get info
        params = np.concatenate([p.detach().cpu().numpy().flatten() for p in layer.parameters()])
        grads = np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in layer.parameters()])
        print("param dim", params.shape)
        print("param std", params.std())
        print("param mean", params.mean())
        print("grad mean", grads.mean())
        print("grad std", grads.std())

    def test(self):
        print("testing...")
        return super().test()

    def train(self):
        return super().train()

    @staticmethod
    def show_img(img):
        if len(img.shape) == 4:
            img = img[0]
        dims = tuple(img.shape)
        if isinstance(img, torch.Tensor):
            if img.is_cuda:
                img = img.cpu()
            img = img.numpy()
            # typically color is not last
            if len(dims) == 3:
                img = np.swapaxes(np.swapaxes(img, 2, 0), 1, 0)
        img = img.squeeze()
        if len(img.shape):
            cmap = "Greys"
        else:
            cmap = None
        plt.imshow(img, cmap=cmap)

    def single_batch(self, idx: int=0) -> (torch.Tensor, torch.Tensor):
        """ dim: BxCxHxW """
        # TODO unneccessary iterating, jump would be better (however, itertools itslice doesnt work)
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
        self.show_img(img[0])
        self.holder.print_shapes = True
        out = self.holder.forward(img)  # pass thru
        print("label: ", out.detach().cpu().numpy())
        self.model.model.print_shapes = False

    def data_info(self):
        print("train set", len(self.dataset_factory.train_set()))
        print("train batch", len(self.dataset_factory.train_loader()))
        print("test set", len(self.dataset_factory.test_set()))
        print("test batch", len(self.dataset_factory.test_loader()))

    def model_info(self):
        print("model family", self.model.model.family())
        print("optimizer", self.holder.optimizer)
        print("parameters", len(list(self.holder.model.parameters())))
