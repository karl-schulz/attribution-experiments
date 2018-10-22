from __future__ import print_function

import numpy as np
import torch.nn.functional as F
import torch.autograd
from methods.base import *
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image

class AvgMethod(AttributionMethod):
    """ Something than can make a methods heatmap from several heatmaps """

    def __init__(self, model: torch.nn.Module, steps: int, cc_transform, mode):
        self.steps = steps
        self.verbose = False
        self.times_input = False
        self.cc_transform = cc_transform
        self.model = model
        self.mode = mode

    @abc.abstractmethod
    def get_map(self, img: torch.Tensor, target: torch.Tensor):
        pass

    def get_gradients(self, images, input, target: torch.Tensor):
        # no stochastics!
        self.model.eval()

        assert len(images[0].shape) == 4, "{} makes dim {}".format(images[0].shape, len(images[0].shape))  # C x N x N

        # init
        grads = np.zeros((len(images), *images[0].shape))

        for i in tqdm(range(self.steps), ncols=100):
            # reset grad
            self.model.zero_grad()

            # pass thru
            img_var = torch.autograd.Variable(images[i], requires_grad=True)
            logits = self.model(img_var)
            probs = F.softmax(logits, dim=1)
            np_target = target.item()
            if self.mode == "probs":
                grad_eval_point = torch.FloatTensor(1, probs.size()[-1]).zero_()
                grad_eval_point[0][np_target] = 1.0
                grad_eval_point = grad_eval_point.to(input.device)
                criterion = probs
            elif self.mode == "logits":
                grad_eval_point = torch.FloatTensor(1, logits.size()[-1]).zero_()
                grad_eval_point[0][np_target] = 1.0
                grad_eval_point = grad_eval_point.to(input.device)
                criterion = logits
            elif self.mode == "nll_loss":
                loss = F.nll_loss(logits, torch.as_tensor(np_target).to(input.device))
                grad_eval_point = None
                criterion = loss
            else:
                raise ValueError

            # calc grad
            criterion.backward(gradient=grad_eval_point)
            grad = img_var.grad
            # muss hier gemacht werden, weil die dim des gradienten nohc die channels enthält
            if self.times_input:
                grad *= input.cpu().numpy().squeeze()

            # collect grad
            if len(grad.shape) == 3:
                # add color dimension
                np.expand_dims(grad, axis=0)

            grads[i, :, :, :, :] = grad

        return grads

    def get_averages(self, grads: np.array):

        # process color channels
        if self.cc_transform == "max":
            grads = np.max(grads, axis=2)
        elif self.cc_transform == "max-of-abs":
            grads = np.max(np.abs(grads), axis=2)
        elif self.cc_transform == "mean-of-abs":
            grads = np.mean(np.abs(grads), axis=2)
        elif self.cc_transform == "mean":
            grads = np.mean(grads, axis=2)
        else:
            raise ValueError(self.cc_transform)

        # average up steps
        avg_grad = np.average(grads, axis=0).squeeze()

        return avg_grad

class SmoothGrad(AvgMethod):

    def __init__(self, model: torch.nn.Module, std, steps=50, cc_transform="max-of-abs", mode="probs"):
        super().__init__(model=model, steps=steps, cc_transform=cc_transform, mode=mode)
        self.std = std
        self.model = model

    def name(self):
        if self.std != 0:
            ret = "SG: steps={}; std={}".format(self.steps, self.std)
        else:
            ret = "Grads: "
        if self.times_input:
            ret += "; [*in]"
        return ret

    def get_map(self, img: torch.Tensor, target: torch.LongTensor) -> np.array:

        # noised test images
        noises = [torch.randn(*img.shape).to(img.device) * self.std for _ in range(0, self.steps)]
        noise_images = [img + noises[i] for i in range(0, self.steps)]
        noise_images = [torch.clamp(img, 0, 1) for img in noise_images]

        # calc gradients
        grads = super().get_gradients(images=noise_images, input=img, target=target)
        return super().get_averages(grads)


class IntegratedGradients(AvgMethod):
    def __init__(self, model: torch.Tensor, baseline=0.0, steps=50, cc_transform="max-of-abs", mode="probs"):
        """
        :param baseline: start point for interpolation (0-1 grey, or "inv", or "avg")
        :param steps: resolution
        :param cc_transform: how to evaluate the color channel gradients
        """
        super().__init__(model=model, steps=steps, cc_transform=cc_transform, mode=mode)
        self.bl = baseline
        self.steps = steps

    def name(self):
        ret = "IG: steps={}; bl={}; mode={}".format(self.steps, self.bl, self.mode)
        if self.times_input:
            ret += "; [*in]"
        return ret

    def get_map(self, img: torch.Tensor, target: torch.LongTensor) -> np.array:

        # no stochastics!
        self.model.eval()

        # scaled test images
        if isinstance(self.bl, str):
            if self.bl == "inv":
                bl_img = 1 - img
            elif self.bl == "avg":
                bl_img = img.mean()
            else:
                raise ValueError
        else:
            bl_img = self.bl

        scaled_images = [((float(i) / self.steps) * (img - bl_img) + bl_img) for i in range(1, self.steps + 1)]

        if isinstance(self, str):
            # debug
            np_img = scaled_images[5].cpu().numpy()[0,0,:,:] * 255
            plt.imshow(Image.fromarray(np_img))
            plt.show()
            np_img = scaled_images[25].cpu().numpy()[0,0,:,:] * 255
            plt.imshow(Image.fromarray(np_img))
            plt.show()

        grads = super().get_gradients(images=scaled_images, input=img, target=target)
        avg_grad = super().get_averages(grads)

        # sanity check
        if self.verbose:
            print("avg = {:.4f}".format(np.sum(avg_grad)))
            print("sum = {:.4f}".format(np.sum(avg_grad) * float(self.steps)))
            print("score = {:.4f}".format(torch.nn.Softmax(dim=1)(self.model(img)).max()))
            print("bl score = {:.4f}".format(torch.nn.Softmax(dim=1)(self.model(scaled_images[0].to(img.device))).max()))

        return avg_grad

