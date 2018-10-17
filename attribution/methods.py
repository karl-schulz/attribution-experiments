from __future__ import print_function
import abc
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.autograd
import numpy as np

class AttributionMethod:
    """ Something than can make a attribution heatmap from a model plus input"""
    @abc.abstractmethod
    def get_map(self, img: torch.Tensor, target: torch.Tensor):
        pass

    @abc.abstractmethod
    def name(self):
        pass

class Occlusion(AttributionMethod):

    def __init__(self, model, size, stride=None, patch_color=0, patch_type="color"):
        if not stride:
            stride = size
        self.model = model
        self.size = size
        self.stride = stride
        self.patch_color = patch_color
        self.patch_type = patch_type

    def occlude(self, img, x, y) -> np.array:
        patched = img
        if self.patch_type == "color":
            patched[0, 0, x:x + self.size, y:y + self.size] = self.patch_color
        elif self.patch_type == "avg":
            img[0, 0, x:x + self.size, y:y + self.size] = np.average(img[0, 0, x:x + self.size, y:y + self.size])
        elif self.patch_type == "inv":
            img[0, 0, x:x + self.size, y:y + self.size] = 1 - img[0, 0, x:x + self.size, y:y + self.size]
        else:
            raise ValueError("unknown patch type " + self.patch_type)
        return patched

    def get_map(self, img, target):
        self.model.eval()
        x = 0
        y = 0
        # img: 1xCxNxN
        w = img.shape[2]
        h = img.shape[3]
        wsteps = 1 + int(np.floor((w-self.size) / self.stride))
        hsteps = 1 + int(np.floor((h-self.size) / self.stride))
        steps = wsteps * hsteps
        with torch.no_grad():
            baseline = self.model(img).squeeze()[target.cpu().numpy()]
            hmap = np.zeros((w, h))
            for step in tqdm(range(steps), ncols=100):
                wstep = step % wsteps
                hstep = int((step - wstep) / wsteps)
                y = hstep * self.stride
                x = wstep * self.stride
                assert((x + self.size) <= w)
                assert((y + self.size) <= h)
                patched = self.occlude(img.cpu().numpy(), x, y)
                out = self.model(torch.as_tensor(patched, device=target.device))
                score = out.squeeze()[target.cpu().numpy()]
                # nicht ganz sauber
                # TODO fix stride attribution
                hmap[x:x+self.stride,y:y+self.stride] = - (score - baseline)
        return hmap

    def name(self):
        return "OC [{}]".format(self.patch_type)


class AvgMethod(AttributionMethod):
    """ Something than can make a attribution heatmap from several heatmaps """

    def __init__(self, model: torch.nn.Module, steps: int, times_input: bool, cc_transform, verbose=False):
        self.steps = steps
        self.times_input = times_input
        self.verbose = verbose
        self.cc_transform = cc_transform
        self.model = model

    @abc.abstractmethod
    def get_map(self, img: torch.Tensor, target: torch.Tensor):
        pass

    def get_gradients(self, images, input, target: torch.Tensor):
        # no stochastics!
        self.model.eval()

        # init
        shape = images[0].shape
        grads = np.zeros((shape[2], shape[3], self.steps))

        for i in tqdm(range(self.steps), ncols=100):
            # reset grad
            self.model.zero_grad()

            # pass thru
            tmp_image = images[i]
            img_var = torch.autograd.Variable(tmp_image, requires_grad=True)
            output = self.model(img_var)
            loss = F.nll_loss(output, target)

            loss.backward(retain_graph=True)

            # get grad
            grad = torch.autograd.grad(loss, img_var)[0].cpu().squeeze().numpy()

            # muss hier gemacht werden, weil die dim des gradienten nohc die channels enthält
            if self.times_input:
                grad *= input.cpu().numpy().squeeze()

            # collect grad
            if len(grad.shape) == 2:
                # add color dimension
                grad.unsqueeze(0)
            # add/avg/... up color channels
            if self.cc_transform == "max":
                grads[:, :, i] = np.max(grad, axis=0)
            elif self.cc_transform == "max-of-abs":
                grads[:, :, i] = np.max(np.abs(grad), axis=0)
            elif self.cc_transform == "mean-of-abs":
                grads[:, :, i] = np.mean(np.abs(grad), axis=0)
            elif self.cc_transform == "mean":
                grads[:, :, i] = np.mean(grad, axis=0)
            else:
                raise ValueError(self.cc_transform)

        return grads

    def get_averages(self, target_label, grads: np.array):

        # sanity check
        if self.verbose:
            print("mean stddev per pixel for {} = {:.4f}".format(target_label, np.std(grads, axis=2).mean()))
            print("mean mean per pixel for {} = {:.4f}".format(target_label, np.mean(grads)))

        avg_grad = np.average(grads, axis=2)  # img_tensor.cpu().numpy().squeeze()*

        return avg_grad

class SmoothGrad(AvgMethod):

    def __init__(self, model: torch.nn.Module, std, steps=50, times_input=False, cc_transform="max-of-abs"):
        super().__init__(model, steps, times_input, cc_transform=cc_transform)
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
        # 0 <= val <= 1
        noise_images = [torch.clamp(img, 0, 1) for img in noise_images]
        # calc gradients
        grads = super().get_gradients(images=noise_images, input=img, target=target)

        return super().get_averages(target.cpu().numpy(), grads)


class IntegratedGradients(AvgMethod):
    def __init__(self, model: torch.Tensor, baseline=0.0, steps=50, only_positive=False, times_input=False, cc_transform="max-of-abs"):
        """
        :param baseline: start point for interpolation
        :param steps: resolution
        :param only_positive: clamp?
        :param times_input: if to multiply with the source image
        :param cc_transform: how to evaluate the color channel gradients
        """
        super().__init__(model, steps, times_input=times_input, cc_transform=cc_transform)
        self.only_positive = only_positive
        self.bl = baseline
        self.steps = steps

    def name(self):
        ret = "IG: steps={}; baseline={}".format(self.steps, self.bl)
        if self.times_input:
            ret += "; [*in]"
        if self.only_positive:
            ret += "; [OP]"
        return ret

    def get_map(self, img: torch.Tensor, target: torch.LongTensor) -> np.array:

        # no stochastics!
        self.model.eval()

        # scaled test images
        scaled_images = [((float(i) / self.steps) * (img - self.bl) + self.bl) for i in range(1, self.steps + 1)]

        grads = super().get_gradients(images=scaled_images, input=img, target=target)
        avg_grad = super(IntegratedGradients, self).get_averages(target.cpu().numpy(), grads)

        # postprocess
        if self.only_positive:
            avg_grad = np.maximum(avg_grad, 0)

        return avg_grad

