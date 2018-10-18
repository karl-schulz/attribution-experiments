
from methods.base import *
import torch
import cv2
import numpy as np


class GradCAM(AttributionMethod):

    def __init__(self, model: torch.nn.Module, device, layer: torch.nn.Module):
        """
        :param model: model containing the softmax-layer
        :param device: dev
        :param layer: evaluation layer - object or name or id
        """
        self.layer = layer
        self.model = model
        self.device = device
        self.fmaps = None
        self.grads = None
        self.probs = None

        def func_f(module, input, output):
            self.fmaps = output.detach()

        def func_b(module, grad_in, grad_out):
            self.grads = grad_out[0].detach()

        layer.register_forward_hook(func_f)
        layer.register_backward_hook(func_b)

    def name(self):
        return "GradCAM of {}".format(type(self.layer))

    def pass_through(self, img):
        self.model.eval()
        img.requires_grad_()  # TODO remove this?
        self.model.zero_grad()
        return self.model(img)

    def get_map(self, img: torch.Tensor, target: torch.Tensor):

        # feed in
        self.model.eval()
        preds = self.pass_through(img)

        # calc grads
        grad_eval_point = torch.FloatTensor(1, preds.size()[-1]).zero_()
        grad_eval_point[0][preds.argmax().item()] = 1.0
        grad_eval_point = grad_eval_point.to(self.device)
        preds.backward(gradient=grad_eval_point, retain_graph=True)

        # weight maps
        maps = self.fmaps.detach().cpu().numpy()[0,]
        weights = self.grads.detach().cpu().numpy().mean(axis=(2,3))[0,:]

        # avg maps
        gcam = np.zeros(maps.shape[0:], dtype=np.float32)
        # sum up weighted fmaps
        for i, w in enumerate(weights):
            gcam += w * maps[i, :, :]

        # avg pool over layers
        gcam = np.mean(gcam, axis=0)
        # relu
        gcam = np.maximum(gcam, 0)
        gcam = cv2.resize(gcam, (224, 224))

        # rescale to [0,1]
        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam

