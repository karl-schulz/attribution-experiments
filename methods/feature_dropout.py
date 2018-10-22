
from methods.base import *
import torch
import cv2
import numpy as np


def replace_feature(noised, original, idx):
    noised_layer = Noised(original.features[idx])
    noised.features[idx] = noised_layer


class Noised(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        print("hooking {}".format(type(original).__name__))

    def forward(self, x):
        x = self.original.forward(x)
        return x


class FeatureOcclusion(AttributionMethod):

    def __init__(self, model: torch.nn.Module, layer):
        model.features[layer] = Noised(model.features[layer])

    def name(self):
        return "GradCAM of {}".format(type(self.layer).__name__)

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
        grad_eval_point.to(self.device)
        preds.backward(gradient=grad_eval_point, retain_graph=True)

        # weight maps
        maps = self.fmaps.detach().cpu().numpy()[0,]
        weights = self.grads.detach().cpu().numpy().mean(axis=(2,3))[0,:]

        # avg maps
        gcam = np.zeros(maps.shape[0:], dtype=np.float32)
        # sum up weighted fmaps
        for i, w in enumerate(weights):
            gcam += w * maps[i, :, :]

        # avg pool over feature maps
        gcam = np.mean(gcam, axis=0)
        # relu
        gcam = np.maximum(gcam, 0)
        gcam = cv2.resize(gcam, (224, 224))

        # rescale to [0,1]
        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam

