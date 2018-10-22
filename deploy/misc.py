from .dataloader import *
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
import numpy as np
import cv2
import abc
import torch

def find_layer(module, layer, module_name=None, strict=True):
    """
    search layer by name, id or object and return it, error otherwise
    :return:
    """
    if id(layer) == id(module):
        return module
    if module_name == layer:
        return module
    if isinstance(module, torch.nn.Sequential):
        for name, sub in module.named_children():
            found = find_layer(sub, layer, module_name=name, strict=False)
            if found:
                return found
    if strict:
        raise RuntimeError
    else:
        return False


def analyze_hmap(hmap: np.array):
    print("heatmap: {}".format(hmap.shape))
    print("max: {:.4f}".format(hmap.max()))
    print("min: {:.4f}".format(hmap.min()))
    print("avg: {:.4f}".format(hmap.mean()))
    print("std: {:.4f}".format(hmap.std()))

def compare_methods(methods: list, model, img, eval_class=None, img_trafo=None, hm_trafo=None, blur=None, cols=4, viz=None):
    rows = int(np.ceil((len(methods) + 1) / cols))
    dpi = 100
    tile = 1600/cols/dpi
    fig, subs = plt.subplots(rows, cols, figsize=(tile*cols,tile*rows))
    subs = subs.flatten()

    # fig.subplots_adjust(top=0.9, bottom=0.10, left=0.10, right=0.9, hspace=0.3, wspace=0.2)

    # show real img
    sub = subs[0]
    np_img = img.cpu().numpy().squeeze()
    # maybe switch axes etc (for resnet input)
    if img_trafo:
        np_img = img_trafo(np_img)
    byteimg = (np_img * 255).astype("uint8")
    sub.imshow(Image.fromarray(byteimg), 'Greys')
    sub.set_label(model(img).argmax().cpu().numpy())

    # calc heatmaps and show them
    hmaps = []
    for i, method in enumerate(methods):
        out = model(img)
        result_label = torch.LongTensor(eval_class) if eval_class else out.argmax().unsqueeze(0)
        heatmap = method.get_map(img=img, target=result_label)
        if hm_trafo:
            heatmap = hm_trafo(heatmap)

        if blur:
            from scipy.ndimage.filters import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=blur)

        # show heatmap
        sub = subs[i + 1]
        if viz:
            viz.plot(heatmap, img=np_img, fig=sub)
        else:
            im = Heatmap().plot(heatmap, img=np_img, fig=sub)
            fig.colorbar(im, ax=sub)
        sub.set_title(method.name())
        hmaps.append(heatmap)

    # adjust imgs
    # fig.subplots_adjust(top=0.9, bottom=0.10, left=0.10, right=0.9, hspace=0.3, wspace=0.2)
    fig.subplots_adjust(hspace=0.15)
    return hmaps

class Viz:
    @abc.abstractmethod
    def plot(self, hmap: np.array, img: np.array, fig):
        pass

    @staticmethod
    def rectify(hmap, signed, vmin=0.0, vmax=1.0):
        if signed:
            # signed - 0 is at bias, max is at absmax(hmap)
            bias = (vmax - vmin)/2
            scale = max(abs(hmap.max()), abs(hmap.min()))
        else:
            # unsigned - 0 is at min(hmap), max is at max(hmap)
            bias = vmin
            scale = hmap.max() - hmap.min()
        if scale:  # dont divide by 0
            hmap = (hmap / scale) * vmax + bias
        else:
            print("scale == 0!")
        return hmap

    @staticmethod
    def put_scale(hmap, vmin=0.0, vmax=1.0):
        tile = int(np.round(hmap.shape[0] / 20))
        hmap[0:tile, 0:tile] = vmax
        hmap[0:tile, tile:2 * tile] = vmin
        return hmap

class Heatmap(Viz):
    def plot(self, hmap: np.array, img: np.array, fig):
        vmin = 0
        vmax = 255
        hmap = super().rectify(hmap, vmin=vmin, vmax=vmax, signed=True)
        return fig.imshow(hmap.astype(int), cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax)

class Overlay(Viz):
    def plot(self, hmap: np.array, img: np.array, fig):
        assert len(hmap.shape) == 2, "hmap has not the right shape: {}".format(hmap.shape)
        hmap = Heatmap().rectify(np.maximum(hmap.astype(float), 0), signed=False)
        hmap = super().put_scale(hmap)
        hmap_alpha = np.minimum(0.8, hmap*2)
        hmap = cv2.applyColorMap(np.uint8(255 - 255*hmap), cv2.COLORMAP_JET)
        hmap = np.float32(hmap) / 255.0
        overlay = img.copy() # dont override img
        # TODO das geht schöner
        overlay[:,:,0] = img[:,:,0] * (1-hmap_alpha) + hmap[:,:,0] * hmap_alpha
        overlay[:,:,1] = img[:,:,1] * (1-hmap_alpha) + hmap[:,:,1] * hmap_alpha
        overlay[:,:,2] = img[:,:,2] * (1-hmap_alpha) + hmap[:,:,2] * hmap_alpha
        return fig.imshow(overlay)
