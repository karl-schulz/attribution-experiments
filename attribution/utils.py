import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
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

class Heatmap(Viz):

    def prepare(self, hmap, vmin=0, vmax=255):
        abs_max = max(abs(hmap.max()), abs(hmap.min()))
        if abs_max:
            hmap = 128 + hmap / abs_max * 127
        hmap[0, 0] = vmin
        hmap[0, 1] = vmax
        return hmap

    def plot(self, hmap: np.array, img: np.array, fig):
        vmin = 0
        vmax = 255
        hmap = self.prepare(hmap, vmin=vmin, vmax=vmin)
        return fig.imshow(hmap, cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax)

class Overlay(Viz):

    def plot(self, hmap: np.array, img: np.array, fig):
        hmap = Heatmap().prepare(hmap)
        img = 0.8 * hmap + 0.2 * img
        return fig.imshow(hmap, cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax)
