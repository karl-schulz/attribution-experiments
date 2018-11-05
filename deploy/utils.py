import torch
import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()

def to_np(t: torch.Tensor):
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    return t.numpy()