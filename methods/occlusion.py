import numpy as np
from tqdm import tqdm
from methods.base import *

class Occlusion(AttributionMethod):

    def __init__(self, model, size, stride=None, patch_value=0, patch_type="color"):
        if not stride:
            stride = size
        self.model = model
        self.size = size
        self.stride = stride
        self.patch_value = patch_value
        self.patch_type = patch_type

    def occlude(self, img, x, y) -> np.array:
        patched = img
        if self.patch_type == "color":
            patched[0, 0, x:x + self.size, y:y + self.size] = self.patch_value
        elif self.patch_type == "avg":
            img[0, 0, x:x + self.size, y:y + self.size] = np.average(img[0, 0, x:x + self.size, y:y + self.size])
        elif self.patch_type == "inv":
            img[0, 0, x:x + self.size, y:y + self.size] = 1 - img[0, 0, x:x + self.size, y:y + self.size]
        else:
            raise ValueError("unknown patch type {}".format(self.patch_type))
        return patched

    def get_map(self, img, target):
        self.model.eval()
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
                # TODO fix stride methods
                hmap[x:x+self.stride,y:y+self.stride] = - (score - baseline)
        return hmap

    def name(self):
        return "OC [{}]".format(self.patch_type)

