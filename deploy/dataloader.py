from torchvision.transforms import ToTensor, RandomCrop, CenterCrop, Resize
from torchvision import datasets, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import yaml
import h5py

def shift_color_dim(img):
    """ shift color axes to back """
    return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)

def get_mnist_loader(args, train=False, show=False):
    kwargs = {'num_workers': 1, 'pin_memory': True}  # if use_cuda else {}
    if show:
        return torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist',
                           train=False,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=1,
            shuffle=False,
            **kwargs)
    else:
        return torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)

class DataLoader:
    def __init__(self):
        self.device = torch.device("cuda")
        filename = '../data/imagenet_full/imagenet.hdf5'
        dictname = '../data/imagenet_full/dict.txt'
        self.dataset = ImageNetDataset(filename, train=False)
        with open(dictname, encoding='utf-8') as data_file:
            self.dict = yaml.load(data_file.read())

    def prepare(self, img):
        img = Resize(256)(img)
        # img = RandomCrop((224,224))(img)
        img = CenterCrop((224, 224))(img)
        img = ToTensor()(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        return img

    @staticmethod
    def get_result(model, img):
        model.eval()
        return model(img).argmax().detach().cpu().numpy()

    @staticmethod
    def show_img(img):
        plt.figure()
        plt.imshow(shift_color_dim(img.cpu().numpy()[0]))

    def get_label(self, num):
        return self.dict[num]

    def read(self, num):
        for i, (img, real_label) in enumerate(self.dataset):
            if i == num:
                return self.prepare(img)

class ImageNetDataset:
    def __init__(self, hdf5_filename, train, transform=None):
        self.hdf5_filename = hdf5_filename
        self.train = train
        self.dataset_name = 'train' if train else 'validation'
        self.transform = transform
        self.open = False
        self.h5 = None
        self.h5_images = None
        self.h5_targets = None

        with h5py.File(hdf5_filename, 'r') as tmp_h5:
            h5_targets = tmp_h5[self.dataset_name + '/targets']
            self.length = len(h5_targets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.open:
            self.h5 = h5py.File(self.hdf5_filename, 'r', swmr=True)
            self.h5_images = self.h5[self.dataset_name + '/images']
            self.h5_targets = self.h5[self.dataset_name + '/targets']
            self.open = True
        target = self.h5_targets[idx]
        jpg_bytes = self.h5_images[idx].tobytes()
        pil_image = Image.open(io.BytesIO(jpg_bytes))
        if self.transform is not None:
            img = self.transform(pil_image)
        else:
            img = pil_image
        return img, int(target)


class ClutteredMNIST:
    def __init__(self, data_dir, shape=(100, 100), n_clutters=6, clutter_size=8):
        self.dataset = datasets.MNIST(data_dir, download=True, train=True)
        self.shape = shape
        self.n_clutters = n_clutters
        self.clutter_size = clutter_size

    def __getitem__(self, idx):
        clutter_pos = []

        def place_clutter():
            rand_idx = np.random.randint(0, len(self.dataset))
            clutter_img = np.array(self.dataset[rand_idx][0])
            h, w = clutter_img.shape

            cs = self.clutter_size
            # select patch
            rh = np.random.randint(0, h - cs)
            rw = np.random.randint(0, w - cs)
            patch = clutter_img[rh:rh + cs, rw:rw + cs]

            # place patch
            rh = np.random.randint(0, self.shape[0] - cs)
            rw = np.random.randint(0, self.shape[1] - cs)
            canvas[rh:rh + cs, rw:rw + cs] = patch
            clutter_pos.append([rh, rw])

        canvas = np.zeros(self.shape, dtype=np.uint8)
        for _ in range(self.n_clutters):
            place_clutter()

        img, label = self.dataset[idx]
        img = np.array(img)

        num_rh = np.random.randint(0, self.shape[0] - img.shape[0])
        num_rw = np.random.randint(0, self.shape[1] - img.shape[1])

        canvas[num_rh:num_rh + img.shape[0], num_rw:num_rw + img.shape[1]] = img

        return canvas, label
