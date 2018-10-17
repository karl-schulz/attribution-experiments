
import PIL
import io
import yaml
import h5py

def pil_bgr2rgb(im):
    b, g, r = im.split()
    im = PIL.Image.merge("RGB", (r, g, b))
    return im

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
        pil_image = PIL.Image.open(io.BytesIO(jpg_bytes))
        if self.transform is not None:
            img = self.transform(pil_image)
        else:
            img = pil_image
        return img, int(target)

# setup data
filename = '../data/imagenet_full/imagenet.hdf5'
show_loader = ImageNetDataset(filename, train=False)
dct = None
with open('../data/imagenet_full/dict.txt', encoding='utf-8') as data_file:
    dct = yaml.load(data_file.read())
