import nibabel as nib
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.transforms import Compose, Normalize

DATA_PATH = "data/adni/"

# Mean and Variance of cropped training data
MU = 0.1291
SIGMA = 0.2361

# lazy loading for minimal RAM


class NiftiDataset(Dataset):
    """Load MRI Data with nibabel."""

    def __init__(self, nifti_files, labels, transform=None):
        self.nifti_files = nifti_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.nifti_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = self.labels[idx]
        path = self.nifti_files[idx]

        img = nib.load(path)
        x = torch.tensor(img.get_fdata()).float()
        x = x.unsqueeze(0)

        if self.transform:
            x = self.transform(x)

        return (x, y)

# load all files at the same time, if enough RAM is available


def load_nifti(files, transform=None):
    X = []
    for fpath in files:
        img = nib.load(fpath)
        x = torch.tensor(img.get_fdata()).float()
        x = x.unsqueeze(0)
        if transform is not None:
            x = transform(x)
        X.append(x)
        del img
    return torch.stack(X)

# TensorDataset does not support transforms


class TransformTensorDataset(Dataset):
    """Load MRI Data with nibabel."""

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return (x, y)

# define preprocessing and data augmentation transforms


class Crop:
    def __init__(self, xcrop, ycrop, zcrop):
        self.xcrop = xcrop
        self.ycrop = ycrop
        self.zcrop = zcrop

    def __call__(self, x):
        return x[:, self.xcrop[0]:self.xcrop[1], self.ycrop[0]:self.ycrop[1], self.zcrop[0]:self.zcrop[1]]


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            # torch doesnt seem to like negative step size
            # x = x[::-1, :, :]
            x = torch.flip(x, [1])
        return x


class RandomShift:
    def __init__(self, shift=5, fill=0):
        self.shift = shift
        self.fill = fill

    def __call__(self, x):
        s = torch.randint(-self.shift, self.shift, (3,))
        padded = F.pad(x, [self.shift] * 6, "constant", self.fill)
        padded = torch.roll(padded, (s[0], s[1], s[2]), (1, 2, 3))
        return padded[:, self.shift:-self.shift, self.shift:-self.shift, self.shift:-self.shift]


crop = Crop((10, 110), (13, 133), (5, 105))

augment = Compose([RandomFlip(), RandomShift(shift=3)])

tf_train = Compose([
    Normalize([MU], [SIGMA]),
    augment
])

tf_test = Compose([
    Normalize([MU], [SIGMA])
])
