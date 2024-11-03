"""
Handle preprocessed ADNI files from Nifti files.
"""

import nibabel as nib
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from ad_fidelity.transforms import Center, MinMaxNorm, BoxCrop, RandomSagittalFlip, RandomTranslation
from torchvision.transforms.v2 import Compose, GaussianNoise

# TODO: implement lazy loading of ADNI files (slower, but requires way less RAM)

# adni nifti image affine
AFFINE = torch.tensor([
    [  -1.5,    0.,     0.,    90. ],
    [   0.,     1.5,    0.,  -126. ],
    [   0.,     0.,     1.5,  -72. ],
    [   0.,     0.,     0.,     1. ],
])

def train_test_datasets(train_files, train_labels, test_files, test_labels):
    """Setup Train and Test Datasets."""
    train_ds = NiftiDataset(train_files, train_labels, augment=True)
    # pass train dataset stats to test dataset
    test_ds = NiftiDataset(test_files, test_labels, augment=False, mu=train_ds.mu, vmin=train_ds.vmin, vmax=train_ds.vmax)
    return train_ds, test_ds


class NiftiDataset(Dataset):
    LOWER = [10, 13, 5]
    UPPER = [110, 133, 105]

    def __init__(self, nifti_files, labels, augment=True, center=True, crop=True, minmaxnorm=True,
                 mu=None, vmin=None, vmax=None, lower=None, upper=None):
        self.nifti_files = nifti_files
        if isinstance(labels, torch.Tensor):
            self.labels = labels.clone().detach()
        else:
            self.labels = torch.tensor(labels)
        self.samples = None
        self.augment = augment
        samples = []
        for file_path in tqdm(self.nifti_files):
            x = nib.load(file_path).get_fdata()
            x = torch.from_numpy(x)
            samples.append(x[torch.newaxis])
        self.samples = torch.stack(samples).float()
        # use CN mean to center data
        self.vmin = vmin if vmin is not None else self.samples.min()
        self.vmax = vmax if vmax is not None else self.samples.max()
        self.mu = mu if mu is not None else self.class_mean(0)
        self.mu_normed = (self.mu  - self.vmin) / (self.vmax - self.vmin)
        self.lower = lower if lower is not None else NiftiDataset.LOWER
        self.upper = upper if upper is not None else NiftiDataset.UPPER
        self.minmaxnorm = minmaxnorm
        self.center = center
        self.crop = crop
        self.transform = self.make_transform()
    
    def make_transform(self):
        """Construct data preprocessing transform."""
        transforms = []
        if self.minmaxnorm:
            xform_norm = MinMaxNorm(self.vmin, self.vmax)
            transforms.append(xform_norm)
        if self.center:
            # use normed mu if minmaxnorm transformed mean value
            mu = self.mu_normed if self.minmaxnorm else self.mu
            xform_center = Center(mu)
            transforms.append(xform_center)
        if self.crop:
            xform_crop = BoxCrop(lower=self.lower, upper=self.upper)
            transforms.append(xform_crop)
        if self.augment:
            xform_flip = RandomSagittalFlip()
            transforms.append(xform_flip)
            xform_shift = RandomTranslation(shift=5)
            transforms.append(xform_shift)
        return Compose(transforms)
         
    def class_mean(self, target):
        """Compute the feature mean of all samples labeled target."""
        return self.samples[self.labels == target].mean(0)
     
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples[index]
        if self.transform is not None:
            x = self.transform(x)
        y = self.labels[index]
        return x.float(), y

def get_nifti_files(ad_path: Path, cn_path: Path, ad_label: int=1, cn_label: int=0):
    """Returns nii.gz file paths with corresponding AD (1) and CN (0) labels."""
    ad_files = list(ad_path.glob("*.nii.gz"))
    cn_files = list(cn_path.glob("*.nii.gz"))
    file_paths = np.array(ad_files + cn_files)
    labels = np.array([ad_label] * len(ad_files) + [cn_label] * len(cn_files))
    return file_paths, labels

# Unused
def compute_mean(file_paths):
    """Lazily compute mean over nifti file paths."""
    mu = 0
    n = 0
    for file_path in file_paths:
        x = nib.load(file_path).get_fdata()
        x = torch.from_numpy(x)
        mu += x
        n += 1
    return mu / n

# Unused
def compute_minmax(file_paths):
    """Lazily compute min and max values over nifti file paths."""
    vmin = torch.inf
    vmax = -torch.inf
    for file_path in file_paths:
        x = nib.load(file_path).get_fdata()
        x = torch.from_numpy(x)
        vmin = min(vmin, x.min())
        vmax = max(vmax, x.max())
    return vmin, vmax
