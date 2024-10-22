import nibabel as nib
import torch

from tqdm import tqdm, trange
from torch.utils.data import Dataset

class NiftiDataset(Dataset):
    def __init__(self, nifti_files, labels, augment=True, transform=None, lazy=False):
        self.nifti_files = nifti_files
        if isinstance(labels, torch.Tensor):
            self.labels = labels.clone().detach()
        else:
            self.labels = torch.tensor(labels)
        self.samples = None
        self.augment = augment
        self.transform = transform
        # TODO: implement lazy loading?
        self.lazy = lazy
        samples = []
        for file_path in tqdm(self.nifti_files):
            x = nib.load(file_path).get_fdata()
            x = torch.from_numpy(x)
            samples.append(x[torch.newaxis])
        self.samples = torch.stack(samples).float()
    
    def class_mean(self, target):
        return self.samples[self.labels == target].mean(0)
    
    def center(self, mu=None):
        """Center samples around 0."""
        self.mu = mu
        if self.mu is None:
            self.mu = self.samples.mean(0)
        self.samples -= self.mu
        return self.mu
    
    def norm(self, vmin=None, vmax=None):
        """Norm samples to [0,1]."""
        self.vmin = vmin if vmin is not None else self.samples.min()
        self.vmax = vmax if vmax is not None else self.samples.max()
        self.samples = (self.samples - self.vmin) / (self.vmax - self.vmin)
        return self.vmin, self.vmax
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples[index]
        if self.augment and self.transform is not None:
            x = self.transform(x)
        y = self.labels[index]
        return x.float(), y

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
