"""
Torchvision transforms for augmenting MRI brain images.
"""
import torch
import torch.nn.functional as F

# from torchvision.transforms.v2 import Compose, Transform, RandomCrop, RandomHorizontalFlip


class BoxCrop:
    """Crop the MRI image to a index box."""
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, x):
        return x[
            ...,
            self.lower[0]:self.upper[0],
            self.lower[1]:self.upper[1],
            self.lower[2]:self.upper[2]
        ]

class RandomSagittalFlip:
    """Randomly flip MRI image along sagittal axis."""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            # torch still doesn't like negative step sizes...
            # x = x[::-1]
            # dims: [batch?, channel?, sagittal, coronal, axial]
            x = torch.flip(x, [-3])
        return x

class RandomTranslation:
    """Randomly shift the MRI image into xyz directions with padding."""

    def __init__(self, shift=5, pad=0):
        self.pad = pad
        self.shift = shift
    
    def __call__(self, x):
        # pad each dimension with (shift) elements before and after
        x = F.pad(x, tuple([self.shift] * 6), "constant", value=self.pad)
        shifts = torch.randint(-self.shift, self.shift, [3])
        # without padding, roll might flip elements around to the other side
        x = torch.roll(x, tuple(shifts.tolist()), tuple([-3, -2, -1]))
        return x[..., self.shift:-self.shift, self.shift:-self.shift, self.shift:-self.shift]

class MinMaxNorm:
    """Normalize x to be within 0 and 1."""
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
    
    def __call__(self, x):
        return (x - self.vmin) / (self.vmax - self.vmin)

class Center:
    """Center x around 0 by subtracting mean mu."""
    def __init__(self, mu):
        self.mu = mu
    
    def __call__(self, x):
        return x - self.mu
