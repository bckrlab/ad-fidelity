import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

def plot_slice(data, axis, slice_idx, norm=None, cmap="gray", attr=None, ax=None, alpha=1, scale=True, **kwargs):
    data = np.array(data)
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    vmin=data.min() if scale else None
    vmax=data.max() if scale else None
    d = np.rollaxis(np.array(data), axis, 0)[slice_idx, :, :].T
    ax.imshow(d, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax,
              norm=norm, alpha=alpha, interpolation="none", **kwargs)

def plot_overlay(ax, data, axis, slice_idx, cmap="coolwarm", norm="centered", alpha=-1, threshold=-1, **kwargs):
    data = np.array(data)
    if norm == "centered":
        norm = CenteredNorm()
    if alpha < 0:
        alpha = abs(data)
        if alpha.max() > 0:
            alpha /= alpha.max()
        if threshold >= 0:
            alpha = (alpha > threshold) * np.ones_like(alpha)
        alpha = np.rollaxis(alpha, axis, 0)[slice_idx, :, :].T
    plot_slice(data, axis, slice_idx, ax=ax, alpha=alpha, cmap=cmap, norm=norm, scale=False, **kwargs)

def plot_slices(data, x, y, z, draw_axlines=True, attr=None, alpha=1, **kwargs):
    fig, axs = plt.subplots(1, 3, figsize=(6,2), width_ratios=[145,121,121])
    axlines = [(y,z), (x,z), (x,y)]
    for i, t in enumerate([x,y,z]):
        plot_slice(data, i, t, ax=axs[i], **kwargs)
        if attr is not None:
            plot_overlay(axs[i], attr, i, t, alpha=alpha)
    if draw_axlines:
        for i, (a, b) in enumerate(axlines):
            axs[i].axvline(x=a, linestyle="--", color="lime")
            axs[i].axhline(y=b, linestyle="--", color="lime")
    plt.tight_layout()
    return fig, axs

def plot_attributions(data, attribs, names, x, y, z):
    n = len(names)
    fig, axs = plt.subplots(n, 3, figsize=(5,12))
    for i, name in enumerate(names):
        norm = None if name == "Grad-CAM" else CenteredNorm()
        cmap = "Spectral_r" if name == "Grad-CAM" else "coolwarm"
        axs[i,0].set_ylabel(name)
        for j, v in enumerate([x,y,z]):
            plot_slice(data.squeeze(), j, v, ax=axs[i, j], alpha=0.5)
            plot_overlay(axs[i, j], attribs[name].squeeze(), j, v, alpha=0.5, norm=norm, cmap=cmap)
