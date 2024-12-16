#!/usr/bin/env python

"""
Compute the Fidelity Metric
"""

import argparse
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from torch.types import Tensor
from typing import Union
from pathlib import Path
from ad_fidelity.logging import MLFModelLogger, MLFAttributionLogger, MLFSplitLogger
from ad_fidelity.utils import set_seed

def mask_topk(x: Tensor, attributions: Tensor, k: int, mask: Union[Tensor, float]=0):
    """Mask inputs with top-k relevance according to attributions."""
    assert x.shape == attributions.shape
    x = x.clone().detach()
    # topk returns values and indices
    _, topk = attributions.flatten().topk(k)
    # reshape topk indices to original value of x
    topk = torch.unravel_index(topk, attributions.shape)
    if isinstance(mask, float):
        x[topk] = mask
    else:
        x[topk] = mask[topk]
    return x

class FidelityResult:
    def __init__(self, masked_abs, masked_rel, scores):
        self.masked_abs = masked_abs
        self.masked_rel = masked_rel
        self.scores = scores

    def area_under_curve(self):
        """Approximate integral"""
        auc = torch.trapezoid(y=self.scores, x=self.masked_rel)
        return auc

    def plot(self, ax=None, fill=True, use_abs=False):
        if ax is None:
            fig, ax = plt.subplots()
        x = self.masked_abs if use_abs else self.masked_rel 
        if fill:
            ax.fill_between(x, self.scores, alpha=0.5)
        ax.plot(x, self.scores)
        ax.set_ylabel("score")
        ax.set_xlabel("masked")
        return ax


class FidelityMetric:
    def __init__(self, model):
        self.model = model
     
    def compute(self, x: Tensor, attributions: Tensor, mask: Tensor, target, steps : int = 100, min_masked: float = 0, max_masked: float = 1.0):
        """
        Compute the fidelity metric.
        
        :param x: input sample
        :param attributions: relevance map with the same shape as x
        :param target: target class
        :param mask: how to replace masked inputs
        :param step_mask: how many voxels to mask per step 
        :param max_mask: maximum mask amount
        """
        assert x.shape == attributions.shape
        # sort attribution indices by values
        top = torch.argsort(attributions.flatten(), descending=True)
        top = torch.unravel_index(top, attributions.shape)
        top = torch.stack(top)
        masked_rel = torch.linspace(min_masked, max_masked, steps)
        masked_abs = int(masked_rel * torch.numel(x))
        scores = torch.full(steps, torch.nan)
        for i in trange(steps):
            k = masked_abs[i]
            topk = tuple(top[:, :k])
            z = x.clone().detach()
            z[topk] = mask[topk]
            with torch.no_grad():
                y = self.model(z)
            scores[i] = y[target]
        return FidelityResult(masked_rel, masked_abs, scores)


class MLFRunFidelity:
    """Computes mean test fidelity of attributions for one mlflow run."""

    def __init__(self, run_id):
        self.run_id = run_id
        model_logger = MLFModelLogger(self.run_id)
        self.model = model_logger.load_model()
        split_logger = MLFSplitLogger(self.run_id)
        self.test_ds, self.train_ds = split_logger.load_split()
        self.attribution_logger = MLFAttributionLogger(self.run_id)
    
    def compute_fidelity(self, model, label, target, attribution):
        metric = FidelityMetric(model)
        Z = self.attribution_logger.load_attributions(attribution, target)
        fids = []
        for i, (x, y) in enumerate(self.test_ds):
            if y != label:
                continue
            else:
                z = Z[i]
                mask = torch.zeros_like(x)
                fid = metric.compute(x, z, mask, target, max_masked=0.05)
                fids.append(fid)
        scores = [fid.scores for fid in fids]
        aucs = [fid.area_under_curve() for fid in fids]
        mean_scores = torch.stack(scores).mean(0)
        return dict(scores=mean_scores, aucs=aucs)
    

# for each run
# load model and data and attributions
# for each ad sample in dataset
# compute fidelity metric for each attribution map
# 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=19, help="random seed")
    parser.add_argument("-i", "--run-ids", type=Path, help="json file that contains the run_ids", required=True)
    parser.add_argument("--attribution", choices=ATTRIBUTIONS, default=ATTRIBUTIONS)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)

