import argparse
import captum
import captum.attr as A
import mlflow.artifacts
import nilearn.image
import torch
import mlflow
import nilearn as nil
import nilearn.datasets
import json
import tempfile
import lightning as L
import torchmetrics
import torchmetrics.classification

from ad_fidelity.data import train_test_datasets
from ad_fidelity.utils import set_seed, MLFAttributionLogger, MLFModelLogger, MLFSplitLogger
from pathlib import Path
from collections import namedtuple
from tqdm import tqdm, trange

"""
Compute attribution maps for trained model via captum.
"""

ATTRIBUTIONS = ["random", "occlusion", "gradcam", "saliency", "deconv", "backprob", "deeplift", "deepshap", "lrp", "ig", "ixg"]

def lrp_rule_hack():
    """Monkey patch for captum's LRP to support more torch Modules."""
    import captum.attr._core.lrp as lrp
    # captum's LRP only supports conv2d and batchnorm2d
    # could either add rules to model, or modify supported layer types
    lrp.SUPPORTED_LAYERS_WITH_RULES[torch.nn.Conv3d] = lrp.EpsilonRule
    lrp.SUPPORTED_LAYERS_WITH_RULES[torch.nn.BatchNorm3d] = lrp.EpsilonRule
    lrp.SUPPORTED_LAYERS_WITH_RULES[torch.nn.Dropout3d] = lrp.EpsilonRule

    # since we are using torch lightning, we also need to add rules for the loss function...
    lrp.SUPPORTED_LAYERS_WITH_RULES[torch.nn.CrossEntropyLoss] = lrp.EpsilonRule

    # and even for lightning metrics, although they don't contribute to classifcation...
    lrp.SUPPORTED_LAYERS_WITH_RULES[torchmetrics.classification.BinaryAccuracy] = lrp.EpsilonRule
    lrp.SUPPORTED_LAYERS_WITH_RULES[torchmetrics.classification.BinaryAUROC] = lrp.EpsilonRule


class RandomAttribution(A.Attribution):
    """Random Attribution Dummy."""

    def attribute(self, inputs, **kwargs):
        return torch.randn_like(inputs)

def get_attributers(model, target=0, baselines=None, ds_baselines=None):

    def d(attributer, kwargs):
        return dict(attributer=attributer, kwargs=kwargs)
    
    attributers = dict()
    attributers["occlusion"] = d(A.Occlusion(model), dict(target=target, baselines=baselines, sliding_window_shapes=(1,20,20,20), strides=10))
    attributers["backprob"] = d(A.GuidedBackprop(model), dict(target=target))
    attributers["deconv"] = d(A.Deconvolution(model), dict(target=target))
    attributers["deeplift"] = d(A.DeepLift(model), dict(target=target, baselines=baselines))
    attributers["deepshap"] = d(A.DeepLiftShap(model), dict(target=target, baselines=ds_baselines))
    attributers["ig"] = d(A.IntegratedGradients(model), dict(target=target, baselines=baselines))
    attributers["ixg"] = d(A.InputXGradient(model), dict(target=target))
    attributers["saliency"] = d(A.Saliency(model), dict(target=target, abs=False))
    attributers["lrp"] = d(A.LRP(model), dict(target=target))
    attributers["gradcam"] = d(A.GuidedGradCam(model, model.feature_extractor[2]), dict(target=target))
    attributers["random"] = d(RandomAttribution(model), dict(target=target))

    return attributers

class MLFRunAttribution:
    """Computes attribution maps for all test files of one mlflow run."""

    def __init__(self, run_id):
        self.run_id = run_id
        model_logger = MLFModelLogger(self.run_id)
        self.model = model_logger.load_model()
        split_logger = MLFSplitLogger(self.run_id)
        self.train_ds, self.test_ds = split_logger.load_split()
    
    def attribute_run(self, attributer, attributer_kwargs, attribution, target):
        Z = []
        for x, y in tqdm(self.test_ds):
            # add batch dimension
            x = x[torch.newaxis]
            z = attributer.attribute(x, **attributer_kwargs)
            Z.append(z.clone().detach().cpu()) 
        Z = torch.stack(Z)
        attr_logger = MLFAttributionLogger(self.run_id)
        attr_logger.log_attributions(Z, attribution, target)
        return Z 

class AttributionAtlas:
    """Maps attributed relevance to ROIs."""

    def __init__(self, affine, transform):
        """
        :param affine: target affine for resampling the atlas
        :param transform: list of transforms to apply to the atlas map, like cropping
        """
        self.atlas = nilearn.datasets.fetch_atlas_aal()
        self.roi_img = nilearn.image.resample_img(self.atlas.maps, affine, interpolation="nearest")
        self.roi_map = self.roi_img.get_fdata()
        if transform:
            self.roi_map = transform(self.roi_map)
    
    def attribute_rois(self, attribution):
        roi_scores = dict() 
        for roi, roi_idx in zip(self.atlas["labels"], self.atlas["indices"]):
            roi_mask = (self.roi_map == int(roi_idx))
            roi_size = roi_mask.sum()
            roi_abs = torch.where(roi_mask, attribution, 0).sum()
            roi_rel = roi_abs / roi_rel
            roi_score = dict(size=roi_size, abs=roi_abs, rel=roi_rel)
            roi_scores[roi] = roi_score
        return roi_scores

def parse_attribution_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=19, help="random seed")
    parser.add_argument("-i", "--run-ids", type=Path, help="json file that contains the run_ids", required=True)
    parser.add_argument("--attribution", choices=ATTRIBUTIONS, default=ATTRIBUTIONS)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_attribution_args()
    set_seed(args.seed)

    with open(args.run_ids, "r") as file:
        runs_json = json.load(file)

    run_ids = runs_json["run_ids"]

    if "lrp" in args.attribution:
        lrp_rule_hack()
    
    ds_baselines = torch.randn(3, 1, 100, 120, 100)

    for run_id in run_ids:
        print(f"RUN ID: {run_id}")
        run_attributer = MLFRunAttribution(run_id)
        for target in range(2):
            attributers = get_attributers(run_attributer.model, target, ds_baselines=ds_baselines)
            for attribution_method in args.attribution:
                print(f"ATTRIBUTION: {attribution_method}")
                attributer_dict = attributers[attribution_method]
                attributer = attributer_dict["attributer"]
                attributer_kwargs = attributer_dict["kwargs"]
                run_attributer.attribute_run(attributer, attributer_kwargs, attribution_method, target)

