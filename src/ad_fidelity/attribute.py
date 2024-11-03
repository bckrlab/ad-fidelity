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
from ad_fidelity.utils import MLFLoader
from pathlib import Path
from collections import namedtuple
from tqdm import tqdm, trange

"""
Compute attribution maps for trained model via captum.
"""

ATTRIBUTIONS = ["random", "saliency", "lrp", "ig", "ixg"]

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

def get_attributers(model, target=0, baselines=None):

    def attributer_dict(attributer, kwargs):
        return dict(attributer=attributer, kwargs=kwargs)
    
    attributers = dict()
    attributers["ig"] = attributer_dict(A.IntegratedGradients(model), dict(target=target, baselines=baselines))
    attributers["ixg"] = attributer_dict(A.InputXGradient(model), dict(target=target))
    attributers["saliency"] = attributer_dict(A.Saliency(model), dict(target=target, abs=False))
    attributers["lrp"] = attributer_dict(A.LRP(model), dict(target=target))
    # mappers["gradcam"] = attributer_dict("gradcam", A.GuidedGradCam(model), {})
    attributers["random"] = attributer_dict(RandomAttribution(model), dict(target=target))

    return attributers

class MLFRunAttribution:
    """Computes attribution maps for all test files of one mlflow run."""

    def __init__(self, run_id):
        self.run_id = run_id
        mlf_loader = MLFLoader(self.run_id)
        self.model = mlf_loader.load_model()
        self.train_ds, self.test_ds = mlf_loader.load_data()
    
    def attribute_run(self, attributer, attributer_kwargs, artifact_path):
        attributions = []
        for x, y in tqdm(self.test_ds):
            # add batch dimension
            x = x[torch.newaxis]
            a = attributer.attribute(x, **attributer_kwargs)
            attributions.append(a.clone().detach().cpu()) 
        Z = torch.stack(attributions)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir, artifact_path)
            torch.save(Z, tmp_path)
            mlflow.log_artifact(tmp_path, run_id=run_id)
        return attributions 

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

    #parser.add_argument("--model-uri")
    # parser.add_argument("-o", "--output")
    # parser.add_argument("--methods", choices=methods)
    # parser.add_argument("--baseline")

    parser.add_argument("-i", "--run-ids", type=Path, help="json file that contains the run_ids", required=True)
    parser.add_argument("-o", "--output", type=Path, default=Path("attributions"), help="output path")
    parser.add_argument("--attribution", choices=ATTRIBUTIONS, default=ATTRIBUTIONS)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_attribution_args()

    with open(args.run_ids, "r") as file:
        runs_json = json.load(file)

    run_ids = runs_json["run_ids"]

    if "lrp" in args.attribution:
        lrp_rule_hack()

    for run_id in run_ids:
        print(f"RUN ID: {run_id}")
        run_attributer = MLFRunAttribution(run_id)
        for target in range(2):
            attributers = get_attributers(run_attributer.model, target)
            for attribution_method in args.attribution:
                print(f"ATTRIBUTION: {attribution_method}")
                attributer_dict = attributers[attribution_method]
                attributer = attributer_dict["attributer"]
                attributer_kwargs = attributer_dict["kwargs"]
                artifact_path = f"{attribution_method}_{target}.pt"
                run_attributer.attribute_run(attributer, attributer_kwargs, artifact_path)


