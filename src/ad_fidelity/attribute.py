import argparse
import captum
import nilearn.image
import torch
import mlflow
import nilearn as nil
import nilearn.datasets

"""
Compute attribution maps for trained model via captum.
"""

methods = ["saliency", "ixg", "occlusion", "lrp", "ig"]

# adni nifti image affine
target_affine = torch.tensor([
    [  -1.5,    0.,     0.,    90. ],
    [   0.,     1.5,    0.,  -126. ],
    [   0.,     0.,     1.5,  -72. ],
    [   0.,     0.,     0.,     1. ],
])

class AttributionAtlas():
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-uri")
    parser.add_argument("-o", "--output")
    parser.add_argument("--methods", choices=methods)
    parser.add_argument("--baseline")
    parser.add_argument("--")

    args = parser.parse_args()



