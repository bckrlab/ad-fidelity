import mlflow
import mlflow.artifacts
import mlflow.artifacts
import mlflow.artifacts
import torch
import numpy as np
import tempfile

from ad_fidelity.data import train_test_datasets
from pathlib import Path


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

class MLFModelLogger:
    MODEL_URI = "runs:/{run_id}/model"

    def __init__(self, run_id):
        self.run_id = run_id
    
    def get_uri(self):
        return MLFModelLogger.MODEL_URI.format(run_id=self.run_id)

    def log_model(self, model):
        mlflow.pytorch.log_model()

    def load_model(self):
        model_uri = self.get_uri()
        model = mlflow.pytorch.load_model(model_uri)
        return model
     
class MLFSplitLogger:
    SPLIT_NAME = "split.json"
    SPLIT_URI = "runs:/{run_id}/split.json"

    """Logs and loads the train test split of the data."""
    def __init__(self, run_id):
        self.run_id = run_id
    
    def get_uri(self):
        return MLFSplitLogger.SPLIT_URI.format(run_id=self.run_id)
    
    def log_split(self, train_paths, train_labels, test_paths, test_labels):
        """Returns train-test-split as dictionary."""
        split_dict = {
            "train_paths": train_paths.astype(str).tolist(),
            "train_labels": train_labels.tolist(),
            "test_paths": test_paths.astype(str).tolist(),
            "test_labels": test_labels.tolist()
        }
        mlflow.log_dict(split_dict, run_id=self.run_id)

    def load_split(self):
        split_dict = mlflow.artifacts.load_dict(self.get_uri())
        train_ds, test_ds = train_test_datasets(
            split_dict["train_files"],
            split_dict["train_labels"],
            split_dict["test_files"],
            split_dict["test_labels"],
        )
        return train_ds, test_ds

class MLFAttributionLogger:
    """Logs and loads attribution tensors."""
    ATTRIBUTION_NAME = "{attribution}_{target}.pt"
    ATTRIBUTION_URI = "runs:/{run_id}/{file_name}"

    def __init__(self, run_id):
        self.run_id = run_id
    
    def get_file_name(self, attribution, target):
        return MLFAttributionLogger.ATTRIBUTION_NAME.format(attribution=attribution, target=target)

    def get_uri(self, attribution, target):
        file_name = self.get_file_name(attribution, target)
        return MLFAttributionLogger.ATTRIBUTION_URI.format(run_id=self.run_id, file_name=file_name)
     
    def log_attributions(self, z, attribution, target):
        file_name = self.get_file_name(attribution, target)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir, file_name)
            torch.save(z, tmp_path)
            mlflow.log_artifact(tmp_path, run_id=self.run_id)

    def load_attributions(self, attribution, target):
        # will return local file path
        # won't actually download, since artifacts are already local
        attribution_uri = self.get_uri(attribution, target)
        attribution_path = mlflow.artifacts.download_artifacts(attribution_uri)
        z = torch.load(attribution_path)
        return z

