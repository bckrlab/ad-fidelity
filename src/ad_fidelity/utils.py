import mlflow
import mlflow.artifacts
import mlflow.artifacts
import torch

from ad_fidelity.data import train_test_datasets

class MLFLoader:
    MODEL_URI = "runs:/{run_id}/model"
    SPLIT_URI = "runs:/{run_id}/split.json"
    ATTRIBUTION_URI = "runs:/{run_id}/{attribution}_{target}.pt"

    def __init__(self, run_id):
        self.run_id = run_id

    def load_model(self):
        model_uri = MLFLoader.MODEL_URI.format(run_id=self.run_id)
        model = mlflow.pytorch.load_model(model_uri)
        return model
    
    def load_data(self):
        """Loads train and test datasets for a specific model run."""
        split_uri = MLFLoader.SPLIT_URI.format(run_id=self.run_id)
        split_dict = mlflow.artifacts.load_dict(split_uri)
        train_ds, test_ds = train_test_datasets(
            split_dict["train_files"],
            split_dict["train_labels"],
            split_dict["test_files"],
            split_dict["test_labels"],
        )
        return train_ds, test_ds
    
    def load_attribution(self, attribution, target):
        # will return local file path
        # won't actually download, since artifacts are already local
        attribution_uri = MLFLoader.ATTRIBUTION_URI.format(run_id=self.run_id, attribution=attribution, target=target)
        attribution_path = mlflow.artifacts.download_artifacts(attribution_uri)
        z = torch.load(attribution_path)
        return z
