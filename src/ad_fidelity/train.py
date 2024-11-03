#!/usr/bin/env python

"""
Train models via stratified k-fold.
"""

import numpy as np
import lightning as L
import mlflow
import argparse
import torch
import json

# want to use multiprocessing over multithreading due to Global Interpreter Lock (GIL)
# see multiprocessing vs multithreading vs asyncio
#import multiprocessing as mp
import torch.multiprocessing as mp

from tqdm import tqdm, trange
from ad_fidelity.model import ADCNN
from ad_fidelity.transforms import BoxCrop, RandomSagittalFlip, MinMaxNorm, Center, RandomTranslation
from ad_fidelity.data import NiftiDataset, get_nifti_files, train_test_datasets
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.models import infer_signature
from sklearn.metrics import RocCurveDisplay


class Training:  
    def __init__(self, args):
        self.args = args
        # set rng seed
        self.set_seed(self.args.seed)        
        # load nifti file paths and labels
        self.file_paths, self.labels = get_nifti_files(self.args.ad, self.args.cn) 
        # create a new MLflow Experiment
        mlflow.set_experiment(args.experiment)

    def set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def split2dict(self, train_files, train_labels, test_files, test_labels):
        """Returns train-test-split as dictionary."""
        split = {
            "train_files": train_files.astype(str).tolist(),
            "train_labels": train_labels.tolist(),
            "test_files": test_files.astype(str).tolist(),
            "test_labels": test_labels.tolist()
        }
        return split

    def train(self, train_files, train_labels, test_files, test_labels, run_name=None):
        """One model training run."""
        mlflow.set_experiment(self.args.experiment)
        train_ds, test_ds = train_test_datasets(train_files, train_labels, test_files, test_labels)
        print(train_ds.samples.shape, train_ds.labels.shape)
        print(test_ds.samples.shape, test_ds.labels.shape)
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, persistent_workers=self.args.persistent_workers, num_workers=self.args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=self.args.batch_size, persistent_workers=self.args.persistent_workers, num_workers=self.args.num_workers)

        mlflow_run = mlflow.start_run(run_name=run_name)
        split_dict = self.split2dict(train_files, train_labels, test_files, test_labels)
        mlflow.log_dict(split_dict, "split.json")
        mlflow.log_params(dict(batch_size=self.args.batch_size, n_epochs=self.args.epochs, n_channels=self.args.n_channels, n_hidden=self.args.n_hidden))

        mlflow_logger = MLFlowLogger(run_id=mlflow_run.info.run_id)
        model = ADCNN(n_channels=self.args.n_channels, n_hidden=self.args.n_hidden)
        trainer = L.Trainer(max_epochs=self.args.epochs, logger=mlflow_logger, log_every_n_steps=2, check_val_every_n_epoch=5)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

        # not really necessary, since lightning already does that
        trainer.test(model, dataloaders=test_loader)
        auroc_display = RocCurveDisplay.from_predictions(model.test_labels, model.test_predictions[:,1], plot_chance_level=True)
        mlflow.log_figure(auroc_display.figure_, "auroc.png") 
        # TODO: add signature
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run() 
        return model
    
    def auroc_curve(self, model, dataset):
        predictions = []
        labels = []
        with torch.no_grad():
            for x, y in tqdm(dataset):
                y_hat = model(x).clone().detach()
                predictions.append(y_hat)
                labels.append(y)
            RocCurveDisplay.from_predictions(labels, predictions)


    def skf(self):
        """Perform stratified kfold."""
        skf = StratifiedKFold(self.args.k, shuffle=True, random_state=self.args.seed)
        if self.args.parallel:
            self.skf_parallel(skf)
        else:
            self.skf_sequential(skf)
    
    def skf_parallel(self, skf):
        # pool = mp.Pool(self.args.k)
        runs = []
        for i, (train_idx, test_idx) in enumerate(skf.split(self.file_paths, self.labels)):
            train_kwargs = self.get_train_kwargs(i, train_idx, test_idx)
            # when using workers in the data loaders, daemonic skf runs will result in error
            # since daemonic processes can't have children of their own
            run = mp.Process(target=self.train, kwargs=train_kwargs, daemon=False)
            run.start()
            runs.append(run)
            #pool.apply_async(func=self.train, kwds=train_kwargs)
        for run in runs:
            run.join()
        # pool.close()
        # pool.join()
    
    def skf_sequential(self, skf):
        for i, (train_idx, test_idx) in enumerate(skf.split(self.file_paths, self.labels)):
            train_kwargs = self.get_train_kwargs(i, train_idx, test_idx)
            self.train(**train_kwargs)
 
    def get_train_kwargs(self, i, train_idx, test_idx):
        """Returns key word arguments for calling train."""
        train_kwargs = dict(
            train_files=self.file_paths[train_idx], train_labels=self.labels[train_idx],
            test_files=self.file_paths[test_idx], test_labels=self.labels[test_idx],
            run_name=f"{self.args.run_name}-{i}"
        )
        return train_kwargs


def parse_train_args():
    """Parse train arguments from CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ad", type=Path, required=True, help="directory path to Alzheimer's Disease subjects")
    parser.add_argument("--cn", type=Path, required=True, help="directory path for Control Normal subjects")
    parser.add_argument("--k", type=int, default=5, help="number of stratified k-folds")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--seed", type=int, default=19, help="random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers for data loaders")
    parser.add_argument("--persistent-workers", action="store_true", help="make dataloader workers persistent")
    parser.add_argument("--experiment", type=str, default="ad_fidelity", help="experiment name")
    parser.add_argument("--run-name", type=str, default="split", help="base run names")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--n_hidden", type=int, default=64, help="number of hidden nodes in the classification layer")
    parser.add_argument("--n_channels", type=int, default=5, help="number of channels in the feature extractor")
    parser.add_argument("--parallel", action="store_true", help="run stratified k-fold in parallel")
    parser.add_argument("--register", action="store_true", help="register model in mlflow")
    parser.add_argument("--register_name", type=str, default="adcnn", help="model name for registering")
    parser.add_argument("--preprocess", choices=["center", "center_cn", "standardize", "norm", "flip", "shift"],
                        nargs="+", help="preprocessing steps")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_train_args()

    mp.set_start_method("spawn")

    training = Training(args)

    training.skf()
 
