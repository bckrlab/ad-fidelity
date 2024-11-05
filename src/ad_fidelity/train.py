#!/usr/bin/env python

"""
Train models via stratified k-fold.
"""

import numpy as np
import lightning as L
import mlflow
import argparse
import torch
import datetime as dt
import json

# want to use multiprocessing over multithreading due to Global Interpreter Lock (GIL)
# see multiprocessing vs multithreading vs asyncio
#import multiprocessing as mp
import torch.multiprocessing as mp

from ad_fidelity.model import ADCNN
from ad_fidelity.data import get_nifti_files, train_test_datasets
from ad_fidelity.utils import set_seed, MLFSplitLogger, MLFModelLogger
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
        self.queue = None
        mlflow.set_experiment(args.experiment)

    def set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_train_kwargs(self, i, train_idx, test_idx):
        """Returns key word arguments for calling train."""
        train_kwargs = dict(
            train_files=self.file_paths[train_idx], train_labels=self.labels[train_idx],
            test_files=self.file_paths[test_idx], test_labels=self.labels[test_idx],
            run_name=f"{self.args.run_name}-{i}"
        )
        return train_kwargs

    def train(self, train_files, train_labels, test_files, test_labels, run_name=None):
        """One model training run."""
        mlflow.set_experiment(self.args.experiment)
        train_ds, test_ds = train_test_datasets(train_files, train_labels, test_files, test_labels)
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, persistent_workers=self.args.persistent_workers, num_workers=self.args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=self.args.batch_size, persistent_workers=self.args.persistent_workers, num_workers=self.args.num_workers)

        mlflow_run = mlflow.start_run(run_name=run_name)
        split_logger = MLFSplitLogger(mlflow_run.info.run_id)
        split_logger.log_split(train_files, train_labels, test_files, test_labels)
        mlflow.log_params(dict(batch_size=self.args.batch_size, n_epochs=self.args.epochs, lr=self.args.lr,
                               n_channels=self.args.n_channels, n_hidden=self.args.n_hidden))

        mlflow_logger = MLFlowLogger(run_id=mlflow_run.info.run_id)
        model = ADCNN(n_channels=self.args.n_channels, n_hidden=self.args.n_hidden, lr=self.args.lr)
        trainer = L.Trainer(max_epochs=self.args.epochs, logger=mlflow_logger, log_every_n_steps=2, check_val_every_n_epoch=5)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

        trainer.test(model, dataloaders=test_loader)
        auroc_display = RocCurveDisplay.from_predictions(model.test_labels, model.test_predictions[:,1], plot_chance_level=True)
        mlflow.log_figure(auroc_display.figure_, "auroc.png") 
        # TODO: add signature
        model_logger = MLFModelLogger(mlflow_run.info.run_id)
        model_logger.log_model(model)
        mlflow.end_run() 
        return dict(run_id=mlflow_run.info.run_id)

    def train_worker(self, **train_kwargs):
        """train wrapper for multiprocessing"""
        try:
            result = self.train(**train_kwargs)
            self.queue.put(result)
        except Exception as exception:
            print(exception)
    
    def skf(self):
        """Perform stratified kfold."""
        skf = StratifiedKFold(self.args.k, shuffle=True, random_state=self.args.seed)
        if self.args.parallel:
            self.skf_parallel(skf)
        else:
            self.skf_sequential(skf)
    
    def skf_parallel(self, skf):
        self.queue = mp.Queue()
        runs = []
        for i, (train_idx, test_idx) in enumerate(skf.split(self.file_paths, self.labels)):
            train_kwargs = self.get_train_kwargs(i, train_idx, test_idx)
            # when using workers in the data loaders, daemonic skf runs will result in error
            # since daemonic processes can't have children of their own
            run = mp.Process(target=self.train_worker, kwargs=train_kwargs, daemon=False)
            run.start()
            runs.append(run)
        for run in runs:
            run.join()
        results = [self.queue.get() for i in range(self.args.k)]
        self.queue.close()
        self.queue = None
        self.write_results(results)
    
    def skf_sequential(self, skf):
        results = []
        for i, (train_idx, test_idx) in enumerate(skf.split(self.file_paths, self.labels)):
            train_kwargs = self.get_train_kwargs(i, train_idx, test_idx)
            result = self.train(**train_kwargs)
            results.append(result)
        self.write_results(results)
    
    def write_results(self, results):
        """Write run ids to a json file."""
        file_path = self.args.output.with_suffix(".json")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        results_dict = {
            "run_ids": [run["run_id"] for run in results]
        }
        with open(file_path, "w") as file:
            json.dump(results_dict, file) 

def parse_train_args():
    """Parse train arguments from CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19, help="random seed")
    parser.add_argument("--experiment", type=str, default="ad_fidelity", help="experiment name")
    parser.add_argument("--run-name", type=str, default="split", help="base run names")
    parser.add_argument("--ad", type=Path, required=True, help="directory path to Alzheimer's Disease subjects")
    parser.add_argument("--cn", type=Path, required=True, help="directory path for Control Normal subjects")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--k", type=int, default=5, help="number of stratified k-folds")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers for data loaders")
    parser.add_argument("--persistent-workers", action="store_true", help="make dataloader workers persistent")
    parser.add_argument("--n_hidden", type=int, default=64, help="number of hidden nodes in the classification layer")
    parser.add_argument("--n_channels", type=int, default=5, help="number of channels in the feature extractor")
    parser.add_argument("--parallel", action="store_true", help="run stratified k-fold in parallel")
    parser.add_argument("-o", "--output", type=Path, default=None, help="where to store the output file")    

    args = parser.parse_args()

    if args.output is None:
        time_stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"run_ids/{time_stamp}.json"
        args.output = Path(file_name)
    
    return args

if __name__ == "__main__":
    args = parse_train_args()

    set_seed(args.seed)
    mp.set_start_method("spawn")

    training = Training(args)

    training.skf()
 
