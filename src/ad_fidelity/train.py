"""
Train models via stratified k-fold.
"""

import numpy as np
import lightning as L
import mlflow
import argparse
import torch

# want to use multiprocessing over multithreading due to Global Interpreter Lock (GIL)
# see multiprocessing vs multithreading vs asyncio
import multiprocessing as mp

from tqdm import tqdm, trange
from ad_fidelity.model import ADCNN
from ad_fidelity.transforms import BoxCrop, RandomSagittalFlip, MinMaxNorm, Center, RandomTranslation
from ad_fidelity.data import NiftiDataset, compute_mean, compute_minmax
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.models import infer_signature

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_nifti_files(ad_path: Path, cn_path: Path):
    ad_files = list(ad_path.glob("*.nii.gz"))
    cn_files = list(cn_path.glob("*.nii.gz"))
    file_paths = np.array(ad_files + cn_files)
    labels = np.array([1] * len(ad_files) + [0] * len(cn_files))
    return file_paths, labels

def make_transforms(train_files, train_labels):
    vmin, vmax = compute_minmax(train_files)
    mu_cn = compute_mean(train_files[train_labels == 0])
    mu_cn_normed = (mu_cn - vmin) / (vmax - vmin)

    tf_crop = BoxCrop(lower=[10, 13, 5], upper=[110, 133, 105])
    tf_norm = MinMaxNorm(vmin, vmax)
    tf_center = Center(mu_cn_normed)
    tf_process = Compose([tf_norm, tf_center, tf_crop])
    tf_flip = RandomSagittalFlip()
    tf_shift = RandomTranslation(shift=5, pad=0)
    tf_augment = Compose([tf_flip, tf_shift])
    tf_train = Compose([tf_augment, tf_process])
    tf_test = Compose([tf_process])
    return tf_train, tf_test

def train(train_files, train_labels, test_files, test_labels, experiment_name="ad_fidelity",
          batch_size=16, n_epochs=50, n_hidden=64, n_channels=5, num_workers=4):
    # load data
    tf_train, tf_test = make_transforms(train_files, train_labels)
    train_ds = NiftiDataset(train_files, train_labels, augment=True, transform=tf_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_ds = NiftiDataset(test_files, test_labels, augment=True, transform=tf_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    mlflow_run = mlflow.start_run()
    mlflow.log_params(dict(batch_size=batch_size, n_epochs=n_epochs, n_channels=n_channels, n_hidden=n_hidden))
    mlflow_logger = MLFlowLogger(run_id=mlflow_run.info.run_id)
    model = ADCNN(n_channels=n_channels, n_hidden=n_hidden)
    trainer = L.Trainer(max_epochs=n_epochs, logger=mlflow_logger, log_every_n_steps=2)
    trainer.fit(model=model, train_dataloaders=train_loader)
    # not really necessary, since lightning already does that
    model.eval()
    trainer.test(model, dataloaders=test_loader)
    # TODO: add signature
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()
    return model

def skf_parallel(skf, train_kwargs):
    pool = mp.Pool(args.k)
    for train_idx, test_idx in skf.split(file_paths, labels):
        p_kwargs = dict(
            train_files=file_paths[train_idx], train_labels=labels[train_idx],
            test_files=file_paths[test_idx], test_labels=labels[test_idx],
            n_epochs=args.epochs, num_workers=args.num_workers, experiment_name=args.experiment,
            batch_size=args.batch_size, n_channels=args.n_channels, n_hidden=args.n_hidden
        )
        pool.apply_async(func=train, kwds=p_kwargs)
        # p = mp.Process(target=train, kwargs=p_kwargs)
        # procs.append(p)
    pool.close()
    pool.join()

def skf_sequential():
    pass

if __name__ == "__main__":
    # parse cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ad", type=Path, required=True, help="directory path to Alzheimer's Disease subjects")
    parser.add_argument("--cn", type=Path, required=True, help="directory path for Control Normal subjects")
    parser.add_argument("--k", type=int, default=5, help="number of folds")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--seed", type=int, default=19, help="random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loaders")
    parser.add_argument("--experiment", type=str, default="ad_fidelity", help="experiment name")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--n_hidden", type=int, default=64, help="number of hidden nodes in the classification layer")
    parser.add_argument("--n_channels", type=int, default=5, help="number of channels in the feature extractor")
    parser.add_argument("--parallel", action="store_true", help="run stratified k-fold in parallel")
    parser.add_argument("--register", action="store_true", help="register model in mlflow")
    parser.add_argument("--register_name", type=str, default="adcnn", help="model name for registering")
    parser.add_argument("--preprocess", choices=["center", "center_cn", "standardize", "norm", "flip", "shift"],
                        nargs="+", help="preprocessing steps")

    args = parser.parse_args()

    # set rng seed
    set_seed(args.seed)

    file_paths, labels = get_nifti_files(args.ad, args.cn)
    
    # create a new MLflow Experiment
    mlflow.set_experiment(args.experiment)
    
    # stratified kfold and split
    skf = StratifiedKFold(args.k, shuffle=True, random_state=args.seed)

    train_kwargs = dict(
        n_epochs=args.epochs, num_workers=args.num_workers, experiment_name=args.experiment,
        batch_size=args.batch_size, n_channels=args.n_channels, n_hidden=args.n_hidden
    )

    if args.parallel:
        pool = mp.Pool(args.k)
        for train_idx, test_idx in skf.split(file_paths, labels):
            p_kwargs = dict(
                train_files=file_paths[train_idx], train_labels=labels[train_idx],
                test_files=file_paths[test_idx], test_labels=labels[test_idx],
                n_epochs=args.epochs, num_workers=args.num_workers, experiment_name=args.experiment,
                batch_size=args.batch_size, n_channels=args.n_channels, n_hidden=args.n_hidden
            )
            pool.apply_async(func=train, kwds=p_kwargs)
            # p = mp.Process(target=train, kwargs=p_kwargs)
            # procs.append(p)
        pool.close()
        pool.join()
    else:
        for train_idx, test_idx in skf.split(file_paths, labels):
            train(
                file_paths[train_idx], labels[train_idx], file_paths[test_idx], labels[test_idx],
                n_epochs=args.epochs, num_workers=args.num_workers, experiment_name=args.experiment,
                batch_size=args.batch_size, n_channels=args.n_channels, n_hidden=args.n_hidden
            )

