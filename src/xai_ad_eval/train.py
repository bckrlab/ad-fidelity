import numpy as np
import lightning as L
import mlflow
import argparse
import torch

from tqdm import tqdm, trange
from xai_ad_eval.model import ADCNN
from xai_ad_eval.transforms import BoxCrop, RandomSagittalFlip, MinMaxNorm, Center, RandomTranslation
from xai_ad_eval.data import NiftiDataset, compute_mean, compute_minmax
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.models import infer_signature

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

def train(train_files, train_labels, test_files, test_labels, n_epochs=50, num_workers=4):
    # load data
    tf_train, tf_test = make_transforms(train_files, train_labels)
    train_ds = NiftiDataset(train_files, train_labels, augment=True, transform=tf_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=num_workers)
    test_ds = NiftiDataset(test_files, test_labels, augment=True, transform=tf_test)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=num_workers)

    mlflow.start_run()
    mlflow_logger = MLFlowLogger()
    model = ADCNN()
    trainer = L.Trainer(max_epochs=n_epochs, logger=mlflow_logger, log_every_n_steps=2)
    trainer.fit(model=model, train_dataloaders=train_loader)
    model.eval()
    trainer.test(model, dataloaders=test_loader)
    # save model
    # TODO: add signature
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()
    return model


if __name__ == "__main__":
    # parse cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ad", type=Path, required=True, help="directory path to Alzheimer's Disease subjects")
    parser.add_argument("--cn", type=Path, required=True, help="directory path for Control Normal subjects")
    parser.add_argument("--k", type=int, default=5, help="number of folds")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--seed", default=19, help="random seed")
    parser.add_argument("--num_workers", default=4, help="number of workers for data loaders")

    args = parser.parse_args()

    # set rng seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # search nifti files
    ad_files = list(args.ad.glob("*.nii.gz"))
    cn_files = list(args.cn.glob("*.nii.gz"))

    files = np.array(ad_files + cn_files)
    labels = np.array([1] * len(ad_files) + [0] * len(cn_files))
    
    # create a new MLflow Experiment
    mlflow.set_experiment("xai_ad_eval")
    
    # stratified kfold and split
    skf = StratifiedKFold(args.k, shuffle=True, random_state=args.seed)

    for train_idx, test_idx in skf.split(files, labels):
        train(
            files[train_idx], labels[train_idx], files[test_idx], labels[test_idx],
            n_epochs=args.epochs, num_workers=args.num_workers
        )

