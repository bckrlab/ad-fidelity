import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from torchmetrics import Accuracy, AUROC, MeanMetric

class Conv3DBlock(torch.nn.Module):
    def __init__(self, conv3d_kwargs=None, pool3d_kwargs=None, batchnorm3d_kwargs=None, dropout3d_kwargs=None):
        super().__init__()
        self.conv3d = nn.Conv3d(**conv3d_kwargs)
        self.maxpool3d = nn.MaxPool3d(**pool3d_kwargs)
        self.batchnorm = nn.BatchNorm3d(**batchnorm3d_kwargs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(**dropout3d_kwargs)
    
    def forward(self, x):
        x = self.conv3d(x)
        x = self.maxpool3d(x)
        # debatable: batchnorm before or after relu?
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ADCNN(L.LightningModule):
    def __init__(self, n_channels=5, kernel_size=3, n_hidden=64, p=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # construct arguments
        conv3d_kwargs_input = dict(in_channels=1, out_channels=n_channels, kernel_size=kernel_size, padding="same")
        conv3d_kwargs = dict(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding="same")
        pool3d_kwargs = dict(kernel_size=2)
        batchnorm3d_kwargs = dict(num_features=n_channels)
        dropout3d_kwargs = dict(p=0.1)
        # feature extractor
        self.feature_extractor = nn.Sequential(
            Conv3DBlock(conv3d_kwargs_input, pool3d_kwargs, batchnorm3d_kwargs, dropout3d_kwargs),
            Conv3DBlock(conv3d_kwargs, pool3d_kwargs, batchnorm3d_kwargs, dropout3d_kwargs),
            Conv3DBlock(conv3d_kwargs, pool3d_kwargs, batchnorm3d_kwargs, dropout3d_kwargs)
        )
        # original paper by Dyrba et al. (2020) used input shape: 88x94x32
        # however, seems to already cut off brain parenchyma
        # use crop: Crop((10, 110), (13, 133), (5, 105))
        # input: 100x120x100
        # feature maps shape: (n_channels, 12, 15, 12)
        self.hidden_size = n_channels * 12 * 15 * 12
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, n_hidden),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(p=p),
            nn.Linear(n_hidden, 2)
        )
        # logging metrics
        # average: micro (samplewise), macro (compute per label and average), weighted (compute per label and weighted average)
        self.train_acc = Accuracy("binary", average="macro")
        self.test_acc = Accuracy("binary", average="macro")
        self.val_acc = Accuracy("binary", average="macro")
        self.test_auroc = AUROC(task="binary")
        self.test_predictions = None
        self.test_labels = None
        self.loss_fun = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.hidden_size)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        preds = torch.argmax(F.softmax(y_hat, dim=1), dim=1)
        loss = self.loss_fun(y_hat, y)
        # log train loss per epoch
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.train_acc(preds, y)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc)
    
    def on_test_epoch_start(self):
        self.test_predictions = []
        self.test_labels = []
        return super().on_test_epoch_start()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        p_hat = F.softmax(y_hat, dim=1)
        c_hat = torch.argmax(p_hat, dim=1)
        self.test_acc(c_hat, y)
        self.test_auroc(p_hat[:,1], y)
        self.test_labels.append(y.clone().detach().cpu())
        self.test_predictions.append(p_hat.clone().detach().cpu())

    def on_test_epoch_end(self):
        self.test_predictions = torch.concat(self.test_predictions)
        self.test_labels = torch.concat(self.test_labels)
        self.log("test_acc", self.test_acc)
        self.log("test_auroc", self.test_auroc)
     
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fun(y_hat, y)
        # in validation step, log defaults to on_epoch
        self.log("val_loss", loss)
        c_hat = torch.argmax(F.softmax(y_hat, dim=1), dim=1)
        self.val_acc(c_hat, y)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc)

