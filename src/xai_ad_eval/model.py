import torch
import lightning as L

class ADCNN(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
