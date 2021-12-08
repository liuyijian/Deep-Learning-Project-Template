import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import pytorch_lightning as pl

# paper:  https://openreview.net/forum?id=TVHS5Y4dNvM

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(pl.LightningModule):

    def __init__(self, dim, depth, kernel_size=8, patch_size=4, n_classes=10):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.convMixer = nn.Sequential(
            nn.Conv2d(3, self.dim, kernel_size=self.patch_size, stride=self.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(self.dim),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(self.dim, self.dim, self.kernel_size, groups=self.dim, padding='same'),
                    nn.GELU(),
                    nn.BatchNorm2d(self.dim)
                )),
                nn.Conv2d(self.dim, self.dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(self.dim)
            ) for i in range(self.depth)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dim, self.n_classes)
        )
        self.save_hyperparameters('dim', 'depth', 'kernel_size', 'patch_size', 'n_classes')

    def forward(self, x):
        out = self.convMixer(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("cross_entropy_train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("cross_entropy_val_loss", loss, prog_bar=True)
        self.log("acc_val", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("cross_entropy_test_loss", loss, prog_bar=True)
        self.log("acc_test", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        return optimizer