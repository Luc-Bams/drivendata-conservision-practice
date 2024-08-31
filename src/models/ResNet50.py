import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torchmetrics.functional import accuracy


class ResNet50(pl.LightningModule):
    def __init__(self, num_classes: int = 8, lr: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        # self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        # self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()

        self.num_classes = num_classes
        self.classifier = nn.Linear(num_filters, self.num_classes)

    def forward(self, x):
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)

        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"train_acc": acc, "train_loss": loss}
        loss = metrics["train_loss"]
        self.log_dict(metrics, prog_bar=False, on_step=True, on_epoch=True),
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=False, on_step=True, on_epoch=True)
        return metrics

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        id = batch["image_id"]
        y_hat = self(x)
        return id, y_hat

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        softmax_y_hat = F.softmax(y_hat, dim=1)
        argmax_y = torch.argmax(y, dim=1)
        acc = accuracy(
            softmax_y_hat, argmax_y, task="multiclass", num_classes=self.num_classes
        )
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=1e-5
        )
        return optimizer
