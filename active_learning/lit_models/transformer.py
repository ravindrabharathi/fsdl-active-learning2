import torch.nn as nn

import wandb

from .base import BaseLitModel
from .metrics import CharacterErrorRate


class TransformerLitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.

    The module must take x, y as inputs, and have a special predict() method.
    """

    def __init__(self, model, args=None):
        super().__init__(model, args)

        self.mapping = self.model.data_config["mapping"]
        inverse_mapping = {val: ind for ind, val in enumerate(self.mapping)}
        start_index = inverse_mapping["<S>"]
        end_index = inverse_mapping["<E>"]
        padding_index = inverse_mapping["<P>"]

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_index)

        ignore_tokens = [start_index, end_index, padding_index]
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    def forward(self, x):
        return self.model.predict(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("val_loss", loss, prog_bar=True)

        pred = self.model.predict(x)
        # Hide lines below until Lab 5
        pred_str = "".join(self.mapping[_] for _ in pred[0].tolist() if _ != 3)
        try:
            self.logger.experiment.log({"val_pred_examples": [wandb.Image(x[0], caption=pred_str)]})
        except AttributeError:
            pass
        # Hide lines above until Lab 5
        self.val_cer(pred, y)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        pred = self.model.predict(x)
        # Hide lines below until Lab 5
        pred_str = "".join(self.mapping[_] for _ in pred[0].tolist() if _ != 3)
        try:
            self.logger.experiment.log({"test_pred_examples": [wandb.Image(x[0], caption=pred_str)]})
        except AttributeError:
            pass
        # Hide lines above until Lab 5
        self.test_cer(pred, y)
        self.log("test_cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
