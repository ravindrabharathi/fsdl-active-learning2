import argparse
import pytorch_lightning as pl
import torch
import numpy as np

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100

class MaxAccuracyLogger(pl.callbacks.Callback):
    """W&B does not yet provide the possibility to visualize the maximum (instead of last) logged metric, see
    https://github.com/wandb/client/issues/736. This class helps us to keep track of the best accuracies over 
    multiple epochs by always logging the maximum up to now.
    """

    train_acc_max = 0
    val_acc_max = 0
    train_f1_max = 0
    val_f1_max = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        # hook is also called after initial "test" validation before any training, but does not contain our relevant metrics there
        if "train_acc" in metrics:
            self.train_acc_max = max(self.train_acc_max, metrics["train_acc"].item())
            self.val_acc_max = max(self.val_acc_max, metrics["val_acc"].item())
            self.train_f1_max = max(self.train_f1_max, metrics["train_f1"].item())
            self.val_f1_max = max(self.val_f1_max, metrics["val_f1"].item())
            train_size = trainer.datamodule.get_ds_length('train')

            pl_module.log("train_acc_max", self.train_acc_max)
            pl_module.log("val_acc_max", self.val_acc_max)
            pl_module.log("train_f1_max", self.train_f1_max)
            pl_module.log("val_f1_max", self.val_f1_max)
            pl_module.log("train_size", train_size)


class Accuracy(pl.metrics.Accuracy):
    """Accuracy Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)


class F1_Score(pl.metrics.F1):
    """F1-Score Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn.functional, loss)
        '''
        if loss not in ("ctc", "transformer"):
            self.loss_fn = getattr(torch.nn.functional, loss)
        '''    

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)
        
        binary = self.args.get("binary", False)
        if binary:
            num_classes = 2
        else:
            num_classes = 4

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_f1 = F1_Score(num_classes=num_classes, average='macro')
        self.val_f1 = F1_Score(num_classes=num_classes, average='macro')
        self.test_f1 = F1_Score(num_classes=num_classes, average='macro')

        self.predictions = np.array([])
        self.train_size = 0
        self.val_predictions = torch.Tensor()
        self.logging = True

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=-1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)

        if self.logging:
            self.log("train_loss", loss,on_step=False, on_epoch=True,prog_bar=False)
            self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_size", self.trainer.datamodule.get_ds_length('train'), on_step=False, on_epoch=True, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        #print('validating ')
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=-1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)

        if self.logging:
            self.log("val_loss", loss, on_step=False, on_epoch=True,prog_bar=False)
            self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_size", self.trainer.datamodule.get_ds_length('train'), on_step=False, on_epoch=True, prog_bar=False)

        # store validation predictions
        if len(self.val_predictions) in [0, 10778]:
            self.val_predictions = logits.detach()
        else:
            self.val_predictions = torch.cat([self.val_predictions, logits.detach()])

    def reset_predictions(self):
        print('\nResetting Predictions\n')
        self.predictions=np.array([]) 
         

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=-1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        
        if self.predictions.shape[0]==0:
            self.predictions=preds.cpu().detach().numpy()
        else:  
            self.predictions=np.vstack([self.predictions,preds.cpu().detach().numpy()])
        
        if self.logging:
            self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=False)
            self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_size", self.trainer.datamodule.get_ds_length('train'), on_step=False, on_epoch=True, prog_bar=False)
