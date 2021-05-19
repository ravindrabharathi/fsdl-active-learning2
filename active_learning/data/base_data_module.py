"""Base DataModule class."""
import argparse
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader

from active_learning import util
from active_learning.data.util import BaseDataset


def load_and_print_info(data_module_class) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    util.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")
    return filename


BATCH_SIZE = 128
NUM_WORKERS = 0
DEBUG_OUTPUT = True


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.reduced_pool = self.args.get("reduced_pool", False)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]
        self.data_unlabelled = Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        parser.add_argument(
            "--reduced_pool",
            type=bool,
            default=False,
            help="Whether to take only a fraction of the pool (allows for faster results during development)",
        )

        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def unlabelled_dataloader(self):
        return DataLoader(
            self.data_unlabelled,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def get_ds_length(self, ds_name="unlabelled"):

        if ds_name == "unlabelled":
            return len(self.data_unlabelled.data)
        elif ds_name == "train":
            return len(self.data_train.data)
        elif ds_name == "test":
            return len(self.data_test.data)
        elif ds_name == "val":
            return len(self.data_val.data)
        else:
            raise NameError("Unknown Dataset Name " + ds_name)

    def expand_training_set(self, sample_idxs):

        # get x_train, y_train
        x_train = self.data_train.data
        y_train = self.data_train.targets
        # get unlabelled set
        x_pool = self.data_unlabelled.data
        y_pool = self.data_unlabelled.targets

        # get new training examples
        x_train_new = x_pool[sample_idxs]
        y_train_new = y_pool[sample_idxs]

        # remove the new examples from the unlabelled pool
        mask = np.ones(x_pool.shape[0], bool)
        mask[sample_idxs] = False
        self.x_pool = x_pool[mask]
        self.y_pool = y_pool[mask]

        # add new examples to training set
        self.x_train = np.concatenate([x_train, x_train_new])
        self.y_train = np.concatenate([y_train, y_train_new])

        self.data_train = BaseDataset(self.x_train, self.y_train, transform=self.transform)
        self.data_test = BaseDataset(self.x_pool, self.y_pool, transform=self.transform)
        self.data_unlabelled = BaseDataset(self.x_pool, self.y_pool, transform=self.transform)
        print("New train set size", len(self.x_train))
        print("New unlabelled pool size", len(self.x_pool))

    def get_activation_scores(self, model):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # for some reason, lightning module is not yet on cuda, even if it was initialized that way --> transfer it
        model = model.to(device)

        # pytorch dataloader for batch-wise processing
        all_samples = self.unlabelled_dataloader()

        # initialize pytorch tensors to store activation scores
        out_layer_1 = torch.Tensor().to(device)
        out_layer_2 = torch.Tensor().to(device)
        out_layer_3 = torch.Tensor().to(device)

        model.eval()
        with torch.no_grad():

            # loop through batches in unlabelled pool
            for batch_features, _ in all_samples:

                # move features to device
                batch_features = batch_features.to(device)

                # extract intermediate and final activations
                out1, out2, out3 = model(batch_features, extract_intermediate_activations=True)

                # store batch results
                out_layer_1 = torch.cat([out_layer_1, out1])
                out_layer_2 = torch.cat([out_layer_2, out2])
                out_layer_3 = torch.cat([out_layer_3, out3])

        return out_layer_1, out_layer_2, out_layer_3

    def get_pool_probabilities(self, model, T=10):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # for some reason, lightning module is not yet on cuda, even if it was initialized that way --> transfer it
        model = model.to(device)

        # pytorch dataloader for batch-wise processing
        all_samples = self.unlabelled_dataloader()

        if DEBUG_OUTPUT:
            print("Processing pool of instances to generate probabilities")
            print("(Note: Based on the pool size this takes a while. Will generate debug output every 5%.)\n")
            five_percent = int(self.get_ds_length(ds_name="unlabelled") / all_samples.batch_size / 20)
            i = 0
            percentage_output = 5

        # initialize pytorch tensor to store acquisition scores
        all_outputs = torch.Tensor().to(device)

        # set model to eval (non-training) modus and enable dropout layers
        model.eval()
        _enable_dropout(model)

        if self.reduced_pool:
            print("NOTE: Reduced pool dev parameter activated, will only process first batch")
            all_samples = [next(all_samples._get_iterator())]

        # process pool of instances batch wise
        for batch_features, _ in all_samples:

            with torch.no_grad():
                outputs = torch.stack(
                    [
                        torch.softmax(model(batch_features.to(device)), dim=-1)  # probabilities from logits  # logits
                        for t in range(T)  # multiple calculations
                    ],
                    dim=-1,
                )

            all_outputs = torch.cat([all_outputs, outputs], dim=0)

            if DEBUG_OUTPUT:
                i += 1
                if i > five_percent:
                    print(f"{percentage_output}% of samples in pool processed")
                    percentage_output += 5
                    i = 0

        if DEBUG_OUTPUT:
            print("100% of samples in pool processed\n")

        return all_outputs


def _enable_dropout(model):
    for each_module in model.modules():
        if each_module.__class__.__name__.startswith("Dropout"):
            each_module.train()
