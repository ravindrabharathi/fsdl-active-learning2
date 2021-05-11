"""MNIST DataModule"""
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from active_learning.data.util import BaseDataset
import argparse

import torch
from torchvision.datasets import MNIST as TorchMNIST
from torchvision import transforms

from active_learning.data.base_data_module import BaseDataModule, load_and_print_info

# NOTE: temp fix until https://github.com/pytorch/vision/issues/1938 is resolved
from six.moves import urllib

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"
N_TRAIN = 2000
N_VAL = 10000
BINARY = False
DEBUG_OUTPUT = True


class MNIST(BaseDataModule):
    """
    MNIST DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """
    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--n_train_images", type=int, default=N_TRAIN)
        parser.add_argument("--n_validation_images", type=int, default=N_VAL)
        parser.add_argument("--reduced_pool", type=bool, default=False, help="Whether to take only a fraction of the pool (allows for faster results during development)")
        return parser

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.n_train_images = self.args.get("n_train_images", N_TRAIN)
        self.n_validation_images = self.args.get("n_validation_images", N_VAL)
        self.binary = self.args.get("binary", BINARY)

        self.transform = None

        self.dims = (1, 1, 64, 64) 
        self.output_dims = (1,)
        self.mapping = list(range(10))

        self.prepare_data(args)
        self.init_setup(args)

    def prepare_data(self, *args, **kwargs) -> None:
        """Download train and test MNIST data from PyTorch canonical source."""

        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)

        TorchMNIST(self.data_dir, train=True, download=True)
        TorchMNIST(self.data_dir, train=False, download=True)

    def init_setup(self, args: argparse.Namespace, stage=None) -> None:
        """Split into train, val, test and pool."""

        # load train set initially to calculate mean/std for normalization
        mnist = TorchMNIST(self.data_dir, train=True).data.float()

        # define transformation to normalize data
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=mnist.mean()/255, std=mnist.std()/255)
            ])
        
        # load MNIST train dataset with transformation
        mnist_full = TorchMNIST(self.data_dir, train=True, transform=transform)

        # randomly divide into val / train_pool
        data_val_subset, train_pool_subset = random_split(mnist_full, [self.n_validation_images, 60000-self.n_validation_images])

        # convert to DataLoaders and extract data
        data_val_x, data_val_y = next(iter(DataLoader(mnist_full, sampler=data_val_subset.indices, batch_size=len(data_val_subset.indices))))
        self.data_val = BaseDataset(data_val_x, data_val_y)
        train_pool_x, train_pool_y = next(iter(DataLoader(mnist_full, sampler=train_pool_subset.indices, batch_size=len(train_pool_subset.indices))))

        # create dataset from train_pool
        train_pool = BaseDataset(train_pool_x, train_pool_y)

        # randomly divide into train / pool
        data_train_subset, data_pool_subset = random_split(train_pool, [self.n_train_images, 60000-self.n_validation_images-self.n_train_images])

        # convert to DataLoaders and extract data
        data_train_x, data_train_y = next(iter(DataLoader(train_pool, sampler=data_train_subset.indices, batch_size=len(data_train_subset.indices))))
        self.data_train = BaseDataset(data_train_x, data_train_y)
        data_unlabelled_x, data_unlabelled_y = next(iter(DataLoader(train_pool, sampler=data_pool_subset.indices, batch_size=len(data_pool_subset.indices))))
        self.data_unlabelled = BaseDataset(data_unlabelled_x, data_unlabelled_y)

        # load test dataset
        self.data_test = TorchMNIST(self.data_dir, train=False, transform=transform)

        print(f"\nInitial training set size: {len(self.data_train)} - shape: {self.data_train.data.shape}")
        print(f"Initial unlabelled pool size: {len(self.data_unlabelled)} - shape: {self.data_unlabelled.data.shape}")
        print(f"Validation set size: {len(self.data_val)} - shape: {self.data_val.data.shape}\n")

        assert self.data_train.data.shape[1:] == torch.Size([1, 224, 224]), f"invalid data_train shape: {self.data_train.data.shape[1:]}"
        assert self.data_val.data.shape[1:] == torch.Size([1, 224, 224]), f"invalid data_val shape: {self.data_val.data.shape[1:]}"
        assert self.data_unlabelled.data.shape[1:] == torch.Size([1, 224, 224]), f"invalid data_unlabelled shape: {self.data_unlabelled.data.shape[1:]}"


    def __repr__(self):
        basic = f"MNIST Dataset\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None and self.data_unlabelled is None:
            return basic

        # deepcode ignore unguarded~next~call: call to just initialized train_dataloader always returns data
        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/Val/Pool data shapes: {self.data_train.data.shape}, {self.data_val.data.shape}, {self.data_unlabelled.data.shape}\n"
            f"Train/val sizes: {len(self.data_train)}, {len(self.data_val)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), (x*1.0).mean(), (x*1.0).std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Pool size of labeled samples to do active learning from: {len(self.data_unlabelled)}\n"
        )
        return basic + data


if __name__ == "__main__":
    load_and_print_info(MNIST)
