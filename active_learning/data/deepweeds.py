import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 128
NUM_WORKERS = 4


class DeepweedsDataset(Dataset):
    """ Cassava Dataset """

    def __init__(self, root_dir, df, transform=None):

        self.images_dir = os.path.join(root_dir, "images")
        self.image_urls = np.asarray(df["Filename"])
        self.labels = np.asarray(df["Label"])
        self.transform = transform

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        # Get and load image
        image_path = os.path.join(self.images_dir, self.image_urls[idx])
        image = Image.open(image_path)
        # Perform transforms if any
        if self.transform:
            image = self.transform(image)
        # Get label
        label = self.labels[idx]
        return image, label


class DeepweedsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=32, help="Number of examples to operate on per forward step."
        )
        parser.add_argument("--num_workers", type=int, default=4, help="Number of additional processes to load data.")

        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {}

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def __init__(self, transform=None, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        """
        The image data and label csv file are being copied from google drive for the time being to
        ./data/deepweeds folder of the cloned repo during execution of colab notebook. Will add a way to
        download and place them in appropriate folders later on.
        """

        self.root_dir = "./data/deepweeds"
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.data_df = pd.read_csv("./data/deepweeds/labels_deep_weeds.csv")
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.init_setup()

    def init_setup(self, stage=None):
        print("INIT SETUP CALLED!!\n___________________\n")

        train_df1, self.val_df = train_test_split(self.data_df, test_size=0.2, shuffle=True, random_state=42)
        self.train_df, self.test_df = train_test_split(train_df1, test_size=0.9, shuffle=True, random_state=42)
        self.train_df = self.train_df.reset_index(drop=True)
        self.val_df = self.val_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        self.data_train = DeepweedsDataset(self.root_dir, self.train_df, self.transform)
        self.data_val = DeepweedsDataset(self.root_dir, self.val_df, self.transform)
        self.data_unlabelled = DeepweedsDataset(self.root_dir, self.test_df, self.transform)
        self.data_test = DeepweedsDataset(self.root_dir, self.test_df, self.transform)

    def setup(self, stage=None):
        self.data_train = DeepweedsDataset(self.root_dir, self.train_df, self.transform)
        self.data_val = DeepweedsDataset(self.root_dir, self.val_df, self.transform)
        self.data_unlabelled = DeepweedsDataset(self.root_dir, self.test_df, self.transform)
        self.data_test = DeepweedsDataset(self.root_dir, self.test_df, self.transform)

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
            self.data_unlabelled,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def get_ds_length(self, ds_name="unlabelled"):

        if ds_name == "unlabelled":
            return len(self.data_unlabelled)
        elif ds_name == "train":
            return len(self.data_train)
        elif ds_name == "test":
            return len(self.data_test)
        elif ds_name == "val":
            return len(self.data_val)
        else:
            raise NameError("Unknown Dataset Name " + ds_name)

    def get_num_classes():
        return 9

    # convert any float values in Label column to int
    def convert_to_int(self, x):
        return int(x)

    def expand_training_set(self, sample_idxs):

        train_df2 = self.test_df.loc[sample_idxs, :]
        self.test_df = self.test_df[~self.test_df.isin(train_df2)].dropna()
        self.train_df = pd.concat([self.train_df, train_df2], axis=0)
        self.train_df["Label"] = self.train_df.Label.map(self.convert_to_int)
        self.test_df["Label"] = self.test_df.Label.map(self.convert_to_int)
        self.test_df = self.test_df.reset_index(drop=True)
        self.train_df = self.train_df.reset_index(drop=True)

        self.data_train = DeepweedsDataset(self.root_dir, self.train_df, self.transform)
        self.data_test = DeepweedsDataset(self.root_dir, self.test_df, self.transform)
        self.data_unlabelled = DeepweedsDataset(self.root_dir, self.test_df, self.transform)
        print("New train set size", len(self.data_train))
        print("New unlabelled pool size", len(self.data_test))
