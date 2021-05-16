"""
W&B drought watch benchmark data set
"""
from pathlib import Path
import os
import shutil
import zipfile

from torchvision import transforms
import h5py
import toml

from active_learning.data.base_data_module import _download_raw_dataset, BaseDataModule, load_and_print_info
from active_learning.data.util import BaseDataset

import tensorflow as tf
import numpy as np

DEBUG_OUTPUT = True
IMG_DIM = 65
NUM_CLASSES = 4
N_TRAIN = 10000
N_VAL = 10778  # full available validation set
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
BINARY = False
RGB = False

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "droughtwatch"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "droughtwatch"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "droughtwatch"
PROCESSED_DATA_FILE_TRAINVAL = PROCESSED_DATA_DIRNAME / "trainval.h5"
PROCESSED_DATA_FILE_POOL = PROCESSED_DATA_DIRNAME / "pool.h5"


class DroughtWatch(BaseDataModule):
    """
    DataModule for W&B drought watch benchmark data set. Downloads data as ZIP in TFRecord format and converts it to PyTorch format.
    Learn more at https://wandb.ai/wandb/droughtwatch/benchmark or https://github.com/wandb/droughtwatch.
    """

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--n_train_images", type=int, default=N_TRAIN)
        parser.add_argument("--n_validation_images", type=int, default=N_VAL)
        parser.add_argument("--bands", type=str, default=",".join(BANDS))
        parser.add_argument(
            "--binary",
            action="store_true",
            help="Whether to run binary classification experiment (default is multi-class)",
        )
        parser.add_argument(
            "--rgb", action="store_true", help="Whether to include RGB channels only (default is all 11 channels)"
        )
        return parser

    def __init__(self, args=None):
        super().__init__(args)

        self.n_train_images = self.args.get("n_train_images", N_TRAIN)
        self.n_validation_images = self.args.get("n_validation_images", N_VAL)
        self.bands = self.args.get("--bands", ",".join(BANDS)).split(",")
        self.binary = self.args.get("binary", BINARY)
        self.rgb = self.args.get("rgb", RGB)

        _check_disk_dataset_availability(self)

        self.mapping = list(range(NUM_CLASSES))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, len(self.bands), IMG_DIM, IMG_DIM)  # tensor * no. bands * height * width
        self.output_dims = (1,)
        self.prepare_data()
        self.init_setup()

    def prepare_data(self, *args, **kwargs) -> None:
        _check_disk_dataset_availability(self)

    def init_setup(self):
        print("INIT SETUP DATA CALLED  \n-------------\n")

        with h5py.File(PROCESSED_DATA_FILE_TRAINVAL, "r") as f:
            self.x_train = f["x_train"][:]
            self.y_train = f["y_train"][:].squeeze().astype(int)
            self.x_val = f["x_val"][:]
            self.y_val = f["y_val"][:].squeeze().astype(int)

        with h5py.File(PROCESSED_DATA_FILE_POOL, "r") as f:
            self.x_pool = f["x_pool"][:]
            self.y_pool = f["y_pool"][:].squeeze().astype(int)

        # set random seed for reproducibility
        np.random.seed(48)

        # concatenate train and pool before shuffling
        x_all = np.concatenate([self.x_train, self.x_pool])
        y_all = np.concatenate([self.y_train, self.y_pool])

        # get initial training set length
        train_len = x_all.shape[0]
        init_len = self.n_train_images

        # get permuted array of indices
        idxs = np.random.permutation(np.arange(train_len))

        # sample initial training set
        self.x_train = x_all[idxs[:init_len]]
        self.y_train = y_all[idxs[:init_len]]

        # remaining examples become unlabelled poolF
        self.x_pool = x_all[idxs[init_len:]]
        self.y_pool = y_all[idxs[init_len:]]

        print("\nScenario:")

        # clip labels for binary classification
        if self.binary is True:
            self.y_train = np.clip(self.y_train, 0, 1)
            self.y_pool = np.clip(self.y_pool, 0, 1)
            self.y_val = np.clip(self.y_val, 0, 1)
            assert set(self.y_train) == set(self.y_pool) == set(self.y_val) == {0, 1}
            print("- Binary classification")

        else:
            assert set(self.y_train) == set(self.y_pool) == set(self.y_val) == {0, 1, 2, 3}
            print("- Multi-class classification")

        # select only RGB channels
        if self.rgb is True and self.x_train.shape[-1] == 11:
            self.x_train = self.x_train[..., (3, 2, 1)]
            self.x_pool = self.x_pool[..., (3, 2, 1)]
            self.x_val = self.x_val[..., (3, 2, 1)]
            assert self.x_train.shape[1:] == self.x_pool.shape[1:] == self.x_val.shape[1:] == (65, 65, 3)
            print("- RGB channels")

        elif self.rgb is False and self.x_train.shape[-1] == 11:
            print("- All channels")

        else:
            print(f"- Custom channel specification: {self.bands}")

        self.data_train = BaseDataset(self.x_train, self.y_train, transform=self.transform)
        self.data_val = BaseDataset(self.x_val, self.y_val, transform=self.transform)
        self.data_test = BaseDataset(self.x_pool, self.y_pool, transform=self.transform)
        self.data_unlabelled = BaseDataset(self.x_pool, self.y_pool, transform=self.transform)

        print(f"\nInitial training set size: {len(self.data_train)} - shape: {self.data_train.data.shape}")
        print(f"Initial unlabelled pool size: {len(self.data_unlabelled)} - shape: {self.data_unlabelled.data.shape}")
        print(f"Validation set size: {len(self.data_val)} - shape: {self.data_val.data.shape}\n")

    def setup(self, stage: str = None) -> None:

        self.data_train = BaseDataset(self.x_train, self.y_train, transform=self.transform)
        self.data_val = BaseDataset(self.x_val, self.y_val, transform=self.transform)
        self.data_test = BaseDataset(self.x_pool, self.y_pool, transform=self.transform)
        self.data_unlabelled = BaseDataset(self.x_pool, self.y_pool, transform=self.transform)

    def __repr__(self):
        basic = f"DroughtWatch Dataset\nDims: {self.dims}\n"
        if (
            self.data_train is None
            and self.data_val is None
            and self.data_test is None
            and self.data_unlabelled is None
        ):
            return basic

        # deepcode ignore unguarded~next~call: call to just initialized train_dataloader always returns data
        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val sizes: {len(self.data_train)}, {len(self.data_val)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), (x*1.0).mean(), (x*1.0).std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Pool size of labeled samples to do active learning from: {len(self.data_unlabelled)}\n"
        )
        return basic + data


def _check_disk_dataset_availability(self):

    if not (os.path.exists(PROCESSED_DATA_FILE_TRAINVAL) and os.path.exists(PROCESSED_DATA_FILE_POOL)):
        print("Processed DroughtWatch dataset not yet available on disk, initiating download/processing")
        _download_and_process_droughtwatch(self)

    else:
        with h5py.File(PROCESSED_DATA_FILE_TRAINVAL, "r") as f:
            n_train_samples = len(f["y_train"][:].squeeze())
            n_val_samples = len(f["y_val"][:].squeeze())

        if not (n_train_samples == self.n_train_images and n_val_samples == self.n_validation_images):
            print("Processed DroughtWatch dataset on disk does not have correct size, initiating reprocessing")
            _download_and_process_droughtwatch(self)


def _download_and_process_droughtwatch(self):

    metadata = toml.load(METADATA_FILENAME)

    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(self, metadata["filename"], DL_DATA_DIRNAME)


def _load_data(data_path):
    def file_list_from_folder(folder, data_path):
        folderpath = os.path.join(data_path, folder)
        filelist = []
        for filename in os.listdir(folderpath):
            if filename.startswith("part-") and not filename.endswith("gstmp"):
                filelist.append(os.path.join(folderpath, filename))
        return filelist

    train = file_list_from_folder("train", data_path)
    val = file_list_from_folder("val", data_path)

    return train, val


def _parse_tfrecords(self, filelist, buffer_size, include_viz=False):

    # tf record parsing function
    def _parse_(serialized_example, keylist=self.bands):

        features = {
            "B1": tf.io.FixedLenFeature([], tf.string),
            "B2": tf.io.FixedLenFeature([], tf.string),
            "B3": tf.io.FixedLenFeature([], tf.string),
            "B4": tf.io.FixedLenFeature([], tf.string),
            "B5": tf.io.FixedLenFeature([], tf.string),
            "B6": tf.io.FixedLenFeature([], tf.string),
            "B7": tf.io.FixedLenFeature([], tf.string),
            "B8": tf.io.FixedLenFeature([], tf.string),
            "B9": tf.io.FixedLenFeature([], tf.string),
            "B10": tf.io.FixedLenFeature([], tf.string),
            "B11": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }

        example = tf.io.parse_single_example(serialized_example, features)

        def getband(example_key):
            img = tf.io.decode_raw(example_key, tf.uint8)
            return tf.reshape(img[: IMG_DIM ** 2], shape=(IMG_DIM, IMG_DIM, 1))

        bandlist = [getband(example[key]) for key in keylist]

        # combine bands into tensor
        image = tf.concat(bandlist, -1)
        label = tf.cast(
            example["label"], tf.int32
        )  # NOTE: no one-hot encoding for PyTorch optimizer! (different than in TensorFlow)

        return {"image": image}, label

    # create tf dataset from filelist
    tfrecord_dataset = tf.data.TFRecordDataset(filelist)

    # convert, shuffle and batch the dataset
    tfrecord_dataset = tfrecord_dataset.map(lambda x: _parse_(x)).shuffle(90000, seed=42).batch(buffer_size)
    tfrecord_iterator = iter(tfrecord_dataset)

    return tfrecord_iterator


def _process_raw_dataset(self, filename: str, dirname: Path):

    print("Unzipping DroughtWatch file...")
    curdir = os.getcwd()
    os.chdir(dirname)

    if not os.path.exists(DL_DATA_DIRNAME / "droughtwatch_data"):
        with zipfile.ZipFile(filename, "r") as zip_file:
            zip_file.extractall()

    print("Loading train/validation datasets as TF tensor")
    train_tfrecords, val_tfrecords = _load_data(DL_DATA_DIRNAME / "droughtwatch_data")
    train_image_iterator = _parse_tfrecords(self, train_tfrecords, self.n_train_images)
    val_image_iterator = _parse_tfrecords(self, val_tfrecords, self.n_validation_images)
    train_images, train_labels = train_image_iterator.get_next()
    val_images, val_labels = val_image_iterator.get_next()

    print("Converting train/valildation TF tensors to Numpy")
    x_train = train_images["image"].numpy()
    y_train = train_labels.numpy()
    x_val = val_images["image"].numpy()
    y_val = val_labels.numpy()

    print("Saving train/validation to HDF5 in compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)

    with h5py.File(PROCESSED_DATA_FILE_TRAINVAL, "w") as f:
        f.create_dataset("x_train", data=x_train, compression="lzf")  # take only rows within train_indices
        f.create_dataset("y_train", data=y_train, compression="lzf")
        f.create_dataset("x_val", data=x_val, compression="lzf")  # take only n_validation_images rows
        f.create_dataset("y_val", data=y_val, compression="lzf")

    print("Saving all remaining labeled images to separate HDF5 pool...")
    pool_images, pool_labels = train_image_iterator.get_next()
    x_pool = pool_images["image"].numpy()
    y_pool = pool_labels.numpy()

    with h5py.File(PROCESSED_DATA_FILE_POOL, "w") as f:
        f.create_dataset("x_pool", data=x_pool, compression="gzip", maxshape=(90000, 65, 65, 11), chunks=True)
        f.create_dataset("y_pool", data=y_pool, compression="gzip", maxshape=(90000,), chunks=True)

    for pool_images, pool_labels in train_image_iterator:
        x_pool = pool_images["image"].numpy()
        y_pool = pool_labels.numpy()

        with h5py.File(PROCESSED_DATA_FILE_POOL, "a") as hf:

            hf["x_pool"].resize((hf["x_pool"].shape[0] + x_pool.shape[0]), axis=0)
            hf["x_pool"][-x_pool.shape[0] :] = x_pool

            hf["y_pool"].resize((hf["y_pool"].shape[0] + y_pool.shape[0]), axis=0)
            hf["y_pool"][-y_pool.shape[0] :] = y_pool

    print("Cleaning up...")
    shutil.rmtree("droughtwatch_data")
    os.chdir(curdir)


if __name__ == "__main__":
    load_and_print_info(DroughtWatch)
