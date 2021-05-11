"""MNIST DataModule"""
from active_learning.data.util import BaseDataset, split_dataset
import argparse

from torchvision.datasets import MNIST as TorchMNIST
from torchvision import transforms

from active_learning.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"

# NOTE: temp fix until https://github.com/pytorch/vision/issues/1938 is resolved
from six.moves import urllib  # pylint: disable=wrong-import-position, wrong-import-order


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

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.dims = (1, 64, 64) 
        self.output_dims = (1,)
        self.mapping = list(range(10))

        self.prepare_data(args)
        self.setup()

    def prepare_data(self, *args, **kwargs) -> None:
        """Download train and test MNIST data from PyTorch canonical source."""

        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)

        TorchMNIST(self.data_dir, train=True, download=True)
        TorchMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        """Split into train, val, test and pool."""
        mnist_full = TorchMNIST(self.data_dir, train=True, transform=self.transform) # transform=self.transform
        mnist_full = BaseDataset(mnist_full.data.float().unsqueeze(0), mnist_full.targets)

        val_fraction = self.n_validation_images/60000
        pool_fraction = (60000-self.n_validation_images-self.n_train_images)/(60000-self.n_validation_images)

        self.data_val, train_and_pool = split_dataset(mnist_full, val_fraction, seed = 43)

        self.data_train, self.data_unlabelled = split_dataset(train_and_pool, pool_fraction, seed = 43)

        data_test_mnist = TorchMNIST(self.data_dir, train=False, transform=self.transform) # transform=self.transform


        #self.data_train = BaseDataModule(data_train_mnist, data_train_mnist.targets)
        #self.data_val = BaseDataModule(data_val_mnist.data, data_val_mnist.targets)
        self.data_test = BaseDataset(data_test_mnist.data.float().unsqueeze(0), data_test_mnist.targets)
        #self.data_unlabelled = BaseDataModule(data_unlabelled_mnist.data, data_unlabelled_mnist.targets)

        print(f"\nInitial training set size: {len(self.data_train)}")
        print(f"Initial unlabelled pool size: {len(self.data_unlabelled)}")
        print(f"Validation set size: {len(self.data_val)}\n")

    def __repr__(self):
        basic = f"MNIST Dataset\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None and self.data_unlabelled is None:
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

'''
    def get_ds_length(self ,ds_name='unlabelled'):
    
        if ds_name=='unlabelled':
            return len(self.data_unlabelled.data)
        elif ds_name=='train':
            return len(self.data_train.data)
        elif ds_name=='test' :
            return len(self.data_test.data)
        elif ds_name=='val' :
            return len(self.data_val.data)
        else:
            raise NameError('Unknown Dataset Name '+ds_name) 
           

    def expand_training_set(self, sample_idxs):

        #get x_train, y_train
        x_train=self.data_train.data
        y_train=self.data_train.targets
        #get unlabelled set
        x_pool=self.data_unlabelled.data
        y_pool=self.data_unlabelled.targets

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
        self.data_unlabelled=BaseDataset(self.x_pool, self.y_pool, transform=self.transform)
        print('New train set size', len(self.x_train))
        print('New unlabelled pool size', len(self.x_pool))


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
                        torch.softmax( # probabilities from logits
                            model(batch_features.to(device)), dim=-1) # logits
                        for t in range(T) # multiple calculations
                    ]
                , dim = -1)

            all_outputs = torch.cat([all_outputs, outputs], dim = 0)

            if DEBUG_OUTPUT:
                i += 1
                if i > five_percent:
                    print(f"{percentage_output}% of samples in pool processed")
                    percentage_output += 5
                    i = 0

        if DEBUG_OUTPUT:
            print(f"100% of samples in pool processed\n")
        
        return all_outputs

def _enable_dropout(model):
    for each_module in model.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
'''


if __name__ == "__main__":
    load_and_print_info(MNIST)
