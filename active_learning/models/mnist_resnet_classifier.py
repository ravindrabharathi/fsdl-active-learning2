import argparse
from typing import Any, Dict
import torch
import torch.nn as nn
import torchvision.transforms as tt
import torchvision

PRETRAINED = False
NUM_CLASSES = 10
NUM_CHANNELS = 11
DROPOUT = True
DROPOUT_PROB = 0.5
DROPOUT_HIDDEN_DIM = 512
BINARY = False


class MNISTResnetClassifier(nn.Module):
    """Classify an image of arbitrary size through a (pretrained) ResNet network"""

    def __init__(self, data_config: Dict[str, Any] = None, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        n_classes = self.args.get("n_classes", NUM_CLASSES)
        pretrained = self.args.get("pretrained", PRETRAINED)
        self.dropout = self.args.get("dropout", DROPOUT)
        binary = self.args.get("binary", BINARY)

        self.preprocess = tt.Compose(
            [
                tt.Resize(224),
                tt.Normalize(
                    mean=33.3184 / 225, std=78.5675 / 225
                ),  # MNIST-train specific calculated via TorchMNIST(...).data.float().mean() / .std()
            ]
        )

        # override values if binary is specified
        if binary is True:
            n_classes = 2
            print("Overriding n_classes parameter, setting n_classes to 2")

        # base ResNet model
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)

        # changing the architecture of the laster layers
        # if dropout is activated, add an additional fully connected layer with dropout before the last layer
        # split classification head into different parts to extract intermediate activations
        if self.dropout:

            # first fully connected layer
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features)  # additional fc layer

            # first part of additional classification head
            self.head_part_1 = nn.Sequential(
                nn.BatchNorm1d(self.resnet.fc.in_features),  # adding batchnorm
                nn.ReLU(),  # additional nonlinearity
                nn.Dropout(DROPOUT_PROB),  # additional dropout layer
                nn.Linear(self.resnet.fc.in_features, DROPOUT_HIDDEN_DIM),  # additional fc layer
            )

            # second part of classification head
            self.head_part_2 = nn.Sequential(
                nn.BatchNorm1d(DROPOUT_HIDDEN_DIM),  # adding batchnorm
                nn.ReLU(),  # additional nonlinearity
                nn.Dropout(DROPOUT_PROB),  # additional dropout layer
                nn.Linear(DROPOUT_HIDDEN_DIM, n_classes),  # same fc layer as we had before
            )

        # otherwise just adapt no. of classes in last fully-connected layer
        else:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)

        print("Adapting first convolutional layer to only one input channel\n")
        existing_layer = self.resnet.conv1

        new_layer = nn.Conv2d(
            in_channels=1,
            out_channels=existing_layer.out_channels,
            kernel_size=existing_layer.kernel_size,
            stride=existing_layer.stride,
            padding=existing_layer.padding,
            bias=existing_layer.bias,
        )

        new_layer.weight[:, 0, :, :] = existing_layer.weight[
            :, 0, :, :
        ].clone()  # copying the weights from the old to the new layer
        new_layer.weight = nn.Parameter(new_layer.weight)

        self.resnet.conv1 = new_layer

    def forward(self, x: torch.Tensor, extract_intermediate_activations: bool = False) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor (H, W can be arbitrary, will be reshaped by reprocessing)

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """

        x = self.preprocess(x)
        x = self.resnet(x)

        if self.dropout:

            if extract_intermediate_activations:

                y = self.head_part_1(x)
                z = self.head_part_2(y)

                return x, y, z

            else:

                x = self.head_part_1(x)
                x = self.head_part_2(x)

                return x

        else:

            return x

    def add_to_argparse(parser):
        parser.add_argument("--pretrained", action="store_true", default=PRETRAINED)
        parser.add_argument("--n_classes", type=int, default=NUM_CLASSES)
        parser.add_argument("--dropout", type=bool, default=DROPOUT)
        return parser
