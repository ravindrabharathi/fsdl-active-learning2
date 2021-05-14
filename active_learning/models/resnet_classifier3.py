import argparse
from typing import Any, Dict
import torch
import torch.nn as nn
import torchvision.transforms as tt
import torchvision
import warnings

PRETRAINED = True
NUM_CLASSES = 9
NUM_CHANNELS = 3
DROPOUT = False
DROPOUT_PROB = 0.5
DROPOUT_HIDDEN_DIM = 512
BINARY = False
RGB = True

class ResnetClassifier3(nn.Module):
    """Classify an image of arbitrary size through a (pretrained) ResNet network"""

    def __init__(self, data_config: Dict[str, Any] = None, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        n_channels = self.args.get("n_channels", NUM_CHANNELS)
        n_classes = self.args.get("n_classes", NUM_CLASSES)
        pretrained = self.args.get("pretrained", PRETRAINED)
        self.dropout = self.args.get("dropout", DROPOUT)
        binary = self.args.get("binary", BINARY)
        rgb = True #self.args.get("rgb", RGB)

        # override values if binary or rgb are specified
        if binary == True:
            n_classes = 2
            print("Overriding n_classes parameter, setting n_classes to 2")
        if rgb == True:
            n_channels = 3
            print("setting n_channels to 3\n")

        # base ResNet model
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)

        for param in self.resnet.parameters():
            param.requires_grad = False

        # changing the architecture of the laster layers
        # if dropout is activated, add an additional fully connected layer with dropout before the last layer
        # split classification head into different parts to extract intermediate activations
        if self.dropout:    
        
            # first fully connected layer
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features) # additional fc layer

            # first part of additional classification head
            self.head_part_1 = nn.Sequential(
                nn.BatchNorm1d(self.resnet.fc.in_features), # adding batchnorm
                nn.ReLU(), # additional nonlinearity
                nn.Dropout(DROPOUT_PROB), # additional dropout layer
                nn.Linear(self.resnet.fc.in_features, DROPOUT_HIDDEN_DIM) # additional fc layer
            )

            # second part of classification head
            self.head_part_2 = nn.Sequential(
                nn.BatchNorm1d(DROPOUT_HIDDEN_DIM), # adding batchnorm
                nn.ReLU(), # additional nonlinearity
                nn.Dropout(DROPOUT_PROB), # additional dropout layer
                nn.Linear(DROPOUT_HIDDEN_DIM, n_classes) # same fc layer as we had before
            )

        # otherwise just adapt no. of classes in last fully-connected layer
        else:
            self.resnet.fc = nn.Sequential(
                                nn.Linear(self.resnet.fc.in_features,self.resnet.fc.in_features),
                                nn.BatchNorm1d(self.resnet.fc.in_features),
                                nn.ReLU(),

                                nn.Linear(self.resnet.fc.in_features,n_classes)
                                )

        

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
        if self.dropout:
        
            if extract_intermediate_activations:

                x = self.preprocess(x)
                x = self.resnet(x)
                y = self.head_part_1(x)
                z = self.head_part_2(y)

                return x, y, z

            else:
                #print(x.shape)

                #x = self.preprocess(x)
                x = self.resnet(x)
                x = self.head_part_1(x)
                x = self.head_part_2(x)

                return x
            
        else:

            x = x.float()
            #x = self.preprocess(x)
            x = self.resnet(x)

            return x

    def get_num_classes(self):
        return NUM_CLASSES

    def add_to_argparse(parser):
        parser.add_argument("--pretrained", type=bool, default=PRETRAINED)
        parser.add_argument("--n_classes", type=int, default=NUM_CLASSES)
        parser.add_argument("--n_channels", type=int, default=NUM_CHANNELS)
        parser.add_argument("--dropout", type=bool, default=DROPOUT)
        return parser
