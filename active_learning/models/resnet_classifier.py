import argparse
from typing import Any, Dict
import torch
import torch.nn as nn
import torchvision.transforms as tt
import torchvision

PRETRAINED = True
NUM_CLASSES = 4
NUM_CHANNELS = 11

class ResnetClassifier(nn.Module):
    """Classify an image of arbitrary size through a (pretrained) ResNet network"""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        n_channels = self.args.get("n_channels", NUM_CHANNELS)
        n_classes = self.args.get("n_classes", NUM_CLASSES)
        pretrained = self.args.get("pretrained", PRETRAINED)

        # base ResNet model
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)

        # preprocessing steps to resize images (adapted from https://pytorch.org/hub/pytorch_vision_resnet/)
        new_channel_mean = 0.485 # from existing channel 0
        new_channel_std = 0.229 # from existing channel 0
        self.preprocess = tt.Compose([
            tt.Resize(224),
            tt.Normalize(
              mean=[0.485, 0.456, 0.406, 
                new_channel_mean, new_channel_mean, new_channel_mean, new_channel_mean, new_channel_mean, new_channel_mean, new_channel_mean, new_channel_mean], 
              std=[0.229, 0.224, 0.225, 
                new_channel_std, new_channel_std, new_channel_std, new_channel_std, new_channel_std, new_channel_std, new_channel_std, new_channel_std]),
        ])

        

        # adapting the no. of input channels to the first conv layer 
        # (adapted from https://discuss.pytorch.org/t/how-to-modify-the-input-channels-of-a-resnet-model/2623/10)
        existing_layer = self.resnet.conv1

        new_layer = nn.Conv2d(in_channels=n_channels, 
                        out_channels=existing_layer.out_channels, 
                        kernel_size=existing_layer.kernel_size, 
                        stride=existing_layer.stride, 
                        padding=existing_layer.padding,
                        bias=existing_layer.bias)


        new_layer.weight[:, :existing_layer.in_channels, :, :] = existing_layer.weight.clone() # copying the weights from the old to the new layer
        
        copy_weights = 0 # take channel 0 weights to initialize new ones
        for i in range(n_channels - existing_layer.in_channels): # copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
            channel = existing_layer.in_channels + i
            new_layer.weight[:, channel:channel+1, :, :] = existing_layer.weight[:, copy_weights:copy_weights+1, :, :].clone()

        new_layer.weight = nn.Parameter(new_layer.weight)

        self.resnet.conv1 = new_layer

        for param in self.resnet.parameters():
            param.requires_grad = False

        # adapting the no. of output classes in the model's fully-connected layer
        #self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes) 
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features,self.resnet.fc.in_features),
            nn.BatchNorm1d(self.resnet.fc.in_features),
            nn.ReLU(),

            nn.Linear(self.resnet.fc.in_features,n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return x

    def add_to_argparse(parser):
        parser.add_argument("--pretrained", type=bool, default=PRETRAINED)
        parser.add_argument("--n_classes", type=int, default=NUM_CLASSES)
        parser.add_argument("--n_channels", type=int, default=NUM_CHANNELS)
        return parser
