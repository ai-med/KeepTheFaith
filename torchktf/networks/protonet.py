# This file is part of Faithful Explanations in Convolutional Neural Networks for Case-Based Reasoning (KeepTheFaith).
#
# KeepTheFaith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KeepTheFaith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KeepTheFaith. If not, see <https://www.gnu.org/licenses/>.
from typing import Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchktf.networks.blocks import ConvBlock, ReLU1, ResBlock
from torchktf.networks.convnets import Network

from torchvision.models import densenet161, DenseNet161_Weights, resnet152, ResNet152_Weights, resnet50, ResNet50_Weights, vgg16, VGG16_Weights, resnext50_32x4d, ResNeXt50_32X4D_Weights, wide_resnet50_2, Wide_ResNet50_2_Weights

BLOCKMAPPER = {'ConvNet': ConvBlock, 'ResNet': ResBlock}
PROTO_NONLINEARITY_MAPPER = {'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'relu1': ReLU1}

PRETRAINED_WEIGHTS = {
        "vgg16": [vgg16, VGG16_Weights.IMAGENET1K_V1, 512],
        "resnet50": [resnet50, ResNet50_Weights.IMAGENET1K_V2, 2048],
        "resnet152": [resnet152, ResNet152_Weights.IMAGENET1K_V2, 2048],
        "densenet161": [densenet161, DenseNet161_Weights.IMAGENET1K_V1, 2208],
        "resnet50x": [resnext50_32x4d, ResNeXt50_32X4D_Weights.IMAGENET1K_V2, 2048],
        "wideresnet50": [wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V2, 2048]
        }


class ProtoNet(Network):
    def __init__(
        self,
        in_channels: int = 1,
        n_outputs: int = 3,
        n_basefilters: int = 16,
        n_blocks: int = 4,
        layer_size: int = 2,
        block_type: str = 'ConvNet',
        optim_features: bool = True,
        n_protos_per_class: int = 3,
        proto_dims: int = 64,
        proto_activation_func: Union[str, Callable] = 'linear',
        proto_nonlinearity: str = 'relu1',
    ):
        assert proto_nonlinearity in PROTO_NONLINEARITY_MAPPER.keys()
        if block_type in BLOCKMAPPER.keys():
            super().__init__(
                in_channels,
                n_outputs,
                n_basefilters,
                n_blocks,
                layer_size,
                BLOCKMAPPER[block_type],
            )
        else:
            super().__init__(
                in_channels,
                n_outputs,
                n_basefilters,
                n_blocks,
                layer_size,
                ConvBlock,
                )
            backbone, weights, n_features_out = PRETRAINED_WEIGHTS[block_type]
            self.blocks = backbone(weights=weights)
            if "resnet" in block_type:
                self.blocks = nn.Sequential(*(list(self.blocks.children())[:-2]))
            else:
                self.blocks = self.blocks.features
            self.n_filters_out = n_features_out
        self.gap = None
        self.fc = None

        if not optim_features:  # freeze encoder weights
            for x in self.blocks.parameters():
                x.requires_grad = False

        # quick n dirty save params
        self.in_channels = in_channels
        self.n_outputs = n_outputs
        self.n_basefilters = n_basefilters
        self.n_blocks = n_blocks
        self.layer_size = layer_size
        self.block_type = block_type
        self.optim_features = optim_features
        self.n_protos_per_class = n_protos_per_class
        self.proto_dims = proto_dims
        self.proto_activation_func = proto_activation_func
        self.proto_nonlinearity = proto_nonlinearity
        #
        self.in_channels
        self.prototype_activation_function = proto_activation_func
        self.n_pp_cl = n_protos_per_class
        self.n_classes = n_outputs
        self.n_protos = self.n_pp_cl * self.n_classes
        self.eps = 1e-6
        self.p_classmapping = torch.zeros(self.n_protos, self.n_classes)
        for i, row in enumerate(self.p_classmapping):
            row[i // self.n_pp_cl] = 1
        self.p_classmapping = nn.Parameter(self.p_classmapping)

        self.feat_extractor = nn.Sequential(
                nn.Conv2d(in_channels=self.n_filters_out, out_channels=proto_dims, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=proto_dims, out_channels=proto_dims, kernel_size=1),
                PROTO_NONLINEARITY_MAPPER[proto_nonlinearity](),
        )

        self.prototypes = nn.Parameter(torch.rand((self.n_protos, proto_dims, 1, 1)), requires_grad=True) # currently only init in R+

        self.ones = nn.Parameter(torch.ones(self.prototypes.size()), requires_grad=False)
        self.clf = nn.Linear(self.n_protos, self.n_classes, bias=False)

        self._initialize_weights()

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        l2square is calculated with the quadratic formulat: (a-b)^2 = a^2 - 2ab + b^2.
        || a - b ||^2_2 = sum_i (a_i - b_i)^2 = ... = sum_i a_i^2 - 2 sum_i a_i*b_i + sum_i b_i^2 = <1, a^2> - 2* <a, b> + <1, b^2>
        '''
        x2 = x ** 2  # a^2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)  # <1, a_i^2>

        p2 = self.prototypes ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3)) # b^2
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)  # b^2

        xp = F.conv2d(input=x, weight=self.prototypes)  # ab
        intermediate_result = - 2 * xp + p2_reshape  # use broadcas; -2ab + b^2
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)  # || a - b ||^2_2

        return distances


    def forward(self, x):
        conv_features = self.blocks(x)
        conv_features = self.feat_extractor(conv_features)
        distances = self._l2_convolution(conv_features)  # vectorized

        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.n_protos)
        if self.prototype_activation_function == 'log':
            prototype_activations = torch.log((min_distances + 1) / (min_distances + self.eps))
        elif self.prototype_activation_function == 'linear':
            prototype_activations = -min_distances
        else:
            prototype_activations = self.prototype_activation_function(min_distances)
        logits = self.clf(prototype_activations)
        return logits, min_distances, conv_features, distances


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.p_classmapping)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.clf.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        if self.block_type in BLOCKMAPPER.keys():
            for m in self.feat_extractor.modules():
                if isinstance(m, nn.Conv2d):
                    # every init technique has an underscore _ in the name
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def get_conv_info(self):
        strides = []
        paddings = []
        kernels = []
        for x in self.blocks.modules():
            if isinstance(x, torch.nn.Conv2d):
                assert x.kernel_size[0] == x.kernel_size[1]
                assert x.stride[0] == x.stride[1]
                assert x.padding[0] == x.padding[1]
                if x.kernel_size[0] != 1:  # downsample of ResNet. Don't include this stride twice! 
                    kernels.append(x.kernel_size[0])
                    strides.append(x.stride[0])
                    paddings.append(x.padding[0])
            elif isinstance(x, torch.nn.MaxPool2d):
                kernels.append(x.kernel_size)
                strides.append(x.stride)
                paddings.append(x.padding)
        return kernels, strides, paddings
