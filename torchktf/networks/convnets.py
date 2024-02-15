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
from collections import OrderedDict
from typing import Any, Dict

import torch.nn as nn

from torchktf.networks.blocks import ConvBnReLU, ResBlock, ConvBlock


class Backbone(nn.Module):
    def __init__(
        self,
        in_channels,
        n_basefilters,
        n_blocks,
        block_type,
        layer_size,
    ):
        if n_blocks < 2:
            raise ValueError(f'n_blocks must be at least 2, but got {n_blocks}')
        super().__init__()
        layers = [
            ('conv1', ConvBnReLU(in_channels, n_basefilters)),
            ('pool1', nn.MaxPool2d(3, stride=2)),
            ('block1', block_type(n_basefilters, n_basefilters))
        ]
        n_filters = n_basefilters
        for i in range(n_blocks-1):
            layers.append(
                (f'block{i+2}_0', block_type(n_filters, 2 * n_filters, stride=2))
            )
            for j in range(1, layer_size):
                layers.append(
                    (f'block{i+2}_{j}', block_type(2 * n_filters, 2 * n_filters))
                )
            n_filters = 2 * n_filters
        self.n_filters_out = n_filters
        self.blocks = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.blocks(x)


class Network(Backbone):

    def __init__(
        self,
        in_channels,
        n_outputs,
        n_basefilters,
        n_blocks,
        layer_size,
        block_type,
    ):
        super().__init__(
            in_channels,
            n_basefilters,
            n_blocks,
            block_type,
            layer_size,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        n_out = n_outputs if n_outputs > 2 else 1
        self.fc = nn.Linear(self.n_filters_out, n_out)

    def forward(self, x):
        out = self.blocks(x)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ConvNet(Network):
    def __init__(
        self,
        in_channels: int = 1,
        n_outputs: int = 3,
        n_basefilters: int = 16,
        n_blocks: int = 4,
        layer_size: int = 2,
    ):
        super().__init__(
            in_channels,
            n_outputs,
            n_basefilters,
            n_blocks,
            layer_size,
            ConvBlock,
        )


class ResNet(Network):
    def __init__(
        self,
        in_channels: int = 1,
        n_outputs: int = 3,
        n_basefilters: int = 16,
        n_blocks: int = 4,
        layer_size: int = 2,
    ):
        super().__init__(
            in_channels,
            n_outputs,
            n_basefilters,
            n_blocks,
            layer_size,
            ResBlock,
        )
