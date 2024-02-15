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
from torchktf.networks.blocks import ResBlock
from torchktf.xai.sum import LPSum

import torch.nn as nn

from torchvision.models.resnet import Bottleneck

class LPResBlock(ResBlock):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(in_channels, out_channels, stride)
        self.plus = LPSum()

    def forward(self, inputs):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out = self.plus((out, residual))
        out = self.relu(out)
        return out


class LPBottleneck(nn.Module):
    def __init__(
        self,
        conv1,
        bn1,
        relu,
        conv2,
        bn2,
        conv3,
        bn3,
        downsample
    ):
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.conv3 = conv3
        self.bn3 = bn3
        self.downsample = downsample
        self.plus = LPSum()
        
    def forward(self, x):
        identity = x
		
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        	
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.plus((out, identity))
        out = self.relu(out)
        
        return out
