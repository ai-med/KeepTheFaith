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
from typing import Tuple
import torch.nn
import torch.nn.functional as F
from torch import Tensor

from torchktf.xai import square


class LPBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(LPBatchNorm1d, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        input_mean, input_var = inputs
        m = F.batch_norm(
            input_mean,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=False,
            momentum=self.momentum,
            eps=self.eps,
        )

        running_var = self.running_var
        weight = self.weight
        # check for channel dimension
        if input_var.dim() == 3:
            running_var = running_var.unsqueeze(dim=1)
            weight = weight.unsqueeze(dim=1)
        invstd_squared = 1.0 / (running_var + self.eps)
        v = input_var * invstd_squared * square(weight)

        return m, v


class LPBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(LPBatchNorm2d, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        input_mean, input_var = inputs
        m = F.batch_norm(
            input_mean,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=False,
            momentum=self.momentum,
            eps=self.eps,
        )

        running_var = self.running_var
        weight = self.weight
        # check for channel dimension
        if input_var.dim() == 4:
            running_var = running_var.unsqueeze(dim=-1).unsqueeze(dim=-1)
            weight = weight.unsqueeze(dim=-1).unsqueeze(dim=-1)
        invstd_squared = 1.0 / (running_var + self.eps)
        v = input_var * invstd_squared * square(weight)

        return m, v
