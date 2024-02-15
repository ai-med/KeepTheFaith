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
import numpy as np
import torch
from torch import Tensor
from torch.nn import Conv1d, Conv2d
import torch.nn.functional as F

from torchktf.xai.player_iterators import AbstractPlayerIterator
from torchktf.xai.protopfaith_config import Baseline

def square(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(x, torch.tensor([2.0], device=x.device))

InputLayerInputs = Tuple[Tensor, Tensor, Tensor]
InputLayerOutputs = Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]


class LPConv1D(Conv1d):
    """
    Propagate distributions over a probabilistic Conv1D layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True):
        super(LPConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        mean, variance = inputs
        m = F.conv1d(
            mean,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        v = F.conv1d(
            variance,
            weight=square(self.weight),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return m, v


class ProbConv1DInput(Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True,
                 *,
                 baseline: Baseline,
                 playergen: AbstractPlayerIterator):
        super(ProbConv1DInput, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.epsilon = 1e-7
        self.baseline = baseline
        self.playergen = playergen

    def _conv1d(self, inputs, weight):
        return F.conv1d(
            inputs,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(
        self,
        inputs: InputLayerInputs,
    ) -> InputLayerOutputs:
        input_, mask, k = inputs
        size_coalition = self.playergen.coalition_size

        # get n_players from shape without batch dimension
        n_players = np.prod(input_.size()[1:])
        # account for size of coalition
        n_players = torch.tensor(
            n_players / size_coalition,
            dtype=torch.float, device=input_.device)

        k = torch.unsqueeze(k, -1)
        assert mask.dim() == input_.dim(), \
            "Inputs must have the same number of dimensions."
        one = torch.as_tensor([1.0], dtype=torch.float, device=input_.device)

        in_ghost, inputs_i = self.baseline.apply_in_prob_layer(input_, mask)

        # find all the indices which are not i (not masked out)
        # note: current approach only works if i is masked out as complete point (- all coordinates)
        # so we can find out which index was masked by looking at dim 2 only
        idx = torch.nonzero(in_ghost[0, 0, :]).squeeze(dim=1)

        conv_m = self._conv1d(input_, self.weight)
        conv_m_i = self._conv1d(inputs_i, self.weight)
        conv_count = self._conv1d(
            in_ghost, weight=torch.ones_like(self.weight)
        )
        # If we mask complete points, conv_count will be zero for the removed point, leading to division by zero later on
        # assert torch.gt(conv_count, 0).all().item() != 0
        conv_v_i = self._conv1d(square(inputs_i), square(self.weight))

        kn = torch.div(k, n_players)
        # here using k' = conv_count * k / n_players is not necessary
        # because conv_count cancels out when computing mean
        m_wo_i = torch.mul(conv_m_i, kn)
        m_w_i = m_wo_i + (conv_m - conv_m_i)

        # Compensate for number of players in the coalition
        k = torch.mul(conv_count, kn)
        v_wo_i = torch.div(conv_v_i, conv_count) - square(torch.div(conv_m_i, conv_count))
        v_wo_i = v_wo_i * k * (one - (k - one) / (conv_count - one))

        # Since player i is only a bias, at this point the variance of the distribution than
        # includes it is the same
        v_w_i = v_wo_i.clone()
        # set all variances for point i to 0 -> the point is non-random
        v_w_i[torch.isnan(v_wo_i)] = 0.0
        # remove point i from mean and variance
        v_wo_i = v_wo_i[:, :, idx]
        m_wo_i = m_wo_i[:, :, idx]

        if isinstance(self.bias, torch.nn.Parameter):
            b = self.bias.view(-1, 1)
            m_wo_i.add_(b)
            m_w_i.add_(b)

        return (m_wo_i, v_wo_i), (m_w_i, v_w_i)


class LPConv2D(Conv2d):
    """
    Propagate distributions over a probabilistic Conv2D layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(LPConv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs)

    def forward(self, inputs):
        if inputs is None:
            import ipdb;ipdb.set_trace()
        mean, variance = inputs

        m = F.conv2d(mean, self.weight, bias=self.bias, stride=self.stride,
                    padding=self.padding, dilation=self.dilation,
                    groups=self.groups)
        v = F.conv2d(variance, square(self.weight), bias=None, stride=self.stride,
                     padding=self.padding, dilation=self.dilation,
                     groups=self.groups)

        return m, v


class ProbConv2DInput(Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, *,
                 playergen: AbstractPlayerIterator,
                 **kwargs) -> None:
        super(ProbConv2DInput, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs)

        self.playergen = playergen

    def _conv2d(self, inputs, kernel):
        return F.conv2d(
            inputs,
            kernel,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

    def forward(self, inputs: InputLayerInputs) -> InputLayerOutputs:
        input_, mask, k = inputs
        device = input_.device

        size_coalition = self.playergen.coalition_size

        n_players = np.prod(input_.size()[1:])
        n_players = torch.tensor(
            n_players / size_coalition,
            dtype=torch.float, device=device)

        # When I say k, I actually mean k coalition-players so need to compensate for it
        k = torch.unsqueeze(torch.unsqueeze(k, -1), -1)
        one = torch.as_tensor([1.0], dtype=torch.float, device=device)

        ghost = torch.ones_like(input_, device=device) * (1.0 - mask)
        inputs_i = input_ * (1.0 - mask)

        conv = self._conv2d(input_, self.weight)
        conv_i = self._conv2d(inputs_i, self.weight)
        conv_count = self._conv2d(ghost, torch.ones_like(self.weight, device=device))
	    # If we mask complete points, conv_count will be zero for the removed point, leading to division by zero later on
        # assert torch.gt(conv_count, 0).all().item() != 0
        conv_v = self._conv2d(square(inputs_i), square(self.weight))

        # Compute mean without feature i
        # Compensate for number of players in current coalition
        mu1 = torch.mul(conv_i, torch.div(k, n_players))  # without i
        # Compute mean of the distribution that also includes player i (acting as bias to expectation)
        mu2 = mu1 + (conv - conv_i)  # with i
        # Compute variance without player i
        v1 = torch.div(conv_v, conv_count) - square(torch.div(conv_i, conv_count))

        # Compensate for number or players in the coalition
        k = torch.mul(conv_count, torch.div(k, n_players))
        v1 = v1 * k * (one - (k - one) / (conv_count - one))
        # Set something different than 0 if necessary
        v1 = torch.clamp(v1, min=0.00001)
        # Since player i is only a bias, at this point the variance of the distribution than
        # includes it is the same
        v2 = v1.clone()

        if isinstance(self.bias, torch.nn.Parameter):
            b = self.bias.view(-1, 1, 1)
            mu1.add_(b)
            mu2.add_(b)

        return torch.cat((mu1, mu2)), torch.cat((v1, v2))  # stacked as (m_wo_i, m_w_i)
