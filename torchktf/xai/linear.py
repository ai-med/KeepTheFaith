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
from torch import Tensor
from torch.nn import functional as F

from torchktf.xai import square


class LPLinear(torch.nn.Linear):
    """
    Lightweight probabilistic linear layer, with inputs being normally distributed.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(LPLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        input_mean, input_var = inputs
        m = F.linear(input_mean, self.weight, self.bias)
        v = F.linear(input_var, square(self.weight))

        return m, v


class ProbLinearInput(torch.nn.Linear):
    """
    Lightweight probabilistic linear input layer. Transforms sampled inputs into a normally distributed inputs.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(ProbLinearInput, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.epsilon = 1e-7

    def forward(
        self, inputs: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Performs probabilistic transformation in the forward pass of a linear operation.
        Args:
            inputs (): Inputs in a tuple of (Input, mask, k)

        Returns: Mean and variance of the input distribution with (mv1) and without (mv2) masked features.

        """

        # Note: Takes zero baseline as default, does not use DaspConfig
        input_, mask, k = inputs
        assert len(mask.shape) == len(
            input_.shape
        ), "Inputs must have same number of dimensions."
        one = torch.tensor([1.0], device=input_.device)
        mask_comp = one - mask
        inputs_i = input_ * mask_comp

        dot = F.linear(input_, self.weight)
        dot_i = F.linear(inputs_i, self.weight)
        dot_mask = torch.sum(mask_comp, dim=1, keepdim=True)
        dot_v = F.linear(square(inputs_i), square(self.weight))
        # Compute mean without feature i
        mu = dot_i / dot_mask
        v = dot_v / dot_mask - square(mu)
        # Compensate for number of players in current coalition
        mu1 = mu * k
        # Compute mean of the distribution that also includes player i (acting as bias to expectation)
        mu2 = mu1 + (dot - dot_i)
        # Compensate for number or players in the coalition
        v1 = v * k * (one - (k - one) / (dot_mask - one))
        # Set something different than 0 if necessary
        v1 = torch.clamp(v1, min=self.epsilon)
        # Since player i is only a bias, at this point the variance of the distribution than
        # includes it is the same
        v2 = v1

        if isinstance(self.bias, torch.nn.Parameter):
            mu1 += self.bias
            mu2 += self.bias

        return (mu1, v1), (mu2, v2)


class Flatten(torch.nn.Module):
    """One layer module that flattens its input, except for batch dimension."""

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x
        else:
            return x.view(x.size(0), -1)
