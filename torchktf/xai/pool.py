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
import numbers
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.normal import Normal

from torchktf.xai import square


def _ab_max_pooling(mv1, mv2, normal, eps=1e-10) -> Tuple[Tensor, Tensor]:
    """
    Takes the probabilistic maximum of two input distributions.
    Args:
        mv1 (): Mean and variance of the first input distribution.
        mv2 (): Mean and variance of the other input distribution.

    Returns: Mean and variance with the maximum of the two distributions.

    """
    mu_a, va = mv1
    mu_b, vb = mv2
    vavb = torch.sqrt(torch.clamp(va + vb, min=eps))
    assert torch.isfinite(vavb).all().item() != 0

    muamub = mu_a - mu_b
    muamub_p = mu_a + mu_b
    alpha = muamub / vavb

    mu_c = vavb * torch.exp(normal.log_prob(alpha)) + muamub * normal.cdf(alpha) + mu_b
    vc = muamub_p * vavb * torch.exp(normal.log_prob(alpha))
    vc += (
        (square(mu_a) + va) * normal.cdf(alpha)
        + (square(mu_b) + vb) * (1.0 - normal.cdf(alpha))
        - square(mu_c)
    )

    return mu_c, vc


class LPMaxLayer(torch.nn.Module):
    """Placeholder Layer for Lightweight Probabilistic Max operation"""

    def __init__(self):
        super(LPMaxLayer, self).__init__()
        self.normal = Normal(loc=0.0, scale=1.0)

    def forward(self, mv: Tuple[Tensor, Tensor]):
        m, v = mv

        # unpack along time dimension
        m_chunks = torch.split(m, 1, dim=2)
        v_chunks = torch.split(v, 1, dim=2)
        # initialize with values of first time point
        m_a = m_chunks[0].squeeze(dim=2)
        v_a = v_chunks[0].squeeze(dim=2)
        for m_i, v_i in zip(m_chunks[1:], v_chunks[1:]):
            m_i.squeeze_(dim=2)
            v_i.squeeze_(dim=2)
            m_a, v_a = _ab_max_pooling((m_a, v_a), (m_i, v_i), self.normal)

        return m_a, v_a


class MaxLayer(torch.nn.Module):
    """Placeholder Layer for Max operation"""

    def __init__(self):
        super(MaxLayer, self).__init__()

    def forward(self, inputs: Tensor):
        return inputs.max(dim=-1)[0]


class LPMaxPool2d(torch.nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(LPMaxPool2d, self).__init__()

        if stride is None:
            stride = kernel_size

        self.kernel_size = self._to_pair(kernel_size)
        self.stride = self._to_pair(stride)
        self.padding = self._to_pair(padding)
        self.dilation = self._to_pair(dilation)

        self.normal = Normal(loc=0.0, scale=1.0)
        self._n_pool = np.prod(self.kernel_size)

    def _to_pair(self, value):
        if isinstance(value, numbers.Integral):
            value = [value] * 2
        return value

    def _unfold(self, inputs):
        input_shape = inputs.size()
        n_channels = input_shape[1]
        patches = F.unfold(inputs, self.kernel_size, self.dilation,
                           self.padding, self.stride)

        n_blocks = patches.size()[2]
        patches = patches.view(-1, n_channels, self._n_pool, n_blocks)

        blocks_per_dim = []
        for d, spatial_size in enumerate(input_shape[2:]):
            n = spatial_size + 2 * self.padding[d] - self.dilation[d] * (self.kernel_size[d] - 1) - 1
            n /= self.stride[d]
            n += 1
            blocks_per_dim.append(int(np.floor(n)))

        return patches, blocks_per_dim

    def forward(self, inputs):
        m, v = inputs

        m_patches, spatial_shape = self._unfold(m)
        v_patches, _ = self._unfold(v)

        # unpack along spatial dimension
        m_chunks = torch.split(m_patches, 1, dim=2)
        v_chunks = torch.split(v_patches, 1, dim=2)
        # initialize with values of first point
        m_a = m_chunks[0].squeeze(dim=2)
        v_a = v_chunks[0].squeeze(dim=2)
        for m_i, v_i in zip(m_chunks[1:], v_chunks[1:]):
            m_i.squeeze_(dim=2)
            v_i.squeeze_(dim=2)
            m_a, v_a = _ab_max_pooling((m_a, v_a), (m_i, v_i), self.normal)

        out_shape = m_a.size()[:2] + tuple(spatial_shape)
        m_a = m_a.view(out_shape)
        v_a = v_a.view(out_shape)

        return m_a, v_a
