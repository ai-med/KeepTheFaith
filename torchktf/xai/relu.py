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
import torch
import torch.nn
from torch import Tensor
from torch.distributions import Normal

from torchktf.xai import square


def forward_relu(mv: Tuple[Tensor, Tensor], epsilon) -> Tuple[Tensor, Tensor]:
    input_mean, input_var = mv
    normal = Normal(
        torch.tensor([0.0], dtype=torch.float32, device=input_mean.device),
        torch.tensor([1.0], dtype=torch.float32, device=input_mean.device),
    )

    v = torch.clamp(input_var, min=epsilon)
    s = torch.sqrt(v)
    m_div_s = input_mean / s
    prob = torch.exp(normal.log_prob(m_div_s))
    m_out = input_mean * normal.cdf(m_div_s) + s * prob
    v_out = (
        (square(input_mean) + v) * normal.cdf(m_div_s)
        + (input_mean * s) * prob
        - square(m_out)
    )
    return m_out, v_out


def forward_relu1(mv, epsilon):

    m_in, v_in = mv
    normal = Normal(
        torch.tensor([0.0], dtype=torch.float32, device=m_in.device),
        torch.tensor([1.0], dtype=torch.float32, device=v_in.device),
    )

    sigma = torch.sqrt(torch.clamp(v_in, min=epsilon))

    one_shifted = (1 - m_in) / sigma
    zero_shifted = - (m_in / sigma)

    pdf_one = torch.exp(normal.log_prob(one_shifted))
    pdf_zero = torch.exp(normal.log_prob(zero_shifted))
    cdf_one = normal.cdf(one_shifted)
    cdf_zero = normal.cdf(zero_shifted)

    m_out = sigma * (pdf_zero - pdf_one) \
        + m_in * (cdf_one - cdf_zero) \
        + 1 - cdf_one

    m_out_square = torch.square(m_out)
    m_in_square = torch.square(m_in)
    v_out = (m_in_square - 2 * m_in * m_out + v_in + 2 * m_out - 1) * cdf_one \
            - (m_in_square - 2 * m_in * m_out + v_in) * cdf_zero \
            - (m_in * sigma - 2 * m_out * sigma + sigma) * pdf_one \
            + (m_in * sigma - 2 * m_out * sigma) * pdf_zero \
            + m_out_square - 2 * m_out + 1

    return m_out, v_out



class LPReLU(torch.nn.Module):
    def __init__(self):
        super(LPReLU, self).__init__()
        self.forward_relu = forward_relu
        self.epsilon = 1e-6

    def forward(self, inputs):
        return self.forward_relu(inputs, self.epsilon)


class LPReLU1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_func = forward_relu1
        self.epsilon = 1e-6

    def forward(self, inputs):
        return self.forward_func(inputs, self.epsilon)
