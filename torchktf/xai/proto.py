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
import torch
from typing import Callable, Tuple, Union
from torch import Tensor
from torchktf.networks.protonet import ProtoNet
from torchktf.xai.pool import LPMaxPool2d



class LPProtoNet(ProtoNet):

    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        n_basefilters: int,
        n_blocks: int,
        layer_size: int,
        block_type: str,
        optim_features: bool,
        n_protos_per_class: int,
        proto_dims: int,
        proto_activation_func: Union[str, Callable],
        proto_nonlinearity: str,
    ):
        super().__init__(
            in_channels,
            n_outputs,
            n_basefilters,
            n_blocks,
            layer_size,
            block_type,
            optim_features,
            n_protos_per_class,
            proto_dims,
            proto_activation_func,
            proto_nonlinearity,
        )
        self.lppool = None

    def _check_lppool(self, x):
        if self.lppool is None:
            kernel_size = x.size(-1)
            self.lppool = LPMaxPool2d(kernel_size, 1, 0, 1)

    def _quadratic(self, inputs):
        mean, var = inputs
        # we treat (X-p)like a multivariate gaussian with diagonal sigma
        m_temp = torch.einsum('bnchw,bnchw->bnhw', mean, mean)  # x.T*x, shape b x n_protos x h x w
        m_temp = var.sum(dim=2) + m_temp  # E[X.T*X] = trace(cov(X)) + X.T*X, broadcasted
        var_temp = var * mean  # broadcast
        var_temp = torch.einsum('bnchw,bnchw->bnhw', mean, var_temp)
        var_temp = 2 * (var ** 2).sum(dim=2) + 4 * var_temp  # Var[X.T*X] = 2 * trace(cov**2(X)) + 4 * mu.T cov(X) mu
        return m_temp, var_temp


    def _protolayer_prob(self, inputs):
        mean, var = inputs
        mean = mean.unsqueeze(1)  # resulting shape BS x 1 x C x H x W
        var = var.unsqueeze(1)
        protos = self.prototypes.unsqueeze(0)  # resulting shape 1 x N x C x 1 x 1

        # (1) X - p -> shift mean, var remains
        mean = mean - protos  # broadcasting -> shape is BS x N x C x H x W

        # (2) (X - p)**2 -> shift mean and var with quadratic forms
        mean, var = self._quadratic((mean, var))

        return mean, var


    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        conv_feats = self.blocks(inputs)
        self._check_lppool(conv_feats[0])
        conv_feats = self.feat_extractor(conv_feats)
        dist_mean, dist_var = self._protolayer_prob(conv_feats)
        mean, var = self.lppool((-dist_mean, dist_var))  # L2 norm -> use maxpool of inverse distances
        mean = -mean  # swap sign to get real distances -> big number is bad!
        return mean.view(-1, self.n_protos), var.view(-1, self.n_protos)