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
import torch.nn as nn
from numpy import prod

class LPFlatten(nn.Module):
    def forward(self, mv):
        input_mean, input_var = mv

        out_shape = prod(input_mean.size()[1:])
        return input_mean.view(-1, out_shape), input_var.view(-1, out_shape)
