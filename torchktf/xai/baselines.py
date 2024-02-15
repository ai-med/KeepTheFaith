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
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


def get_device() -> torch.device:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def _assert_input_compatibility_1d(inputs: Tensor, mask: Tensor):
    assert inputs.shape == mask.shape, "Inputs and mask must have same dimension."
    assert len(inputs.size()) == 3, "Input must have 3 dimensions."


def _assert_input_compatibility_3d(inputs: Tensor, mask: Tensor):
    assert inputs.shape == mask.shape, "Inputs and mask must have same dimension."
    assert len(inputs.size()) == 3, "Input must have 3 dimensions."
    assert inputs.size(1) == 3, "Input must have 3 channels (coordinates)."


def apply_mask_complement(inputs: Tensor, mask: Tensor) -> Tensor:
    assert inputs.dim() == mask.dim(), "Inputs and mask must have same rank."
    mask_comp = torch.ones_like(mask) - mask
    return inputs * mask_comp


class Baseline(ABC):
    """
    An Interface for Baselines used in SVEHNN.
    """

    @abstractmethod
    def get_all_values(self, inputs: Tensor) -> Tensor:
        """
        Calculates baseline values for all entries in the input tensor.
        Args:
            inputs (): Input tensor for which the baseline should be calculated. Can be neglected for baselines like Zero_Baseline.

        Returns: A tensor with same input shape as "inputs" containing the baseline value for each entry.

        """
        pass

    @abstractmethod
    def apply_on_input(self, inputs: Tensor, mask: Tensor) -> Tensor:
        """
        Applies the baseline on an input tensor according to the indices in the mask.
        Args:
            inputs (): Inputs to be masked.
            mask (): Mask, indicating which features to replace with the baseline. 1 = replace. 0 = original value stays.

        Returns: The masked input tensor.

        """
        pass

    def apply_in_prob_layer(
        self, inputs: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Applies the baseline in the Probabilistic Input Layer of the Lightweight Probabilistic Deep Network.
        Args:
            inputs (): Inputs to be masked.
            mask (): Mask, indicating which features to replace with the baseline. 1 = replace. 0 = original value stays.

        Returns: The masked input tensor and the mask complement as tuple.

        """
        inputs_i = self.apply_on_input(inputs, mask)
        in_ghost = apply_mask_complement(
            torch.ones_like(inputs, dtype=torch.float), mask
        )

        return in_ghost, inputs_i


class Maxpool_Baseline(Baseline):
    """
    Remove feature coalitions by neglecting it in a Max-Pooling operation.
    """

    def get_all_values(self, inputs: Tensor) -> Tensor:
        assert inputs is not None, f"Hull must be specified. Got: {None}"
        # N/A -> return identity
        return inputs.clone()

    def get_min_values(self, inputs: Tensor, isvariance=False, local=False) -> Tensor:
        """
        Method specific to the max-pooling baseline. Finds the minimal value in an input tensor
        s.t. the maxpooling operation strictly neglects it.
        Args:
            inputs (): tensor of inputs
            isvariance (): whether the input is a vector of variances.
                Enables us to treat variances differently than means.
            local (): local minimum of maxpool dimension (=True) or global minimum of input (=False)

        Returns: A vector of minimal values.

        """
        if isvariance:
            # we want zero variance for masked features, as there should be no uncertainty for masks.
            return torch.zeros_like(inputs, dtype=torch.float32)
        if local:
            # local minimum
            num_points = inputs.size(2)
            return inputs.min(dim=2)[0].unsqueeze(dim=2).repeat(1, 1, num_points)

        # global minimum
        return inputs.min().expand_as(inputs)

    def apply_on_input(self, inputs: Tensor, mask: Tensor, isvariance=False) -> Tensor:
        assert (
            inputs.shape[2] == mask.shape[2]
        ), "Input must have same shape in point dimension."
        num_channels = inputs.shape[1]
        mask = mask[:, 0:1, :].repeat(1, num_channels, 1)
        baseline = self.get_min_values(inputs, isvariance)
        masked_inputs = apply_mask_complement(inputs, mask) + baseline * mask
        return masked_inputs

    # override necessary, as maxpool does not have the same behavior
    def apply_in_prob_layer(
        self, inputs: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Not masking out anything, use all inputs
        in_ghost = torch.ones_like(inputs, dtype=torch.float)
        inputs_i = inputs.clone()

        return in_ghost, inputs_i


class Zero_Baseline(Baseline):
    """
    Substitute masked features with zero.
    """

    def get_all_values(self, inputs: Tensor) -> Tensor:
        return torch.zeros_like(inputs, dtype=torch.float32, device=inputs.device)

    def apply_on_input(self, inputs: Tensor, mask: Tensor) -> Tensor:
        inputs_i = apply_mask_complement(inputs, mask)
        return inputs_i
