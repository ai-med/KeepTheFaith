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
from enum import Enum
from typing import Tuple, Type, Union

from torch import Tensor

from torchktf.xai.baselines import (
    Baseline,
    Zero_Baseline,
    Maxpool_Baseline,
)
from torchktf.xai.player_iterators import (
    AbstractPlayerIterator,
    DefaultPlayerIterator,
)

InputType = Union[Tensor, Tuple[Tensor, Tensor]]


# Definition of a configuration of all possible baseline types and the respective player iterators
class ProtoPFaithConfig(Enum):
    """Available pre-defined configs for running ProtoPFaith.
    Baseline defines the type of baseline used for substituting features.
    Playergen defines the Player Iterator (choice depends on network and data type used).
    """

    DFLT_ZERO = {"baseline": Zero_Baseline, "playergen": DefaultPlayerIterator}
    DFT_MAXPOOL = {"baseline": Maxpool_Baseline, "playergen": DefaultPlayerIterator}

    def get_baseline(self, **kwarg) -> Baseline:
        """
        Getter method for retrieving the respective baseline of a DaspConfig
        Args:
            kwarg (): Optional argument of respective Baseline class. Mean_KNN_Baseline takes k,
                Hull_Baseline the Hull of the current example.
        Returns: A baseline class object

        """
        return self.value["baseline"](**kwarg)

    def get_playergen(self) -> Type[AbstractPlayerIterator]:
        """
        Getter method for retrieving the player iterator of a ProtoPFaithConfig
        Returns: An UNINSTANTIATED player iterator class

        """
        return self.value["playergen"]

    def set_playergen(
        self, inputs: InputType, random: bool = False, sampling: bool = False
    ) -> AbstractPlayerIterator:
        """
        Instantiates the player iterator with an input.
        Args:
            inputs (): Input tensor which will be passed through the network
        Returns: A player iterator object

        """
        playergen = self.get_playergen()
        return playergen(inputs, random=random, sampling=sampling)
