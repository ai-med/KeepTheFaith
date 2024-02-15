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
import numpy as np
import logging
import torch

LOG = logging.getLogger(__name__)


def spaced_elements(array, num_elems=4):
    return [x[len(x)//2] for x in np.array_split(np.array(array), num_elems)]


class AbstractPlayerIterator(ABC):

    def __init__(self, input, random=False):
        self._assert_input_compatibility(input)
        self.input_shape = input.shape[1:]
        self.random = random
        self.n_players = self._get_number_of_players_from_shape()
        self.permutation = np.array(range(self.n_players), 'int32')
        if random is True:
            self.permutation = np.random.permutation(self.permutation)
        self.i = 0
        self.kn = self.n_players
        self.ks = spaced_elements(range(self.n_players), self.kn)

    def set_n_steps(self, steps):
        self.ks = torch.tensor(
            spaced_elements(np.arange(self.n_players), steps),
            dtype=torch.float32,
        )
        self.kn = len(self.ks)

    def get_number_of_players(self):
        return self.n_players

    def get_explanation_shape(self):
        return self.input_shape

    def get_coalition_size(self):
        return 1

    def get_steps_list(self):
        return self.ks

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == self.n_players:
            raise StopIteration
        m = self._get_masks_for_index(self.i)
        self.i = self.i + 1
        return m

    @abstractmethod
    def _assert_input_compatibility(self, input):
        pass

    @abstractmethod
    def _get_masks_for_index(self, i):
        pass

    @abstractmethod
    def _get_number_of_players_from_shape(self):
        pass


class DefaultPlayerIterator(AbstractPlayerIterator):

    def _assert_input_compatibility(self, input):
        assert len(input.shape) > 1, 'DefaultPlayerIterator requires an input with 2 or more dimensions'

    def _get_number_of_players_from_shape(self):
        return int(np.prod(self.input_shape))

    def _get_masks_for_index(self, i):
        mask = np.zeros(self.n_players, dtype='int32')
        mask[self.permutation[i]] = 1
        return mask.reshape(self.input_shape), mask.reshape(self.input_shape)


class ImagePlayerIterator(AbstractPlayerIterator):
    """ from DASP, changed for PyTorch """

    def __init__(self, input, random=False, window_shape=(1, 1, 1)):
        self.window_shape = window_shape
        assert window_shape[1] == window_shape[2], "window shape should be quadratric"
        assert self.window_shape is not None, "window_shape cannot be None"
        assert input.ndim == 4, "input must be tensor of shape BS x C x H x W"
        assert len(self.window_shape) == 3, "window_shape must contain 3 elements"
        assert 1 <= window_shape[0] <= input.shape[1], \
            "dimension of window_shape for channels must be in range 0..n_input_channels"
        assert window_shape[0] == input.shape[1] or window_shape[0] == 1, \
            "n_channels of window_shape must be 1 or equal to the channel dimension of the input"
        assert input.shape[2] % self.window_shape[1] == 0 and input.shape[3] % self.window_shape[2] == 0, \
            f"input dimensions {input.shape} must be multiple of window_shape dimensions {self.window_shape}"
        super(ImagePlayerIterator, self).__init__(input, random)

    def _input_shape_merged(self):
        shape = list(self.input_shape)
        if self.window_shape[0] > 1:
            shape[0] = 1
        return shape

    def _assert_input_compatibility(self, input):
        assert len(input.shape) == 4, 'ImagePlayerIterator requires an input with 4 dimensions'

    def _get_number_of_players_from_shape(self):
        shape = self._input_shape_merged()
        if self.window_shape[1] > 1:
            shape[1] = shape[1] / self.window_shape[1]
        if self.window_shape[2] > 1:
            shape[2] = shape[2] / self.window_shape[2]
        print('nplayers', np.prod(shape, dtype='int32'))
        LOG.info('nplayers', np.prod(shape, dtype='int32'))
        return np.prod(shape, dtype='int32')

    def _get_sampling_mask_for_indices(self, i: int):
        raise NotImplementedError("This method should not be necessary to be called in ImagePlayerIterator")

    def _get_masks_for_index(self, i):
        mask_input = np.zeros(self.input_shape, dtype='int32')
        mask = np.zeros(self._input_shape_merged())
        i = self.permutation[i]

        nrows, ncols = self.input_shape[1] // self.window_shape[1], self.input_shape[2] // self.window_shape[2]
        row_step = self.window_shape[1]
        col_step = self.window_shape[2]
        coalition_size = row_step*col_step
        self.coalition_size = coalition_size
        row = i // nrows
        col = i % ncols
        LOG.info(row)
        LOG.info(col)

        mask_input[:, row*row_step:(1+row)*row_step, col*col_step:(1+col)*col_step] = 1
        mask[:, row*row_step:(1+row)*row_step, col*col_step:(1+col)*col_step] = 1. / coalition_size

        return mask_input[np.newaxis], mask[np.newaxis]

    def get_explanation_shape(self):
        return self._input_shape_merged()

