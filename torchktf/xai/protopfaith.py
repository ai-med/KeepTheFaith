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
#################################################################
# extended and adapted from:
# https://github.com/marcoancona/DASP
# and
# https://github.com/ai-med/SVEHNN
#################################################################
import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .baselines import Zero_Baseline
from .player_iterators import ImagePlayerIterator
from .lp_parser import parse_model

LOG = logging.getLogger(__name__)


class ProtoPFaith:
    """
    Basic functionality of ProtoPFaith.
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        window_shape: Tuple[int, int, int],
        silent: bool = False,
    ) -> None:
        """

                Args:
                    model (): The non-probabilistic model used for the explanation.
                    inputs (): a torch Tensor with the shape of the input data.
                    silent (): disable tqdm
                """
        self.player_generator = ImagePlayerIterator(inputs, random=False, window_shape=window_shape)
        self.baseline = Zero_Baseline()
        self.lp_model = parse_model(model, self.player_generator)
        self.silent = silent

    def run(
        self,
        x: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        ks_val: Optional[float] = None,
    ) -> np.ndarray:
        """
        Performs a run of ProtoPFaith.
        Args:
            x (): Primary input data.
            feat (): Secondary input data
            steps (): Number of steps = number of coalition sizes to be calculated.
            ks_val (): An option to set the coalition size manually (only used for validation purposes).

        Returns: The approximated Shapley values for the input data.

        """
        self.player_generator.set_n_steps(steps)
        # retrieve (sample) coalition sizes
        ks = self.player_generator.get_steps_list()
        # update steps
        n_steps = ks.shape[0]

        print("ProtoPFaith: Computing %d coalitions sizes:" % ks.shape[0])
        if n_steps < 10:
            LOG.info(ks)
        else:
            LOG.info(
                "%d %d ... %d %d",
                ks[0],
                ks[1],
                ks[-2],
                ks[-1],
            )

        result = None
        batch_size = x.size()[0]

        tile_input = [n_steps] + (x.dim() - 1) * [1]
        tile_mask = [n_steps * batch_size] + (x.dim() - 1) * [1]

        if feat is not None:
            tile_feat_mask = [n_steps * batch_size] + (feat.dim() - 1) * [1]
            tile_feat_input = [n_steps] + (feat.dim() - 1) * [1]
            # account for the number of features in each input type and recalculate coalition sizes
            ks_feat = ks * feat.size()[1] / self.player_generator.n_players
            ks = ks * x.size()[2] / self.player_generator.n_players

        ks = torch.as_tensor(ks, device=x.device)
        self.lp_model = self.lp_model.net.eval().to(x.device)
        print(self.lp_model.clf.weight.device)

        print(
            "Executing attribution for %d features in player generator.",
            self.player_generator.n_players,
        )

        with torch.no_grad():
            with tqdm(
                range(self.player_generator.n_players), disable=self.silent
            ) as progress_bar:
                for i, (mask, mask_output) in enumerate(self.player_generator):
                    if ks_val is not None:
                        # set ks manually if necessary
                        ks = torch.tensor(
                            [ks_val], dtype=torch.float32, device=x.device
                        )
                    if feat is not None:
                        mask, feat_mask = mask
                        mask_output, feat_output = mask_output
                        mask_output = np.stack((mask_output, feat_output), axis=1)

                    mask = torch.as_tensor(mask, device=x.device)

                    inputs = (
                        x.repeat(tile_input).to(x.device),
                        mask.repeat(tile_mask).to(x.device),
                        ks.repeat(batch_size).unsqueeze(dim=1).to(x.device),
                    )
                    if feat is not None:
                        feats = (
                            feat.repeat(tile_feat_input).to(x.device),
                            feat_mask.repeat(tile_feat_mask).to(x.device),
                            ks_feat.repeat(batch_size)
                            .unsqueeze(dim=1)
                            .to(x.device),
                        )
                        # forward pass through probabilistic heterogeneous network
                        with torch.no_grad():
                            mean, _ = self.lp_model(inputs)
                    else:
                        # forward pass through probabilistic homogeneous network
                        with torch.no_grad():
                            mean, _ = self.lp_model(inputs)

                    y_wo_i = mean[:n_steps]  # y1
                    y_w_i = mean[n_steps:]  # y2

                    # the marginal contributions are l2 distances.
                    # as a result, smaller marginal contribution is better.

                    # take mean over all steps
                    y = torch.mean(y_w_i - y_wo_i, dim=0).detach().cpu().numpy()

                    if np.isnan(y).any():
                        raise RuntimeError(
                            "Result contains nans! This should not happen. I recommend to clamp zeros."
                        )

                    # Compute Shapley Values as mean of all coalition sizes
                    mask_shape = mask_output.shape[1:]  # trim batch dimension
                    if result is None:
                        # n_protos, n_inp_chans, n_coordinates, n_points; 
                        result = np.zeros(y.shape + mask_shape, dtype=float)

                    shape_out = list(y.shape) + [1] * len(mask_shape)
                    # import ipdb;ipdb.set_trace()
                    # shape_out += [1] * (mask_output.ndim-1)

                    result += np.reshape(y, shape_out) * mask_output  # broadcasting
                    progress_bar.update()
        return result

