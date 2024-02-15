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
import argparse
from pathlib import Path
from torchktf.modules.utils import load_model_and_data
import numpy as np
import torch
import math
import pandas as pd
import cv2
import copy
from torchktf.xai.protopfaith import ProtoPFaith


def make_dasp_attribution(module, img, steps):

    if 'vgg' in module.net.block_type:
        winsize = (img.size(-3), 1, 1)  # can't make any bigger in original vgg model cause of nan issue of coalition sizes
    else:
        winsize = (img.size(-3), 2, 2)
    explainer = ProtoPFaith(copy.deepcopy(module), img, window_shape=winsize)
    result = explainer.run(img, steps=steps)
    return result

def make_original_attribution(module, img):

    _, _, _, dists = module.net.eval().cuda()(img)
    return dists.detach().cpu().numpy()


def make_attributions(module, imgidx, protoidx, img, dasp_steps, savepath, inverse_norm=None):

    original = make_original_attribution(module, img)[:, protoidx]
    original = cv2.resize(original.transpose(1,2,0), dsize=tuple(img.shape[-2:]), interpolation=cv2.INTER_CUBIC)[np.newaxis, ...]
    np.save(savepath / f"ppnet_proto_{imgidx}_{protoidx}_{dasp_steps}.npy", original)
    attribution = make_dasp_attribution(module, img, dasp_steps)[protoidx]
    np.save(savepath / f"dasp_proto_{imgidx}_{protoidx}_{dasp_steps}.npy", attribution)

    np.save(savepath / f"input_proto_{imgidx}_{protoidx}_{dasp_steps}.npy", inverse_norm(img).cpu().numpy())

    return original, attribution, img


def main(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--idx', type=int)
    parser.add_argument('--dasp_steps', type=int, default=16)
    args = parser.parse_args(args)

    # pathing
    savepath = Path(args.ckpt).parent.parent / 'attributions'
    savepath.mkdir(exist_ok=True, parents=True)

    # dl setup
    module, data, _ = load_model_and_data(args.ckpt)
    data.setup("test")
    module = module.eval().cuda()

    # get img of min distance
    img_indices = np.array([-1] * module.net.n_protos)
    distances = np.array([np.inf] * module.net.n_protos).astype(np.float32)
    for i in range(len(data.push_data)):
        sample, cl = data.push_data.__getitem__(i)
        sample = sample.cuda()
        _, min_distances, _, _ = module.net(sample.unsqueeze(0))
        for j in range(module.net.n_protos):
            if (min_distances[0, j] < distances[j]) and (cl == module.net.p_classmapping[j].argmax()):
                distances[j] = min_distances[0, j]
                img_indices[j] = i
    assert img_indices.min() > -1, img_indices
    imgidx = img_indices[args.idx]

    # select img
    inp = data.push_data.__getitem__(imgidx)[0].unsqueeze(0).cuda()

    # make attributions
    pimg, dasp, inp = make_attributions(module, imgidx, args.idx, inp, args.dasp_steps, savepath, inverse_norm=data.inverse_norm)  # inp is now inverse, i.e. original color space

    return pimg, dasp, img


if __name__ == '__main__':
    main()

