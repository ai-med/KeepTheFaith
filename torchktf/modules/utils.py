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
import math
from pathlib import Path
from typing import Union

from hydra.utils import instantiate as hydra_init
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from yaml import safe_load as yaml_load

import subprocess


def format_float_to_str(x: float) -> str:
    return "{:2.1f}".format(x * 100)


def init_vector_normal(vector: torch.Tensor):
    stdv = 1. / math.sqrt(vector.size(1))
    vector.data.uniform_(-stdv, stdv)


def get_current_stage(epoch, epochs_all, epochs_nam, warmup=10):
    total = epochs_all + epochs_nam
    stage = "nam_only"
    if epoch < 0.5 * warmup:
        stage = "warmup"
    elif epoch < warmup:
        stage = "warmup_protonet"
    elif (epoch - warmup) % total < epochs_all:
        stage = "all"
    return stage


def get_last_valid_checkpoint(path: Path):
    epoch = int(path.stem.split("=")[1].split("-")[0])
    epoch_old = int(path.stem.split("=")[1].split("-")[0])
    config = load_config(path)
    e_all = config.model.epochs_all
    e_nam = config.model.epochs_nam
    warmup = config.model.epochs_warmup

    stage = get_current_stage(epoch, e_all, e_nam, warmup)
    while stage == "nam_only":
        epoch -= 1
        stage = get_current_stage(epoch, e_all, e_nam, warmup)
    if epoch != epoch_old:
        epoch += 1
    ckpt_path = str(path)
    print(f"Previous epoch {epoch_old} was invalid. Valid checkpoint is of epoch {epoch}")
    return ckpt_path.replace(f"epoch={epoch_old}", f"epoch={epoch}")


def init_vectors_orthogonally(vector: torch.Tensor, n_protos_per_class: int):
    # vector has shape (n_protos, n_chans)
    assert vector.size(0) % n_protos_per_class == 0
    torch.nn.init.xavier_uniform_(vector)

    for j in range(vector.size(0)):
        vector.data[j, j // n_protos_per_class] += 1.


def load_config(ckpt_path: Union[str, Path], cluster=False) -> DictConfig:
    config_path = str(Path(ckpt_path).parent.parent / '.hydra' / 'config.yaml')
    with open(config_path) as f:
        y = yaml_load(f)
    config = OmegaConf.create(y)
    return config


def load_model_and_data(ckpt_path: str, cluster=False, device=torch.device("cuda")):
    ''' loads model and data with respect to a checkpoint path
        must call data.setup(stage) to setup data
        pytorch model can be retrieved with model.net '''
    config = load_config(Path(ckpt_path), cluster=cluster)
    data = hydra_init(config.datamodule)
    model = hydra_init(config.module, _recursive_=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    return model, data, config


def get_git_hash():
    return subprocess.check_output([
        "git", "rev-parse", "HEAD"
    ], encoding="utf8").strip()


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding,
                          previous_layer_rf_info):
    n_in = previous_layer_rf_info[0]  # input size
    j_in = previous_layer_rf_info[1]  # receptive field jump of input layer
    r_in = previous_layer_rf_info[2]  # receptive field size of input layer
    start_in = previous_layer_rf_info[3]  # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if (n_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert (n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1)  # sanity check
        assert (pad == (n_out-1)*layer_stride - n_in + layer_filter_size)  # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert (n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1)  # sanity check
        assert (pad == (n_out-1)*layer_stride - n_in + layer_filter_size)  # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1

    pL = math.floor(pad/2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1)*j_in
    start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in
    return [n_out, j_out, r_out, start_out]


def compute_rf_protoL_at_spatial_location(img_size, height_index, width_index, protoL_rf_info):
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    start = protoL_rf_info[3]
    assert (height_index < n)
    assert (width_index < n)

    center_h = start + (height_index*j)
    center_w = start + (width_index*j)

    rf_start_height_index = max(int(center_h - (r/2)), 0)
    rf_end_height_index = min(int(center_h + (r/2)), img_size)

    rf_start_width_index = max(int(center_w - (r/2)), 0)
    rf_end_width_index = min(int(center_w + (r/2)), img_size)

    return [rf_start_height_index, rf_end_height_index,
            rf_start_width_index, rf_end_width_index]


def compute_rf_prototype(img_size, prototype_patch_index, protoL_rf_info):
    img_index = prototype_patch_index[0]
    height_index = prototype_patch_index[1]
    width_index = prototype_patch_index[2]
    rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                       height_index,
                                                       width_index,
                                                       protoL_rf_info)
    return [img_index, rf_indices[0], rf_indices[1],
            rf_indices[2], rf_indices[3]]


def compute_rf_prototypes(img_size, prototype_patch_indices, protoL_rf_info):
    rf_prototypes = []
    for prototype_patch_index in prototype_patch_indices:
        img_index = prototype_patch_index[0]
        height_index = prototype_patch_index[1]
        width_index = prototype_patch_index[2]
        rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                           height_index,
                                                           width_index,
                                                           protoL_rf_info)
        rf_prototypes.append([img_index, rf_indices[0], rf_indices[1],
                              rf_indices[2], rf_indices[3]])
    return rf_prototypes


def compute_proto_layer_rf_info_v2(img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):

    assert (len(layer_filter_sizes) == len(layer_strides))
    assert (len(layer_filter_sizes) == len(layer_paddings))

    rf_info = [img_size, 1, 1, 0.5]

    for i in range(len(layer_filter_sizes)):
        filter_size = layer_filter_sizes[i]
        stride_size = layer_strides[i]
        padding_size = layer_paddings[i]

        rf_info = compute_layer_rf_info(layer_filter_size=filter_size,
                                        layer_stride=stride_size,
                                        layer_padding=padding_size,
                                        previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info


def init_pretrained_weights(model, pretrained_ckpt):
    state_dict = torch.load(pretrained_ckpt, map_location=model.prototypes.device)['state_dict']
    state_dict = {x.replace('net.blocks.', ''): y for x, y in state_dict.items()}
    state_dict = {x: y for x, y in state_dict.items() if x in model.blocks.state_dict().keys()}  # get intersection of layers
    model.blocks.load_state_dict(state_dict)
    return model
