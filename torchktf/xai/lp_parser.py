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
import copy
import torch.nn as nn
import logging

from torchvision.models.resnet import Bottleneck

from torchktf.networks.blocks import ResBlock, ReLU1
from torchktf.xai.linear import LPLinear
from torchktf.xai.convolution import LPConv2D, ProbConv2DInput
from torchktf.xai.batchnorm import LPBatchNorm2d
from torchktf.xai.relu import LPReLU, LPReLU1
from torchktf.xai.pool import LPMaxPool2d
from torchktf.xai.resblock import LPResBlock, LPBottleneck
from torchktf.xai.proto import LPProtoNet
from torchktf.networks.protonet import ProtoNet

LOG = logging.getLogger(__name__)

LP_LAYER_DICT = {
    nn.Linear: (LPLinear, ['in_features', 'out_features', 'bias'], None),
    nn.Conv2d: (LPConv2D, ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias', 'groups'], ProbConv2DInput),  # oder ProbConv2DInput
    nn.BatchNorm2d: (LPBatchNorm2d, ['num_features', 'eps', 'momentum', 'affine', 'track_running_stats'], None),
    nn.ReLU: (LPReLU, [], None),
    ReLU1: (LPReLU1, [], None),
    nn.MaxPool2d: (LPMaxPool2d, ['kernel_size', 'stride', 'padding', 'dilation'], None),
    ResBlock: (LPResBlock, ['in_channels', 'out_channels', 'stride'], None),
    Bottleneck: (LPBottleneck, ['conv1', 'bn1', 'relu', 'conv2', 'bn2', 'conv3', 'bn3', 'downsample'], None),
    ProtoNet: (LPProtoNet, ['in_channels', 'n_outputs', 'n_basefilters', 'n_blocks', 'layer_size', 'block_type',
        'optim_features', 'n_protos_per_class', 'proto_dims',
        'proto_activation_func', 'proto_nonlinearity'], None),
}

def layer_to_lp(target_attr, player):
    assert target_attr.__class__ in LP_LAYER_DICT.keys(), f"{target_attr}, {LP_LAYER_DICT.keys()}"
    new_layer, arguments, new_input_layer = LP_LAYER_DICT[target_attr.__class__]
    arguments = {arg_str: getattr(target_attr, arg_str) for arg_str in arguments}
    is_input = True if ((new_input_layer is not None) and (arguments['in_channels'] <= 3)) else False
    if "bias" in arguments.keys():
        arguments["bias"] = False if arguments["bias"] is None else True
    if is_input:
        arguments["playergen"] = player
        #arguments["baseline"] = Zero_Baseline()

    new_layer = new_input_layer(**arguments) if is_input else new_layer(**arguments)
    return new_layer


def exchange_layers(module, player):
    '''
    Recursively exchanges layers with probabilistic layers, as proposed by from Ancona et al.
    '''
    # replace all children of current module that are actual layers
    for attr_str, mm in module.named_children():
        if any([isinstance(mm, x) for x in LP_LAYER_DICT.keys()]):
            new_layer = layer_to_lp(mm, player)
            setattr(module, attr_str, new_layer)
        else:
            LOG.info(attr_str, f": no LP version found found for layer {mm.__class__}")
       

    # for all non-layer (i.e. Sequential etc.) sub-modules, recursively exchange layers
    for child_name, immediate_child_module in module.named_children():
        exchange_layers(immediate_child_module, player)


def parse_model(model, player):
    state_dict = copy.deepcopy(model.state_dict())
    exchange_layers(model, player)
    model.load_state_dict(state_dict)
    return model
