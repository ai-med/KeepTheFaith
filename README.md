# Keep the Faith: Faithful Explanations in Convolutional Neural Networks for Case-Based Reasoning

[![Preprint](https://img.shields.io/badge/arXiv-2312.09783-b31b1b)](https://arxiv.org/abs/2312.09783)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

This repository contains the to the paper "Keep the Faith: Faithful Explanations in Convolutional Neural Networks for Case-Based Reasoning"

If you use this code, please cite the following:

```
@article{wolf2023keep,
  title={Keep the Faith: Faithful Explanations in Convolutional Neural Networks for Case-Based Reasoning},
  author={Wolf, Tom Nuno and Bongratz, Fabian and Rickmann, Anne-Marie and P{\"o}lsterl, Sebastian and Wachinger, Christian},
  journal={arXiv preprint arXiv:2312.09783},
  year={2023}
}
```

## Installation

First, create and activate the conda environment
```
conda env create --file requirements.yaml
pip install --no-deps -e .
conda activate ktf
```

# Usage

In order to train a model, hydra requires the `data_dir` variable to be set to the folder which contains the data, e.g. `/home/datasets`:
```
python train.py data_dir=/home/datasets
```
Other config variables, e.g. learning rate, model, etc., can be set by appending them to above command call.

Testing a model is done via the test script, which requires the `ckpt_path` variable to be set. This variable is the path to the pytorch lightning checkpoint of a trained model, e.g. `/home/model/checkpoints/epoch=99-bacc.ckpt`:
```
python test.py data_dir=/home/datasets ckpt_path='/home/model/checkpoints/epoch\=99-bacc.ckp'
```

Utility functions for explanations are available in `explain.py`.
