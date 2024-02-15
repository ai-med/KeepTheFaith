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
from typing import Optional
from torch.utils.data import random_split
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import pandas as pd
import logging

from .birds import BIRDS_TRAIN_TRANSFORM, BIRDS_VAL_TRANSFORM
from .utils import INVERSE_NORM
LOG = logging.getLogger(__name__)


class DOGS(Dataset):

    def __init__(self, path: Path, is_train: bool):
        super().__init__()
        self.path = path
        self.basepath = path.parent.parent / "Images"

        self.df = pd.read_csv(path)
        self._images = []
        self._labels = []
        n_gray = 0
        for _, x in self.df.iterrows():
            
            self._images.append(self.basepath / x.file_list)
            self._labels.append(x.labels)
        # LOG.info(f"Number of grayscale images: {n_gray}\nTotal images: {len(self._images)}")

        if is_train:
            self.transform = BIRDS_TRAIN_TRANSFORM
        else:
            self.transform = BIRDS_VAL_TRANSFORM

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        imgpath = self._images[idx]
        with Image.open(imgpath, 'r') as img:
            img.load()
        if img.mode == 'L' or img.mode == 'RGBA':
            img = img.convert('RGB')
        label = self._labels[idx]
        return self.transform(img), label


class DogsDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, fold, batch_size, num_workers, metadata):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = BIRDS_TRAIN_TRANSFORM
        self.val_transform = BIRDS_VAL_TRANSFORM
        self.inverse_norm = INVERSE_NORM
        self.generator = torch.Generator().manual_seed(42)
        self.metadata = metadata

    def setup(self, stage: Optional[str] = None):

        self.train_data = DOGS(self.data_dir / f"splits/{self.fold}-train.csv", True)
        self.eval_data = DOGS(self.data_dir / f"splits/{self.fold}-valid.csv", False)
        self.test_data = DOGS(self.data_dir / f"splits/{self.fold}-test.csv", False)
        self.push_data = DOGS(self.data_dir / f"splits/{self.fold}-train.csv", False)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            generator=self.generator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers // 2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def push_dataloader(self):
        return DataLoader(
            self.push_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
