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
import logging
from typing import Optional
from pathlib import Path

import pytorch_lightning as pl
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tvtransforms

LOG = logging.getLogger(__name__)


def load_pil(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img


class RSNADataSet(Dataset):

    def __init__(self, path: Path, is_train: bool):

        super().__init__()
        self.path = path
        if is_train:
            self.transforms = tvtransforms.Compose([
                tvtransforms.RandomHorizontalFlip(),
                tvtransforms.RandomAffine(45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                tvtransforms.ToTensor()])  # rescales to 0 to 1
        else:
            self.transforms = tvtransforms.Compose([
                tvtransforms.ToTensor()])

        self._load()

    def _load(self):
        csv = pd.read_csv(self.path.parent / (self.path.name + ".csv"))
        self._images = []
        self._labels = []
        for img_path in self.path.iterdir():
            assert img_path.suffix == ".png"
            with Image.open(img_path) as img:
                img.load()
            self._images.append(img)
            label = csv.loc[csv.PTID == img_path.stem]
            assert len(label.index) == 1
            self._labels.append(label.Target.item())

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):

        img = self._images[idx]
        label = self._labels[idx]

        return self.transforms(img), label


class RSNADataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, num_workers, metadata):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.inverse_norm = None
        self.num_workers = num_workers
        self.metadata = metadata

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit' or stage is None:
            self.train_data = RSNADataSet(
                self.data_dir / "train",
                is_train=True
            )
            self.eval_data = RSNADataSet(
                self.data_dir / "val",
                is_train=False
            )

        elif stage == 'test':
            self.test_data = RSNADataSet(
                self.data_dir / "test",
                is_train=False
            )
            self.eval_data = RSNADataSet(
                self.data_dir / "val",
                is_train=False
            )

        self.push_data = RSNADataSet(
            self.data_dir / "train",
            is_train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
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
            num_workers=self.num_workers)
