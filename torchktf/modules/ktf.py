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
import math
from pathlib import Path
from typing import Any, List, Optional
import warnings

import cv2
import numpy as np
# from skimage.transform import resize

from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torchmetrics import MaxMetric

from .base import BaseModule
from .utils import compute_rf_prototype, compute_proto_layer_rf_info_v2, find_high_activation_crop, init_pretrained_weights

LOG = logging.getLogger(__name__)

STAGE2FLOAT = {"warmup": 0.0, "all": 1.0, "clf": 2.0}


class KeepTheFaith(BaseModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        lr_lastlayer: float = 0.0001,
        weight_decay: float = 0.0005,
        l_clst: float = 0.8,
        l_sep: float = 0.08,
        epochs_all: int = 3,
        epochs_clf: int = 4,
        epochs_warmup: int = 10,
        enable_checkpointing: bool = True,
        monitor_prototypes: bool = False,  # wether to save prototypes of all push epochs or just the best one
        enable_save_embeddings: bool = False,
        enable_log_prototypes: bool = False,
        pretrained_weights: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            net=net,
            num_classes=net.n_outputs
        )
        self.automatic_optimization = False

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        self._current_stage = None
        self._current_optimizer = None

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_bacc_save = MaxMetric()

        if pretrained_weights is not None:
            self.net = init_pretrained_weights(self.net, pretrained_weights)


    def _cluster_losses(self, similarities, y):
        prototypes_of_correct_class = self.net.p_classmapping[:, y].T.bool()
        # select mask for each sample in batch. Shape is (bs, n_prototypes)
        other_tensor = torch.tensor(torch.inf, device=similarities.device)  # maximum value of l2norm
        similarities_correct_class = similarities.where(prototypes_of_correct_class, other=other_tensor)
        # ^ min_pool will find the minimum in this tensor, as the distance of wrong classes is inf
        similarities_incorrect_class = similarities.where(~prototypes_of_correct_class, other=other_tensor)
        # ^ same here, min pool will ignore inf of prototypes of correct class
        clst = torch.min(similarities_correct_class, dim=1).values.mean()
        sep = torch.min(similarities_incorrect_class, dim=1).values.mean()

        clst = clst / self.net.n_protos
        sep = - sep / self.net.n_protos

        return clst, sep

    def _classification_loss(self, logits, targets):
        preds = torch.argmax(logits, dim=1)

        xentropy = self.criterion(logits, targets)
        return xentropy, preds

    def on_train_epoch_start(self) -> None:
        cur_stage = self._get_current_stage()
        LOG.debug("Epoch %d, optimizing %s", self.trainer.current_epoch, cur_stage)

        self.log("train/stage", STAGE2FLOAT[cur_stage])

        optim_warmup, optim_all, optim_clf = self.optimizers()
        scheduler_warmup, scheduler_all, scheduler_clf = self.lr_schedulers()
        if cur_stage == "warmup":
            opt = optim_warmup
            sched = scheduler_warmup
        elif cur_stage == "all":
            opt = optim_all
            sched = scheduler_all
        elif cur_stage == "clf":
            opt = optim_clf
            sched = scheduler_clf
            if self.push_protos:
                self.push_prototypes()
        else:
            raise AssertionError()

        self._current_stage = cur_stage
        self._current_optimizer = opt
        self._current_scheduler = sched

    def training_step(self, batch, batch_idx: int):
        cur_stage = self._current_stage

        images, y = batch

        logits, min_distances, conv_features, distances = self.forward(images)
        xentropy, preds = self._classification_loss(logits, y)
        self.log("train/xentropy", xentropy)
        losses = [xentropy]

        if cur_stage != "clf":
            # cluster and seperation cost
            clst, sep = self._cluster_losses(min_distances, y)
            losses.append(self.hparams.l_clst * clst)
            losses.append(self.hparams.l_sep * sep)

            self.log("train/clst", clst)
            self.log("train/sep", sep)

        loss = sum(losses)

        opt = self._current_optimizer
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self._current_scheduler.step()

        self._log_train_metrics(loss, preds, y)

        return {"loss": loss, "preds": preds, "targets": y}

    def _log_train_metrics(
        self, loss: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor,
    ) -> None:
        self.log(f"train/loss/{self._current_stage}", loss, on_step=False, on_epoch=True, prog_bar=False)
        acc = self.train_acc(preds, targets)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def _get_current_stage(self, epoch=None):
        total = self.hparams.epochs_all + self.hparams.epochs_clf

        stage = "clf"
        if self.current_epoch < self.hparams.epochs_warmup:
            stage = "warmup"
        elif (self.current_epoch - self.hparams.epochs_warmup) % total < self.hparams.epochs_all:
            stage = "all"

        self.push_protos = False
        if (self.current_epoch - self.hparams.epochs_warmup) % total == self.hparams.epochs_all:
            self.push_protos = True


        return stage

    def training_epoch_end(self, outputs: List[Any]):
        if self.hparams.enable_log_prototypes and isinstance(self.logger, TensorBoardLogger):
            tb_logger = self.logger.experiment

            tb_logger.add_histogram(
                "train/prototypes", self.net.prototypes, global_step=self.trainer.global_step,
            )

        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx: int):
        images, targets = batch
        logits, _, _, _ = self.forward(images)

        loss, preds = self._classification_loss(logits, targets)

        self._update_validation_metrics(loss, preds, targets)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # compute balanced accuracy
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cmat)

        cur_stage = self._current_stage
        # every 10th epoch is a last layer optim epoch
        if cur_stage == "clf":
            self.val_bacc_save.update(bacc)
            saver = bacc
        else:
            self.val_bacc_save.update(torch.tensor(0., dtype=torch.float32))
            saver = torch.tensor(-float('inf'), dtype=torch.float32, device=bacc.device)
        self.log("val/bacc_save_monitor", self.val_bacc_save.compute(), on_epoch=True)
        self.log("val/bacc_save", saver)

        self._log_validation_metrics()

    def test_step(self, batch, batch_idx: int):
        images, targets = batch
        logits, _, _, _= self.forward(images)

        targets = batch[-1]
        loss, preds = self._classification_loss(logits, targets)

        self._update_test_metrics(loss, preds, targets)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self._log_test_metrics()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        clf = {
            'params': [p for p in self.net.clf.parameters() if p.requires_grad],
            'lr': self.hparams.lr_lastlayer,
            'weight_decay': 0.0,
        }
        prototypes = {
            'params': [self.net.prototypes],
            'lr': self.hparams.lr,
            'weight_decay': 0.0,
        }
        feat_ext = {
            'params': [p for p in self.net.feat_extractor.parameters() if p.requires_grad],
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay,
        }
        encoder = {
            'params': [p for p in self.net.blocks.parameters() if p.requires_grad],
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay,
        }
        if len(encoder['params']) == 0:
            warnings.warn("Encoder seems to be frozen! No parameters require grad.")
        assert len(clf['params']) > 0
        assert len(prototypes['params']) > 0
        assert len(feat_ext['params']) > 0
        assert len(encoder['params']) > 0
        optim_all = torch.optim.AdamW([
            encoder, feat_ext, prototypes
         ])
        optim_clf = torch.optim.AdamW([
            clf
        ])
        optim_warmup = torch.optim.AdamW([
            feat_ext, prototypes
        ])

        training_iterations = len(self.trainer.datamodule.train_dataloader())
        LOG.info("Number of iterations for one epoch: %d", training_iterations)

        # stepping through warmup_protonet is sufficient, as all parameters groups are also in warmup
        scheduler_kwargs = {'max_lr': self.hparams.lr, 'cycle_momentum': False}
        scheduler_warmup = torch.optim.lr_scheduler.CyclicLR(
            optim_warmup,
            base_lr=self.hparams.lr / 10,
            max_lr=self.hparams.lr / 2,
            step_size_up=self.hparams.epochs_warmup * training_iterations,
            cycle_momentum=False
        )
        scheduler_all = torch.optim.lr_scheduler.CyclicLR(
            optim_all,
            base_lr=self.hparams.lr / 2,
            step_size_up=(self.hparams.epochs_all * training_iterations) / 4,
            step_size_down=(self.hparams.epochs_all * training_iterations) / 4,
            **scheduler_kwargs
        )
        scheduler_clf = torch.optim.lr_scheduler.CyclicLR(
            optim_clf,
            base_lr=self.hparams.lr / 2,
            step_size_up=(self.hparams.epochs_clf * training_iterations) / 4,
            step_size_down=(self.hparams.epochs_clf * training_iterations) / 4,
            **scheduler_kwargs
        )

        return ([optim_warmup, optim_all, optim_clf],
                [scheduler_warmup, scheduler_all, scheduler_clf])


    def push_prototypes(self):

        self.eval()

        prototype_shape = self.net.prototypes.shape
        n_prototypes = self.net.n_protos

        global_min_proto_dist = np.full(n_prototypes, np.inf)
        global_min_fmap_patches = np.zeros(
            (n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]))
        global_min_img_indices = np.zeros((n_prototypes), dtype=int)

        proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)

        proto_epoch_dir = Path(self.trainer.log_dir) / f"prototypes_epoch_{self.current_epoch}"
        if self.hparams.enable_checkpointing:
            proto_epoch_dir.mkdir(exist_ok=True)

        push_dataloader = self.trainer.datamodule.push_dataloader()
        search_batch_size = push_dataloader.batch_size

        num_classes = self.net.n_classes

        for push_iter, (search_batch_input, search_y) in enumerate(push_dataloader):

            start_index_of_search_batch = push_iter * search_batch_size

            self.update_prototypes_on_batch(search_batch_input,
                                            start_index_of_search_batch,
                                            global_min_proto_dist,
                                            global_min_fmap_patches,
                                            global_min_img_indices,
                                            num_classes,
                                            search_y,
                                            proto_rf_boxes,
                                            proto_bound_boxes,
                                            proto_epoch_dir,
                                            )

        if self.hparams.enable_checkpointing:
            np.save(proto_epoch_dir / "receptive_field.npy", proto_rf_boxes)
            np.save(proto_epoch_dir / "bounding_boxes.npy", proto_bound_boxes)
            np.save(proto_epoch_dir / "img_indices.npy", global_min_img_indices)

        prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
        self.net.prototypes.data.copy_(torch.tensor(prototype_update, dtype=torch.float32, device=self.device))


    def update_prototypes_on_batch(self,
                                   search_batch,
                                   start_index_of_search_batch,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   global_min_img_indices,
                                   num_classes,
                                   search_y,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   proto_epoch_dir,
                                   ):

        self.eval()

        with torch.no_grad():
            _, _, protoL_input_torch, proto_dist_torch = self.net(search_batch.to(self.device))

        protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

        class_to_img_index = {key: [] for key in range(num_classes)}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index[img_label].append(img_index)

        prototype_shape = self.net.prototypes.shape
        n_prototypes = prototype_shape[0]
        proto_h, proto_w = prototype_shape[2:]
        max_dist = math.prod(prototype_shape[1:])

        for j in range(n_prototypes):
            target_class = torch.argmax(self.net.p_classmapping[j]).item()
            if len(class_to_img_index[target_class]) == 0:  # none of the images belongs to the class of this prototype
                continue
            proto_dist_j = proto_dist_[class_to_img_index[target_class]][:, j, :, :]
            # distnces of all latents to the j-th prototype of this class within the batch

            batch_min_proto_dist_j = np.amin(proto_dist_j)  # minimum distance of latents of this batch to prototype j

            if batch_min_proto_dist_j < global_min_proto_dist[j]:  # save if a new min distance is present in this batch

                batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                                                  proto_dist_j.shape))
                # coordinates of minimum distance of latents of this batch to prototype j
                batch_argmin_proto_dist_j[0] = class_to_img_index[target_class][batch_argmin_proto_dist_j[0]]
                # get batch index instead of class specific batch

                img_index_in_batch = batch_argmin_proto_dist_j[0]
                fmap_height_start_index = batch_argmin_proto_dist_j[1]
                fmap_height_end_index = fmap_height_start_index + proto_h
                fmap_width_start_index = batch_argmin_proto_dist_j[2]
                fmap_width_end_index = fmap_width_start_index + proto_w

                batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                       :,
                                                       fmap_height_start_index:fmap_height_end_index,
                                                       fmap_width_start_index:fmap_width_end_index
                                                       ]
                # latent vector of minimum distance
                global_min_proto_dist[j] = batch_min_proto_dist_j
                global_min_fmap_patches[j] = batch_min_fmap_patch_j
                global_min_img_indices[j] = img_index_in_batch + start_index_of_search_batch

                if self.hparams.enable_checkpointing:

                    layer_filter_sizes, layer_strides, layer_paddings = self.net.get_conv_info()
                    protoL_rf_info = compute_proto_layer_rf_info_v2(search_batch.size()[2],
                                                                          layer_filter_sizes=layer_filter_sizes,
                                                                          layer_strides=layer_strides,
                                                                          layer_paddings=layer_paddings,
                                                                          prototype_kernel_size=prototype_shape[2])
                    rf_prototype_j = compute_rf_prototype(search_batch.size()[2],
                                                                batch_argmin_proto_dist_j,
                                                                protoL_rf_info)

                    # original image
                    original_img_j = search_batch[rf_prototype_j[0]].detach().cpu()
                    if self.trainer.datamodule.inverse_norm is not None:
                        original_img_j = self.trainer.datamodule.inverse_norm(original_img_j).numpy()
                    else:
                        original_img_j = original_img_j.numpy()
                    original_img_j = np.transpose(original_img_j, (1, 2, 0))
                    original_img_size = original_img_j.shape[0]

                    # crop receptive field
                    rf_img_j = original_img_j[ rf_prototype_j[1]:rf_prototype_j[2],
                                              rf_prototype_j[3]:rf_prototype_j[4], :]

                    # save proto receptive field inof
                    proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
                    for q in range(1, 4):
                        proto_rf_boxes[j, q] = rf_prototype_j[q]
                    proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

                    # find highly activated region of the original image
                    proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
                    if (self.net.prototype_activation_function == 'linear') and ((self.net.proto_nonlinearity == 'relu1') or (self.net.proto_nonlinearity == 'sigmoid')):
                        proto_act_img_j = max_dist - proto_dist_img_j
                    elif self.net.prototype_activation_function == 'log':
                        proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + self.net.eps))
                    else:
                        assert False, "invalid activation function"
                    upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                                     interpolation=cv2.INTER_CUBIC)

                    proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
                    proto_img_j = original_img_j[ proto_bound_j[0]:proto_bound_j[1],
                                                 proto_bound_j[2]:proto_bound_j[3], :]

                    # save prototype boundary (rectangular boundary of highly activated region)
                    proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
                    for q in range(1, len(proto_rf_boxes[j])-1):
                        proto_bound_boxes[j, q] = proto_bound_j[q-1]
                    proto_bound_boxes[j, len(proto_rf_boxes[j])-1] = search_y[rf_prototype_j[0]].item()

                    np.save(proto_epoch_dir / f"p_selfactivation_{j}.npy", proto_act_img_j[:, :, np.newaxis])
                    np.save(proto_epoch_dir / f"original_{j}.npy", original_img_j)
                    np.save(proto_epoch_dir / f"rf_{j}.npy", rf_img_j)
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    np.save(proto_epoch_dir / f"original_with_self_act_{j}.npy", overlayed_original_img_j)
                    overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                rf_prototype_j[3]:rf_prototype_j[4]]
                    np.save(proto_epoch_dir / f"rf_with_self_act_{j}.npy", overlayed_rf_img_j)

                    np.save(proto_epoch_dir / f"p_{j}.npy", proto_img_j)

        del class_to_img_index

