# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mresunet import MResUNet


class SupervisedAttentionModule(nn.Module):
    """
    A Supervised Attention Module.
    """

    def __init__(
        self,
        num_channels=3,  # The number of channels of the input
        input_channels=3,  # The number of channels of the original image input `i` (3 if tqu, 1 if t)
    ):
        super(SupervisedAttentionModule, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, input_channels, 1)
        self.conv2 = nn.Conv2d(num_channels, input_channels, 1)
        self.conv3 = nn.Conv2d(num_channels, input_channels, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, i):

        restored = self.conv1(x) + i
        attention_maps = self.sig(self.conv2(restored))
        f_in = self.conv3(x)

        return restored, (attention_maps * f_in) + i


class MSPR(pl.LightningModule):
    """
    A Multi-Stage Progressive Restoration using Supervised Attention Modules
    """

    def __init__(
        self,
        map_size=40,
        lr=0.001,
        input_channels=3,
        nb_enc_boxes=4,
        nb_channels_first_box=64,
        output_type="kappa_map",
        loss="mse",
        final_channels=3,
        **kwargs
    ):
        super(MSPR, self).__init__()

        self.lr = lr

        self.mresunet1 = MResUNet(
            map_size,
            lr,
            input_channels,
            nb_enc_boxes,
            nb_channels_first_box,
            output_type="kappa_map",
            loss="mse",
            final_channels=input_channels,
        )
        self.mresunet2 = MResUNet(
            map_size,
            lr,
            input_channels,
            nb_enc_boxes,
            nb_channels_first_box,
            output_type="kappa_map",
            loss="mse",
            final_channels=input_channels,
        )

        self.sam = SupervisedAttentionModule(
            num_channels=input_channels, input_channels=final_channels
        )

        self.conv1 = nn.Conv2d(input_channels, input_channels, 1)
        self.conv2 = nn.Conv2d(input_channels, input_channels, 1)
        self.reduce_channel = nn.Conv2d(2 * input_channels, input_channels, 1)

        if loss == "msle":
            self.loss = lambda x, y: F.mse_loss(torch.log(x + 1), torch.log(y + 1))
        else:
            self.loss = F.mse_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MResUNet")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--nb_enc_boxes", type=int, default=4)
        parser.add_argument("--nb_channels_first_box", type=int, default=64)
        return parent_parser

    def forward(self, x):
        x1 = self.mresunet1(self.conv1(x))
        restored1, fout = self.sam(x1, i=x)

        x2 = torch.cat((fout, self.conv2(x)), 1)
        x2 = self.reduce_channel(x2)
        restored2 = self.mresunet2(x2)
        return restored1, restored2

    def configure_optimizers(self, step_size=1):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
        return [optimizer]  # , [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        restored1, restored2 = self(x)

        loss = self.loss(restored1, y) + self.loss(restored2, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        r1, r2 = self(x)
        val_loss = self.loss(r1, y) + self.loss(r2, y)
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        _, r = self(x)
        return r


class ProgressiveMassEstimation(pl.LightningModule):
    """Denoise the Q and U maps with two MResUNets, and get the kappa_map from
    the T and the denoised Q and U with another MResUNet."""

    def __init__(
        self,
        map_size=40,
        lr=0.001,
        input_channels=3,
        nb_enc_boxes=4,
        nb_channels_first_box=64,
        loss="mse",
        final_channels=3,
        **kwargs
    ):
        super(ProgressiveMassEstimation, self).__init__()

        self.lr = lr

        self.unets = nn.ModuleList(
            [
                MResUNet(
                    map_size,
                    lr,
                    1,
                    nb_enc_boxes,
                    nb_channels_first_box,
                    output_type="kappa_map",
                    loss="mse",
                    final_channels=1,
                )
                for _ in range(2)
            ]
        )

        self.final_unet = MResUNet(
            map_size,
            lr,
            3,
            nb_enc_boxes,
            nb_channels_first_box,
            output_type="kappa_map",
            loss="mse",
            final_channels=1,
        )

        if loss == "msle":
            self.loss = lambda x, y: F.mse_loss(torch.log(x + 1), torch.log(y + 1))
        else:
            self.loss = F.mse_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MResUNet")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--nb_enc_boxes", type=int, default=4)
        parser.add_argument("--nb_channels_first_box", type=int, default=64)
        return parent_parser

    def forward(self, x):
        denoised = [
            self.unets[0](x[:, [1], :, :]),
            self.unets[1](x[:, [2], :, :]),
        ]

        kappa_map = self.final_unet(
            torch.cat((x[:, [0], :, :], denoised[0], denoised[1]), 1)
        )

        return denoised, kappa_map

    def configure_optimizers(self, step_size=1):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y = batch
        denoised, kappa_map = self(x)

        loss = sum(
            [self.loss(denoised[i], y[1][:, [i+1], :, :]) for i in range(2)]
        ) + 50 * self.loss(kappa_map, y[0])

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        denoised, kappa_map = self(x)

        val_loss_q = self.loss(denoised[0], y[1][:, [1], :, :])
        val_loss_u = self.loss(denoised[1], y[1][:, [2], :, :])
        val_loss_kappa = self.loss(kappa_map, y[0])

        self.log("val_loss_q", val_loss_q)
        self.log("val_loss_u", val_loss_u)
        self.log("val_loss_kappa", val_loss_kappa)
        self.log("val_loss", val_loss_q + val_loss_kappa * 50 + val_loss_u)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        _, r = self(x)
        return r
