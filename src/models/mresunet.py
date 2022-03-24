# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncodingBoxSubStage(nn.Module):
    """
    An encoding sub stage that performs convolution,
    activation, batch normalization, and addition.
    (No dropout).
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        dilation=1,
        stride=1,
        padding=1,
        activation=nn.SELU(),
    ):
        super(EncodingBoxSubStage, self).__init__()

        self.activation_function = activation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.activation_function(y)
        y = self.batch_norm(y)

        # Addition (Residual connection)
        if self.conv.stride == 1:
            return x + y

        return y


class DecodingBoxSubStage(EncodingBoxSubStage):
    """
    A decoding sub stage.
    Introduces dropout before every convolution.
    Can concatenate the input tensor to output of a previous sub stage.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        dilation=1,
        stride=1,
        padding=1,
        activation=nn.SELU(),
        dropout=0.3,
        concatenate=False,
    ):
        super().__init__(
            in_channels * (2 if concatenate else 1),
            out_channels,
            kernel_size,
            dilation,
            stride,
            padding,
            activation,
        )
        self.concatenate = concatenate
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, concat_with=None):
        x = self.dropout(x)

        if self.concatenate:
            x = torch.cat((concat_with, x), 1)

        y = self.conv(x)
        y = self.activation_function(y)
        y = self.batch_norm(y)

        # Addition (Residual connection)
        if self.conv.stride == 1:
            return x + y

        return y


class EncodingBox(nn.Module):
    """
    An encoding box that performs one encoding cycle.
    Uses four sub-stages with dilation rates
    1, 2, 3, 4. Outputs the results of the fourth
    and second sub stages.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        activation=nn.SELU(),
        rescale=True,
    ):
        super(EncodingBox, self).__init__()

        self.sub_stages = nn.ModuleList(
            [
                EncodingBoxSubStage(
                    in_channels=(in_channels if dilation == 1 else out_channels),
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=(2 if (dilation == 1 and rescale) else 1),
                    padding=dilation,
                    activation=activation,
                )
                for dilation in range(1, 5)
            ]
        )

    def forward(self, x):
        x = self.sub_stages[0](x)
        d2 = self.sub_stages[1](x)
        x = self.sub_stages[2](d2)
        x = self.sub_stages[3](x)

        return x, d2


class DecodingBox(nn.Module):
    """
    A helper module that performs one decoding cycle.
    Uses four sub stages with dilation rates
    4, 3, 2, 1. Concatenates `d4` and `d2` to the
    inputs of the first and third sub stage.
    """

    def __init__(
        self,
        in_channels=1,  # `The in_channels of the first sub stage`
        out_channels=1,  # The `out_channels` and `in_channels` of the middle sub stages.
        final_channels=None,  # `The out_channels` of the last sub stage
        kernel_size=3,
        stride=1,
        activation=nn.SELU(),
        final_activation=nn.SELU(),
        dropout=0.2,
    ):
        super(DecodingBox, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.sub_stages = nn.ModuleList(
            [
                DecodingBoxSubStage(
                    in_channels=(in_channels if (dilation == 4) else out_channels),
                    out_channels=(
                        out_channels
                        if (final_channels is None or dilation != 1)
                        else final_channels
                    ),
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                    padding=dilation,
                    activation=(activation if (dilation != 1) else final_activation),
                    dropout=dropout,
                    concatenate={1: False, 2: True, 3: False, 4: True}[dilation],
                )
                for dilation in reversed(range(1, 5))
            ]
        )

    def forward(self, x, d4=None, d2=None):
        # Perform upsampling
        x = self.upsample(x)

        x = self.sub_stages[0](x, d4)
        x = self.sub_stages[1](x)
        x = self.sub_stages[2](x, d2)
        x = self.sub_stages[3](x)

        return x


class MResUNet(pl.LightningModule):
    """
    The modified Residual U-Net as specified in https://arxiv.org/pdf/2003.06135.pdf
    """

    def __init__(
        self,
        map_size=40,
        lr=0.001,
        input_channels=1,
        nb_enc_boxes=4,
        nb_channels_first_box=64,
        output_type="kappa_map",
        loss="mse",
        final_channels=1,
        **kwargs
    ):
        super(MResUNet, self).__init__()

        self.lr = lr
        self.output_type = output_type
        self.input_channels = input_channels
        self.nb_enc_boxes = nb_enc_boxes
        self.nb_channels_first_box = nb_channels_first_box

        if loss == "msle":
            self.loss = lambda x, y: F.mse_loss(torch.log(x + 1), torch.log(y + 1))
        else:
            self.loss = F.mse_loss

        # `nb_enc_boxes` encoding boxes
        self.encoding = nn.ModuleList(
            [
                EncodingBox(
                    in_channels=input_channels,
                    out_channels=nb_channels_first_box,
                    kernel_size=3,
                    rescale=False,
                ),
                *[
                    EncodingBox(
                        in_channels=(2**i) * nb_channels_first_box,
                        out_channels=(2 ** (i + 1)) * nb_channels_first_box,
                        kernel_size=3,
                    )
                    for i in range(nb_enc_boxes - 1)
                ],
            ]
        )

        # (`nb_enc_boxes` - 1) decoding boxes
        self.decoding = nn.ModuleList(
            [
                *[
                    DecodingBox(
                        in_channels=(2 ** (i + 1)) * nb_channels_first_box,
                        out_channels=(2 ** (i + 1)) * nb_channels_first_box,
                        final_channels=(2 ** (i)) * nb_channels_first_box,
                        kernel_size=3,
                        dropout=0.2,
                    )
                    for i in reversed(range(nb_enc_boxes - 2))
                ],
                DecodingBox(
                    in_channels=nb_channels_first_box,
                    out_channels=nb_channels_first_box,
                    final_channels=final_channels,
                    kernel_size=3,
                    dropout=0.2,
                ),
            ]
        )

        # A convolution to divide the number of channels by 2 before the decoding stage
        self.reduce_channel = nn.Conv2d(
            in_channels=(2 ** (nb_enc_boxes - 1)) * nb_channels_first_box,
            out_channels=(2 ** (nb_enc_boxes - 2)) * nb_channels_first_box,
            kernel_size=3,
            padding=1,
        )

        if self.output_type == ["mass"]:
            self.avg = nn.AvgPool2d(map_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MResUNet")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--nb_enc_boxes", type=int, default=4)
        parser.add_argument("--nb_channels_first_box", type=int, default=64)
        return parent_parser

    def forward(self, x):

        # Store outputs of sub-stages with dilation rates: 2, 4
        d4_list = []
        d2_list = []

        # Encoding
        for box in self.encoding:
            x, d = box(x)
            d4_list.append(torch.clone(x))
            d2_list.append(torch.clone(d))

        # Decoding
        x = self.reduce_channel(x)
        for i, box in enumerate(self.decoding):
            x = box(
                x,
                d4=d4_list[self.nb_enc_boxes - 2 - i],
                d2=d2_list[self.nb_enc_boxes - 2 - i],
            )

        # if self.output_type == "kappa_map":
        # x = F.relu(x)

        if self.output_type == ["mass"]:
            x = self.avg(x)

        return x

    def configure_optimizers(self, step_size=1):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
        return [optimizer]  # , [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred
