# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch
import torchmetrics
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
        masses=None,
        final_relu=False,
        **kwargs,
    ):
        super(MResUNet, self).__init__()

        self.lr = lr
        self.output_type = output_type
        self.input_channels = input_channels
        self.nb_enc_boxes = nb_enc_boxes
        self.nb_channels_first_box = nb_channels_first_box
        self.masses = masses
        self.final_relu = final_relu

        if loss == "msle":
            self.loss = lambda x, y: F.mse_loss(
                torch.log(x + 1e-14), torch.log(y + 1e-14)
            )
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

        if ["mass"] == self.output_type:
            # self.fc = torch.nn.Linear(in_features=(final_channels-1)*64*64, out_features=(final_channels-1)*64*64)
            self.avg = nn.AvgPool2d(map_size)
            self.fc_mass = torch.nn.Linear(
                in_features=final_channels * 64 * 64, out_features=1
            )
        # else:
        # self.fc = torch.nn.Linear(in_features=final_channels*64*64, out_features=final_channels*64*64)

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

        if ["mass"] == self.output_type:
            mass = self.fc_mass(x.view(x.shape[0], -1)).view((x.shape[0], 1, 1, 1))
            return mass

        # Final linear layer
        # x = self.fc(x.view(x.shape[0], -1)).view(x.shape)

        if self.final_relu:
            x = F.relu(x)

        return x

    def configure_optimizers(self, step_size=1):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
        return [optimizer]  # , [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        """
        if "mass" in self.output_type:
            # Put mass to front
            y.insert(0, y.pop(self.output_type.index("mass")))
            y_hat, mass = self(x)
            loss = self.loss(y_hat, y[1:])
            mass_loss = self.loss(mass, y[0])
            train_loss = (mass_loss + loss)/2
            self.log("train_loss", train_loss)
            self.log("map_train_loss", loss)
            self.log("mass_train_loss", mass_loss)
            return train_loss
        """

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss)

        if "mass" in self.output_type:
            # Return statistics to plot graphs
            indexes = np.array(
                [(np.abs(yi.item() - self.masses)).argmin() for yi in y],
            )
            guesses = np.array(
                [(np.abs(yi.item() - self.masses)).argmin() for yi in y_hat],
            )
            return {
                "val_loss": val_loss,
                "y": y,
                "y_hat": y_hat,
                "y_i": indexes,
                "y_hat_i": guesses,
            }

    def validation_epoch_end(self, outputs):
        """Plot graphs to writer at the end of each validation epoch"""

        if "mass" in self.output_type:
            preds_i = torch.Tensor(
                np.concatenate([tmp["y_hat_i"] for tmp in outputs])
            ).int()
            targets_i = torch.Tensor(
                np.concatenate([tmp["y_i"] for tmp in outputs])
            ).int()

            preds = torch.Tensor(np.concatenate([tmp["y_hat"] for tmp in outputs]))
            targets = torch.Tensor(np.concatenate([tmp["y"] for tmp in outputs]))

            num_classes = len(self.masses)

            # Plot confusion matrix
            confusion_matrix = torchmetrics.ConfusionMatrix(
                num_classes=num_classes, normalize="true"
            )(preds_i, targets_i)

            df_cm = pd.DataFrame(
                confusion_matrix.numpy(),
                index=range(num_classes),
                columns=range(num_classes),
            )
            plt.figure(figsize=(10, 7))
            fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
            plt.close(fig_)

            plt.plot()

            self.logger.experiment.add_figure(
                "Confusion matrix", fig_, self.current_epoch
            )

            # Show table of statistics of prediction
            pred_std_mean = []
            print(targets)
            for i in sorted(np.unique(targets)):
                tmp = preds[(targets == i)]
                pred_std_mean.append(torch.std_mean(tmp))
            pred_std_mean = np.array(pred_std_mean)

            fig, axs = plt.subplots(1, 1)
            axs.axis("off")
            axs.table(
                cellText=np.array(pred_std_mean),
                rowLabels=sorted(np.unique(targets)),
                colLabels=["std", "mean"],
                loc="center",
            )
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

            self.logger.experiment.add_figure(
                "Prediction table", fig, self.current_epoch
            )

            fig, axs = plt.subplots(1, 1)
            axs.errorbar(
                x=sorted(np.unique(targets)),
                y=np.nan_to_num(pred_std_mean[:, 1], nan=0),
                yerr=np.nan_to_num(pred_std_mean[:, 0], nan=0),
                linestyle="None",
                marker="^",
            )

            self.logger.experiment.add_figure(
                "Prediction plot", fig, self.current_epoch
            )

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred
