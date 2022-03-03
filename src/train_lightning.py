# -*- coding: utf-8 -*-
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torchvision.transforms as transforms

from models.mresunet import MResUNet
from models.resnet import ResNet
from map_parameters import MapParameters, initialize_camb

from lensingmapdatamodule import LensingMapDataModule

# Sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(42, workers=True)

results = initialize_camb("planck_2018_acc")
small_map_parameters = MapParameters(
    cambinifile="planck_2018_acc",
    M200_list=[1e14, 4e14],
    npix=64,
    lpix_amin=0.3,
    ellmaxsky=6000,
    z=0.7,
    profname="nfw",
)
# transform = transforms.Compose([
# transforms.Lambda(lambda x: print(x)),
# transforms.Lambda(lambda x: [i[None, :].float() for i in x]),
# ])
transform = None

dm = LensingMapDataModule(results, small_map_parameters, 44, 2, 4, transform)
print("Data module created")
dm.prepare_data()
print("Data prepared")
print(dm.training_dataset[0])


model = MResUNet(small_map_parameters.parameters["npix"])
print("Model created")


checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints",
    filename="clusterlens-{epoch:02d}-{val_loss:.2f}",
    mode="min",
    # save_top_k=3,
    save_last=True,
)

# early_stopping = EarlyStopping(monitor="val_loss")

logger = TensorBoardLogger(name="my_model", save_dir="save_dir")

trainer = Trainer(
    max_steps=10,
    deterministic=True,
    logger=logger,
    callbacks=[checkpoint_callback],
    profiler="simple",
)
trainer.fit(model, dm)
