from __future__ import annotations

import os
import sys

cg_msl = "/store/code/ai4science/matsciml/"

if os.path.exists(cg_msl):
    sys.path.append(cg_msl)


sm_msl = "/workspace/ai-mat-top/matsciml_top/forks/carmelo_matsciml/"

if os.path.exists(sm_msl):
    sys.path.append(sm_msl)



import pytorch_lightning as pl
from torch.nn import LayerNorm, SiLU

from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import (
    ScalarRegressionTask,
    MultiTaskLitModule,
    GradFreeForceRegressionTask,
    ForceRegressionTask,
)

from data_config import *
from trainer_config import *
from training_utils import *
#
run_type = "experiment"
#

### Set up a log path:
# log_path = os.path.join(
#     "./experiments-2024-logs-full-runs/",
#     run_type,
#     model,
#     "-".join(data),
#     "-".join(targets),
# )

GPUS = 8
opt_target = "val.total_loss"

datasets = ["mp-traj", "gnome"]
model = "faenet"

log_path = os.path.join(
        "/workspace/nosnap/matsciml/full-runs/",
        run_type,
        model,
        "-".join(datasets),
    )

callbacks = setup_callbacks(opt_target, log_path)
logger = setup_logger(log_path)

model_kwargs = available_models[model]
energy_task_1 = ScalarRegressionTask(
    **model_kwargs,
    task_keys=["energy"],
    normalize_kwargs = available_data[datasets[0]][run_type]['normalize_kwargs'],
)
gffr_task_1 = GradFreeForceRegressionTask(
    **model_kwargs,
    normalize_kwargs = available_data[datasets[0]][run_type]['normalize_kwargs'],
)
energy_task_2 = ScalarRegressionTask(
    **model_kwargs,
    task_keys=["energy"],
    normalize_kwargs = available_data[datasets[1]][run_type]['normalize_kwargs'],
)
gffr_task_2 = GradFreeForceRegressionTask(
    **model_kwargs,
    normalize_kwargs = available_data[datasets[1]][run_type]['normalize_kwargs'],
)

train_dset_list = []
val_dset_list = []
for data in datasets:
    dset = deepcopy(available_data[data])
    dm_kwargs = deepcopy(available_data["generic"]["experiment"])
    dset[run_type].pop("normalize_kwargs", None)
    dm_kwargs.update(dset[run_type])
    dataset_name = dset["dataset"]
    dataset = getattr(sys.modules[__name__], dataset_name)
    model_transforms = transforms[model]
    train_dset_list.append(
        dataset(dm_kwargs["train_path"], transforms=model_transforms)
    )
    val_dset_list.append(dataset(dm_kwargs["val_split"], transforms=model_transforms))


train_dset = MultiDataset(train_dset_list)
val_dset = MultiDataset(val_dset_list)
dm = MultiDataModule(
    train_dataset=train_dset,
    val_dataset=val_dset,
    batch_size= 16, #dm_kwargs["batch_size"],
    num_workers=dm_kwargs["num_workers"],
)

# Hard coded for two datasets.
# Need to specify individual tasks for normalizations to be appropriate.
task = MultiTaskLitModule(
    (available_data[datasets[0]]["dataset"], energy_task_1),
    (available_data[datasets[0]]["dataset"], gffr_task_1),
    (available_data[datasets[1]]["dataset"], energy_task_2),
    (available_data[datasets[1]]["dataset"], gffr_task_2),
)

trainer_args = deepcopy(trainer_config["generic"])
trainer_args.update(trainer_config[run_type])
if run_type == "experiment":
    trainer_args.update({"devices": GPUS})

trainer = pl.Trainer(
    callbacks=callbacks, logger=logger, **trainer_args
)
trainer.fit(task, datamodule=dm)

trainer.model.to(device="cpu")
trainer.save_checkpoint("/workspace/nosnap/matsciml/checkpoints/faenet_sam_combo_full_multi_apr19_24.ckpt")
