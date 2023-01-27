import time

from pytorch_lightning.callbacks import Callback


class TimingCallback(Callback):
    def __init__(self):
        self.train_batch_time = []
        self.train_epoch_time = []
        self.validation_batch_time = []
        self.validation_epoch_time = []

    # ### FULL TRAINING
    # def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
    #     """Called when fit, validate, test, predict, or tune begins."""
    #     self.total_time_start = time.time()

    # def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
    #     """Called when fit, validate, test, predict, or tune ends."""
    #     total_time_end = time.time()-self.total_time_start
    #     self.log('total_time', total_time_end)

    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()

    def on_train_end(self, trainer, pl_module):
        train_end_time = time.time() - self.train_start_time

    ### BATCHES
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_batch_timer_start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        train_batch_time = time.time() - self.train_batch_timer_start
        self.train_batch_time.append(train_batch_time)
        self.log(
            "avg_train_batch_time",
            sum(self.train_batch_time) / len(self.train_batch_time),
        )

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        self.validation_batch_timer_start = time.time()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        validation_batch_time = time.time() - self.validation_batch_timer_start
        self.validation_batch_time.append(validation_batch_time)
        self.log(
            "avg_validation_batch_time",
            sum(self.validation_batch_time) / len(self.validation_batch_time),
        )

    ### EPOCHS
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_time_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        train_epoch_time_end = time.time() - self.train_epoch_time_start
        self.train_epoch_time.append(train_epoch_time_end)
        self.log(
            "avg_train_epoch_time",
            sum(self.train_epoch_time) / len(self.train_epoch_time),
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_epoch_timer_start = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        validation_epoch_time = time.time() - self.validation_epoch_timer_start
        self.validation_epoch_time.append(validation_epoch_time)
        self.log(
            "avg_validation_epoch_time",
            sum(self.validation_epoch_time) / len(self.validation_epoch_time)
        )



# times = []
# for _ in range(100):
#     start = time.time()
#     self.soc.log_metric(name=k, value=v)
#     times.append(time.time()-start)

# print(sum(times)/len(times))
