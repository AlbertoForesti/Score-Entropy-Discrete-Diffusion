import numpy as np
from infosedd.infosedd import InfoSEDD
from infosedd.utils import array_to_dataset, get_infinite_loader

from datetime import datetime
from pytorch.utils.data import DataLoader
import pytorch_lightning as pl
import os

class TaskManager:

    def __init__(self, config):
        self.config = config
    
    def _load_infosedd(self):
        if hasattr(self.confg, "checkpoint_path"):
            self.infosedd = InfoSEDD.load(self.config.checkpoint_path, self.config)
        else:
            self.infosedd = InfoSEDD(self.config)
        return self.infosedd
    
    def _trainer_setup(self):
        CHECKPOINT_DIR = self.config.training.checkpoint_dir
        # Add date and time to checkpoint directory
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, timestamp)

        self.logger=pl.loggers.TensorBoardLogger(save_dir=CHECKPOINT_DIR)
        
        self.trainer = pl.Trainer(logger=self.logger,
                         default_root_dir=CHECKPOINT_DIR,
                         accelerator=self.args.training.accelerator,
                         devices=self.args.training.devices,
                         max_steps=self.args.training.max_steps,
                         max_epochs=None,
                         check_val_every_n_epoch=None,
                         val_check_interval=self.args.training.val_check_interval,
                         gradient_clip_val=self.args.optim.gradient_clip_val,
                         accumulate_grad_batches=self.args.training.accum,
                         limit_val_batches=self.args.mc_estimates,)
        return self.trainer
    
    def train_mutinfo(self, x: np.ndarray, y: np.ndarray):
        setattr(self.args, "seq_length", x.shape[1] + y.shape[1])
        setattr(self.args, "x_indices", list(range(x.shape[1])))
        setattr(self.args, "y_indices", list(range(x.shape[1], x.shape[1] + y.shape[1])))
        if self.args["alphabet_size"] is None:
            setattr(self.args, "alphabet_size", max(np.max(x), np.max(y)) + 1)

        data_set = array_to_dataset(x, y)
        train_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=True)
        valid_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=False)
        valid_loader = get_infinite_loader(valid_loader)

        trainer = self._trainer_setup()
        infosedd = self._load_infosedd()

        trainer.fit(model=infosedd, train_dataloaders=train_loader, valid_dataloaders=valid_loader)
        self.mutinfo_estimate = infosedd.mutinfo_estimate

        return infosedd.mutinfo_estimate

    def train_entropy(self, x: np.ndarray):
        
        data_set = array_to_dataset(x)
        setattr(self.args, "seq_length", data_set[0].shape[0])
        train_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=True)
        valid_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=False)
        valid_loader = get_infinite_loader(valid_loader)
        trainer = self._trainer_setup()
        infosedd = self._load_infosedd()
        trainer.fit(model=infosedd, train_dataloaders=train_loader, valid_dataloaders=valid_loader)

        self.entropy_estimate = infosedd.entropy_estimate
        return self.entropy_estimate

    def search_best_subset(self, x: np.ndarray, y: np.ndarray):
        pass