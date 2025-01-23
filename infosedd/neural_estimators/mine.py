import pytorch_lightning as pl
import torch
from mutinfo.estimators.neural.mine import _MINE_backbone
from hydra.utils import instantiate, call
from torch.utils.data import DataLoader
from infosedd.utils import array_to_dataset
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

class MINE(pl.LightningModule):

    def __init__(self, args):
        super(MINE, self).__init__()
        self.args = args
        if hasattr(self.args, 'gt'):
            self.gt = self.args.gt
        else:
            self.gt = None
        self.save_hyperparameters("args")
        self.mutinfo_config = None

        CHECKPOINT_DIR = args.training.checkpoint_dir

        logger=pl.loggers.TensorBoardLogger(save_dir=CHECKPOINT_DIR)
        
        self.trainer = pl.Trainer(logger=logger,
                         default_root_dir=CHECKPOINT_DIR,
                         accelerator=self.args.training.accelerator,
                         devices=self.args.training.devices,
                         max_steps=self.args.training.max_steps,
                         max_epochs=None,
                         check_val_every_n_epoch=None,
                         val_check_interval=self.args.training.val_check_interval,
                         gradient_clip_val=self.args.training.gradient_clip_val,)  

    def __call__(self, *args):
        if len(args) == 2:
            return self.call_mutinfo(*args)
        elif len(args) == 1:
            return self.call_entropy(*args)
        else:
            raise ValueError("Expected 1 or 2 arguments")
    
    def call_entropy(self, x):
        raise NotImplementedError("Entropy estimation not implemented")
    
    def call_mutinfo(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        self.x_indices = list(range(x.shape[1]))
        self.y_indices = list(range(x.shape[1], x.shape[1] + y.shape[1]))
        data_set = array_to_dataset(x, y, return_separated_variables=True, dtype=torch.float32)
        self.train_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=True)
        self.estimate_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=False)
        self.trainer.fit(model=self, train_dataloaders=self.train_loader,
                val_dataloaders=self.estimate_loader)
        ret_dict = {}

        if self.mutinfo_estimate is not None:
            ret_dict["mi"] = self.mutinfo_estimate
        if self.entropy_estimate is not None:
            ret_dict["entropy"] = self.entropy_estimate

        return ret_dict

    def setup(self, stage=None):
        self.entropy_estimate = None
        self.mutinfo_estimate = None
        backbone_factory = call(self.args.get_backbone_factory)
        self.mine = backbone_factory(self.x_shape, self.y_shape)

        if not isinstance(self.mine, _MINE_backbone):
            raise TypeError(f"Expected MINE instance, got {type(self.mine)}")

        self._loss_fn = instantiate(self.args.loss_fn)
    
    def configure_optimizers(self):
        optimizer = instantiate(self.args.optimizer)(params=self.mine.parameters())
        
        # Number of warmup steps
        warmup_steps = self.args.training.warmup
        
        # Create the learning rate scheduler
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def loss_fn(self, mine, batch):
        x, y = batch
        loss = self._loss_fn(mine(x,y), mine(x,y,marginalize=self.args.marginalize))
        return loss

    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.loss_fn(self.mine, batch).mean()
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return {"loss": loss}
    
    def estimate_information_quantities(self):
        if self.args.estimate_entropy:
            raise NotImplementedError("Entropy estimation not implemented")
        if self.args.estimate_mutinfo:
            mi = self.mine.get_mutual_information(
                self.estimate_loader,
                self._loss_fn,
                self.device,
                marginalize=self.args.marginalize,)
            self.mutinfo_estimate = mi
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        loss = self.loss_fn(self.mine, batch).mean()
        self.log("val_loss", loss)
        return {"val_loss": loss}
    
    def on_validation_epoch_end(self):
        self.estimate_information_quantities()
        if self.entropy_estimate is not None:
            self.logger.experiment.add_scalar("val_entropy", self.entropy_estimate, self.global_step)
        if self.mutinfo_estimate is not None:
            self.logger.experiment.add_scalar("val_mutinfo", self.mutinfo_estimate, self.global_step)
    
    def on_train_end(self):
        self.estimate_information_quantities()
        if self.args.estimate_entropy:
            raise NotImplementedError("Entropy estimation not implemented")
        if self.args.estimate_mutinfo:
            self.logger.experiment.add_scalar("final_mutinfo", self.mutinfo_estimate, self.global_step)
