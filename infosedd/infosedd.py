import pytorch_lightning as pl
import torch
import numpy as np
import math
import os
from datetime import datetime
from hydra.utils import instantiate, call

from omegaconf import OmegaConf

from infosedd import sampling
from infosedd import graph_lib
from infosedd import noise_lib
from infosedd import losses

from infosedd.model import SEDD
from infosedd.model.ema import ExponentialMovingAverage
from infosedd.model.mlp import DiffusionMLP
from infosedd.model.unetmlp import UnetMLP_simple
from infosedd.model.two_sedds import DoubleSEDD

from infosedd.utils import array_to_dataset, get_infinite_loader, get_proj_fn
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AutoModelForMaskedLM

from torch.utils.data import DataLoader

class InfoSEDD(pl.LightningModule):

    def __init__(self, args):
        super(InfoSEDD, self).__init__()
        self.save_hyperparameters("args")
        self.args = args

        if not hasattr(self.args, "feature_selection"):
            setattr(self.args, "feature_selection", False)

        CHECKPOINT_DIR = self.args.training.checkpoint_dir
        # Add date and time to checkpoint directory
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, timestamp)

        logger=pl.loggers.TensorBoardLogger(save_dir=CHECKPOINT_DIR)
        
        self.trainer = pl.Trainer(logger=logger,
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
    
    def configure_optimizers(self):
        self.ema.set_device(self.score_model.device)
        if hasattr(self.args, "feature_selection") and self.args.feature_selection:
            params = [self.noise.noise]
        else:
            params = self.score_model.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.args.optim.lr, betas=(self.args.optim.beta1, self.args.optim.beta2), eps=self.args.optim.eps,
                               weight_decay=self.args.optim.weight_decay)
        # Total number of training steps
        total_steps = self.args.training.max_steps
        
        # Number of warmup steps
        warmup_steps = self.args.optim.warmup
        
        # Create the learning rate scheduler
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles=10)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def __call__(self, *args):
        if len(args) == 2:
            return self.call_mutinfo(*args)
        elif len(args) == 1:
            return self.call_entropy(*args)
        else:
            raise ValueError("Expected 1 or 2 arguments")
    
    def call_entropy(self, x: np.ndarray):

        self.entropy_estimate = None
        self.mutinfo_estimate = None
        self.oinfo_estimate = None

        data_set = array_to_dataset(x)

        self.args["seq_length"] = data_set[0].shape[0]
        self._build_score_model()
        self._setup_proj_fn()
        self._setup_graph_and_noise()
        self._setup_info_metric_fns()
        self._setup_loss_fns()

        self.train_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=True)

        self.fit(self.train_loader)

        ret_dict = {}

        if self.entropy_estimate is not None:
            ret_dict["entropy"] = self.entropy_estimate
        
        if self.oinfo_estimate is not None:    
            ret_dict["oinfo"] = self.oinfo_estimate

        return ret_dict

    def call_mutinfo(self, x: np.ndarray, y: np.ndarray, config_override=None):
        self.args["seq_length"] = x.shape[1] + y.shape[1]
        if self.args["alphabet_size"] is None:
            self.args["alphabet_size"] = max(np.max(x), np.max(y)) + 1
        self.mutinfo_estimate = None
        self.entropy_estimate = None
        self.oinfo_estimate = None

        if config_override is not None:
            self.args.update(config_override)

        setattr(self.args, "x_indices", list(range(x.shape[1])))
        setattr(self.args, "y_indices", list(range(x.shape[1], x.shape[1] + y.shape[1])))

        data_set = array_to_dataset(x, y)
        self.train_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=True)
        self.valid_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=False)
        self.valid_loader = get_infinite_loader(self.valid_loader)
        self._build_score_model()
        self._setup_proj_fn()
        self._setup_graph_and_noise()
        self._setup_info_metric_fns()
        self._setup_loss_fns()

        self.fit(self.train_loader, self.valid_loader)

        ret_dict = {}

        if self.mutinfo_estimate is not None:
            ret_dict["mi"] = self.mutinfo_estimate
        if self.entropy_estimate is not None:
            ret_dict["entropy"] = self.entropy_estimate
        if self.oinfo_estimate is not None:
            ret_dict["oinfo"] = self.oinfo_estimate

        return ret_dict
    
    def freeze_score_model(self):
        """Freeze all parameters of the score model to prevent them from being updated during training."""
        for param in self.score_model.parameters():
            param.requires_grad = False
        print("Score model weights frozen.")
    
    def feature_selection(self, x: np.ndarray, y: np.ndarray, config_override=None):
        self.args["seq_length"] = x.shape[1] + y.shape[1]
        self.freeze_score_model()
        if self.args["alphabet_size"] is None:
            self.args["alphabet_size"] = max(np.max(x), np.max(y)) + 1
        self.noise = noise_lib.LearnableNoise(self.noise, self.args.seq_length)
        self.mutinfo_estimate = None
        self.entropy_estimate = None
        self.oinfo_estimate = None

        if config_override is not None:
            self.args.update(config_override)

        setattr(self.args, "x_indices", list(range(x.shape[1])))
        setattr(self.args, "y_indices", list(range(x.shape[1], x.shape[1] + y.shape[1])))

        data_set = array_to_dataset(x, y)
        self.train_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=True)
        self.valid_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=False)
        self.valid_loader = get_infinite_loader(self.valid_loader)

        self._setup_proj_fn()
        self._setup_info_metric_fns()
        self._setup_loss_fns()

        CHECKPOINT_DIR = "feature_selection_"+self.args.training.checkpoint_dir
        # Add date and time to checkpoint directory
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, timestamp)

        logger=pl.loggers.TensorBoardLogger(save_dir=CHECKPOINT_DIR)

        self.trainer = pl.Trainer(logger=logger,
            default_root_dir=CHECKPOINT_DIR,
            accelerator=self.args.training.accelerator,
            devices=self.args.training.devices,
            max_steps=self.args.training.max_steps,
            max_epochs=None,
            check_val_every_n_epoch=10,
            gradient_clip_val=self.args.optim.gradient_clip_val,
            accumulate_grad_batches=self.args.training.accum,
            limit_val_batches=self.args.mc_estimates,)
        
        self.fit(self.train_loader, self.valid_loader)

        ret_dict = {}

        if self.mutinfo_estimate is not None:
            ret_dict["mi"] = self.mutinfo_estimate
        if self.entropy_estimate is not None:
            ret_dict["entropy"] = self.entropy_estimate
        if self.oinfo_estimate is not None:
            ret_dict["oinfo"] = self.oinfo_estimate
        
        ret_dict["noise"] = self.noise.noise.detach().cpu().numpy()

        return ret_dict
    
    def fit(self, train_loader, test_loader=None):
    # Check if trainer exists or is invalid
        if not hasattr(self, 'trainer') or self.trainer is None:
            # Recreate the trainer with the same configuration as in __init__
            CHECKPOINT_DIR = self.args.training.checkpoint_dir
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, timestamp)
            
            # Create logger more safely to prevent weak reference errors
            try:
                # Try to create logger directly
                logger = pl.loggers.TensorBoardLogger(save_dir=CHECKPOINT_DIR)
            except Exception as e:
                print(f"Warning: Could not create TensorBoardLogger: {e}")
                # Fallback to using CSV logger which is more reliable
                logger = pl.loggers.CSVLogger(save_dir=CHECKPOINT_DIR)
            
            self.trainer = pl.Trainer(
                logger=logger,
                default_root_dir=CHECKPOINT_DIR,
                accelerator=self.args.training.accelerator,
                devices=self.args.training.devices,
                max_steps=self.args.training.max_steps,
                max_epochs=None,
                check_val_every_n_epoch=None,
                val_check_interval=self.args.training.val_check_interval,
                gradient_clip_val=self.args.optim.gradient_clip_val,
                accumulate_grad_batches=self.args.training.accum,
                limit_val_batches=self.args.mc_estimates,
            )

        if test_loader is None:
            test_loader = DataLoader(
                train_loader.dataset, 
                batch_size=train_loader.batch_size, 
                shuffle=False, 
                num_workers=getattr(train_loader, 'num_workers', 0)  # Safely get num_workers
            )
        
        if self.args.resume_training and hasattr(self.args, 'checkpoint_path') and self.args.checkpoint_path:
            self.trainer.fit(model=self, train_dataloaders=train_loader,
                val_dataloaders=test_loader, ckpt_path=self.args.checkpoint_path)
        else:
            self.trainer.fit(model=self, train_dataloaders=train_loader,
                val_dataloaders=test_loader)
    
    def predict(self,predict_loader):
        self.trainer.predict(model=self, dataloaders=predict_loader)
    
    @property
    def proj_fn(self):
        return self._proj_fn
    
    @proj_fn.setter
    def proj_fn(self, proj_fn):
        self._proj_fn = proj_fn
        self._setup_info_metric_fns()
    
    def _setup_proj_fn(self):
        
        def default_proj_fn(x, is_score=True):
            return x

        if hasattr(self.args, "get_proj_fn"):
            self._proj_fn = call(self.args.get_proj_fn)
        elif hasattr(self.args, "proj_fn"):
            self._proj_fn = self.args.proj_fn
        else:
            self._proj_fn = default_proj_fn
    
    def _build_score_model(self):
        # build score model
        if self.args.model.name == "mlp":
            score_model = DiffusionMLP(self.args)
        elif self.args.model.name == "unetmlp":
            score_model = UnetMLP_simple(self.args)
        elif "double" in self.args.model.name:
            score_model = DoubleSEDD(self.args)
        elif "sedd" in self.args.model.name:
            score_model = SEDD(self.args)
        else:
            score_model = AutoModelForMaskedLM.from_pretrained(self.args.model.name, trust_remote_code=True)
        self.score_model = score_model
        self.ema = ExponentialMovingAverage(
        score_model.parameters(), decay=self.args.training.ema)

    def _setup_graph_and_noise(self):
        self.graph = graph_lib.get_graph(self.args)
        self.noise = noise_lib.get_noise(self.args)
    
    def _setup_info_metric_fns(self):

        if self.args.estimate_mutinfo:
            self.mutinfo_step_fn = sampling.get_mutinfo_step_fn(self.args, self.graph, self.noise, self.proj_fn)
            self.ema_mutinfo = ExponentialMovingAverage(
                torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True), decay=self.args.training.ema)
        
        if self.args.estimate_entropy:
            self.entropy_step_fn = sampling.get_entropy_step_fn(self.args, self.graph, self.noise, self.proj_fn)
            self.ema_entropy = ExponentialMovingAverage(
                torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True), decay=self.args.training.ema)
        
        if self.args.estimate_oinfo:
            self.oinfo_step_fn = sampling.get_oinfo_step_fn(self.args, self.graph, self.noise, self.proj_fn)
            self.ema_oinfo = ExponentialMovingAverage(
                torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True), decay=self.args.training.ema)
    
    def _setup_loss_fns(self):
        self.loss_fn_train = losses.get_loss_fn(self.args, self.noise, self.graph, True)
        self.loss_fn_test = losses.get_loss_fn(self.args, self.noise, self.graph, False)
    
    def setup(self, stage=None):

        # Initialize weights with xavier uniform
        if stage == "fit" and not self.args.feature_selection:
            for p in self.score_model.parameters():
                if p.dim() > 1:
                    torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        elif self.args.feature_selection:
            self.score_model.eval()

        
        sampling_eps = self.args.sampling.eps
        try:
            self.seq_length = self.args.seq_length
        except:
            self.seq_length = self.args.model.seq_length

        self.sampling_shape = (self.args.training.batch_size // (self.args.ngpus * self.args.training.accum), self.seq_length)
        self.sampling_fn = sampling.get_sampling_fn(self.args, self.graph, self.noise, self.sampling_shape, sampling_eps, "cuda")
        print("Setup done!")
    
    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.loss_fn_train(self.score_model, batch).mean()

        ret_dict = {}
        ret_dict["loss"] = loss

        if self.args.estimate_mutinfo:
            mutinfo_estimate = self.mutinfo_step_fn(self.score_model, batch)
            mutinfo_estimate = torch.tensor(mutinfo_estimate)
            self.ema_mutinfo.update([torch.nn.Parameter(mutinfo_estimate, requires_grad=True)])
            ema_estimate = self.ema_mutinfo.shadow_params[0].item()
            ret_dict["ema_mutinfo"] = ema_estimate
            self.log("ema_mutinfo", ema_estimate, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.logger.experiment.add_scalar("ema_mutinfo", ema_estimate, self.global_step)
        
        if self.args.estimate_oinfo:
            oinfo_estimate = self.oinfo_step_fn(self.score_model, batch)
            oinfo_estimate = torch.tensor(oinfo_estimate)
            self.ema_oinfo.update([torch.nn.Parameter(oinfo_estimate, requires_grad=True)])
            ema_estimate = self.ema_oinfo.shadow_params[0].item()
            ret_dict["ema_oinfo"] = ema_estimate
            self.log("ema_oinfo", ema_estimate, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.logger.experiment.add_scalar("ema_oinfo", ema_estimate, self.global_step)

        if self.args.estimate_entropy:
            entropy_estimate = self.entropy_step_fn(self.score_model, batch)
            entropy_estimate = torch.tensor(entropy_estimate)
            self.ema_entropy.update([torch.nn.Parameter(entropy_estimate, requires_grad=True)])
            ema_estimate = self.ema_entropy.shadow_params[0].item()
            ret_dict["ema_entropy"] = ema_estimate
            self.log("ema_entropy", ema_estimate, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.logger.experiment.add_scalar("ema_entropy", ema_estimate, self.global_step)

        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return ret_dict
    
    def on_after_backward(self):
        # This hook is called after the backward pass
        self.ema.update(self.score_model.parameters())
    
    def on_save_checkpoint(self, checkpoint):
        # Save EMA state
        checkpoint['ema_state'] = self.ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        # Load EMA state
        self._build_score_model()
        self._setup_graph_and_noise()

        if 'ema_state' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state'])

    def validation_step(self, batch, batch_idx):
        self.eval()
        ret_dict = {}
        with torch.no_grad():
            loss = self.loss_fn_test(self.score_model, batch).mean()
            ret_dict["loss"] = loss
            if self.args.estimate_mutinfo:
                mutinfo = self.mutinfo_step_fn(self.score_model, batch)
                ret_dict["mutinfo"] = mutinfo
                self.mutinfo_step_estimates.append(mutinfo)
            if self.args.estimate_oinfo:
                oinfo = self.oinfo_step_fn(self.score_model, batch)
                ret_dict["oinfo"] = oinfo
            if self.args.estimate_entropy:
                entropy = self.entropy_step_fn(self.score_model, batch)
                ret_dict["entropy"] = entropy
        return ret_dict
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        ret_dict = {}
        with torch.no_grad():
            if self.args.estimate_mutinfo:
                mutinfo = self.mutinfo_step_fn(self.score_model, batch)
                ret_dict["mutinfo"] = mutinfo
                self.mutinfo_step_estimates.append(mutinfo)
            if self.args.estimate_oinfo:
                oinfo = self.oinfo_step_fn(self.score_model, batch)
                ret_dict["oinfo"] = oinfo
            if self.args.estimate_entropy:
                entropy = self.entropy_step_fn(self.score_model, batch)
                ret_dict["entropy"] = entropy
        return ret_dict
    
    def on_predict_epoch_start(self):
        self.ema.store(self.score_model.parameters())
        self.ema.copy_to(self.score_model.parameters())
        if self.args.estimate_mutinfo:
            self.mutinfo_step_estimates = []
        if self.args.estimate_oinfo:
            self.oinfo_step_estimates = []
        if self.args.estimate_entropy:
            self.entropy_step_estimates = []
    
    def on_predict_epoch_end(self, *args, **kwargs):
        self.ema.restore(self.score_model.parameters())

        if self.args.estimate_mutinfo:
            self.mutinfo_estimate = np.mean(self.mutinfo_step_estimates)
        if self.args.estimate_oinfo:
            self.oinfo_estimate = np.mean(self.oinfo_step_estimates)
        if self.args.estimate_entropy:
            self.entropy_estimate = np.mean(self.entropy_step_estimates)
    
    def sample(self, num_samples):
        self.to("cuda")
        self.eval()
        self.setup(stage="predict")
        self.sampling_shape = (num_samples, self.seq_length)
        self.sampling_fn = sampling.get_sampling_fn(self.args, self.graph, self.noise, self.sampling_shape, self.args.sampling_eps, "cuda")
        with torch.no_grad():
            samples = self.sampling_fn(self.score_model)
        return samples
    
    def on_validation_epoch_start(self):
        self.ema.store(self.score_model.parameters())
        self.ema.copy_to(self.score_model.parameters())
        if self.args.estimate_mutinfo:
            self.mutinfo_step_estimates = []
        if self.args.estimate_oinfo:
            self.oinfo_step_estimates = []
        if self.args.estimate_entropy:
            self.entropy_step_estimates = []
        

    def on_validation_epoch_end(self):
        self.ema.restore(self.score_model.parameters())

        if self.args.estimate_mutinfo:
            self.mutinfo_estimate = np.mean(self.mutinfo_step_estimates)
            self.logger.experiment.add_scalar("val_mutinfo", self.mutinfo_estimate, self.global_step)
            self.log("val_mutinfo", self.mutinfo_estimate, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            print(f"Mutual information estimate: {self.mutinfo_estimate}")
        if self.args.estimate_oinfo:
            self.oinfo_estimate = np.mean(self.oinfo_step_estimates)
            self.logger.experiment.add_scalar("val_oinfo", self.oinfo_estimate, self.global_step)
            self.log("val_oinfo", self.oinfo_estimate, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.args.estimate_entropy:
            self.entropy_estimate = np.mean(self.entropy_step_estimates)
            self.logger.experiment.add_scalar("val_entropy", self.entropy_estimate, self.global_step)
            self.log("val_entropy", self.entropy_estimate, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def marginalize(self, indices):
        assert isinstance(self.graph, graph_lib.Absorbing), "Graph must be an absorbing graph"
        self.proj_fn = get_proj_fn(indices, self.graph.dim-1)
    
    def condition(self, indices, values):
        self.proj_fn = get_proj_fn(indices, values)
