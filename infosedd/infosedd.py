import pytorch_lightning as pl
import torch
import numpy as np
import math
import os
from datetime import datetime
from hydra.utils import instantiate, call
import sampling
import noise_lib
import losses
from model.ema import ExponentialMovingAverage
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from torch.utils.data import DataLoader

class InfoSEDD(pl.LightningModule):

    def __init__(self, args):
        super(InfoSEDD, self).__init__()
        self.args = args
        self.save_hyperparameters("args")
    
    def configure_optimizers(self):
        self.ema.set_device(self.score_model.device)
        optimizer = instantiate(self.args.optim.optimizer, self.score_model.parameters())
        scheduler = instantiate(self.args.optim.scheduler, optimizer)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def fit(self,train_loader,test_loader=None):

        if test_loader is None:
            test_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=train_loader.num_workers)
        
        if self.args.resume_training:
            self.trainer.fit(model=self, train_dataloaders=train_loader,
                val_dataloaders=test_loader)
    
    def predict(self,predict_loader):
        self.trainer.predict(model=self, dataloaders=predict_loader)
    
    def setup(self, stage=None):
        self.score_model = instantiate(self.args.model, self.args)

        if hasattr(self.args, 'get_proj_fn'):
            assert self.args.proj_fn is not None, "get_proj_fn cannot be None"
            proj_fn = call(self.args.get_proj_fn)
        else:
            proj_fn = lambda x: x

        # Initialize weights with xavier uniform
        for p in self.score_model.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))


        self.ema = ExponentialMovingAverage(
        self.score_model.parameters(), decay=self.args.training.ema)

        self.noise = noise_lib.get_noise(self.args)
        sampling_eps = self.args.sampling.eps
        
        if self.args.estimate_mutinfo:
            self.mutinfo_step_fn = sampling.get_mutinfo_step_fn(self.args, self.graph, self.noise, proj_fn)
            self.ema_mutinfo = ExponentialMovingAverage(
                torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True), decay=self.args.training.ema)
        
        if self.args.estimate_entropy:
            self.entropy_step_fn = sampling.get_entropy_step_fn(self.args, self.graph, self.noise, proj_fn)
            self.ema_entropy = ExponentialMovingAverage(
                torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True), decay=self.args.training.ema)
        
        if self.args.estimate_oinfo:
            self.oinfo_step_fn = sampling.get_oinfo_step_fn(self.args, self.graph, self.noise, proj_fn)
            self.ema_oinfo = ExponentialMovingAverage(
                torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True), decay=self.args.training.ema)

        self.loss_fn_train = losses.get_loss_fn(self.args, self.noise, self.graph, True, sampling_eps)
        self.loss_fn_test = losses.get_loss_fn(self.args, self.noise, self.graph, False, sampling_eps)

        self.entropy_estimate = None
        self.mutinfo_estimate = None
        self.oinfo_estimate = None
    
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
    
    def on_prediction_epoch_start(self):
        self.ema.store(self.score_model.parameters())
        self.ema.copy_to(self.score_model.parameters())
        if self.args.estimate_mutinfo:
            self.mutinfo_step_estimates = []
        if self.args.estimate_oinfo:
            self.oinfo_step_estimates = []
        if self.args.estimate_entropy:
            self.entropy_step_estimates = []
    
    def on_prediction_epoch_end(self):
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
        self.setup()
        self.sampling_shape = (num_samples, self.args.model.seq_length)
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
        if self.args.estimate_oinfo:
            self.oinfo_estimate = np.mean(self.oinfo_step_estimates)
            self.logger.experiment.add_scalar("val_oinfo", self.oinfo_estimate, self.global_step)
            self.log("val_oinfo", self.oinfo_estimate, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.args.estimate_entropy:
            self.entropy_estimate = np.mean(self.entropy_step_estimates)
            self.logger.experiment.add_scalar("val_entropy", self.entropy_estimate, self.global_step)
            self.log("val_entropy", self.entropy_estimate, on_step=False, on_epoch=True, prog_bar=True, logger=True)
