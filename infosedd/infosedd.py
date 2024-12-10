import pytorch_lightning as pl
import torch
import numpy as np
from infosedd import sampling
from infosedd import graph_lib
from infosedd import noise_lib
from infosedd import data
from infosedd import losses

from infosedd.model import SEDD
from infosedd.model.ema import ExponentialMovingAverage
from infosedd.model.mlp import DiffusionMLP
from infosedd.model.unetmlp import UnetMLP_simple

from infosedd.utils import array_to_dataset

from torch.utils.data import DataLoader

class InfoSEDD(pl.LightningModule):

    def __init__(self, args):
        super(InfoSEDD, self).__init__()
        self.args = args
        if hasattr(self.args, 'gt'):
            self.gt = self.args.gt
        else:
            self.gt = None
        self.save_hyperparameters("args")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.score_model.parameters(), lr=self.args.optim.lr, betas=(self.args.optim.beta1, self.args.optim.beta2), eps=self.args.optim.eps,
                               weight_decay=self.args.optim.weight_decay)
        return optimizer
    
    def initialize_model(self):
        
        # build score model
        if self.args.model.name == "mlp":
            score_model = DiffusionMLP(self.args)
        elif self.args.model.name == "unetmlp":
            score_model = UnetMLP_simple(self.args)
        else:
            score_model = SEDD(self.args)
        self.score_model = score_model

        self.ema = ExponentialMovingAverage(
        score_model.parameters(), decay=self.args.training.ema)

        self.noise = noise_lib.get_noise(self.args)
        device = self.device
        sampling_eps = self.args.sampling.eps

        graph = graph_lib.get_graph(self.args, device)
        noise = noise_lib.get_noise(self.args)

        if self.args.cond is not None:
            input_ids = torch.tensor(self.args.cond.input_ids, device=device).long()
            input_locs = torch.tensor(self.args.cond.input_locs, device=device).long()
            indeces_to_discard = list(input_locs)
            indeces_to_keep = [i for i in range(self.args.model.length) if i not in indeces_to_discard]
        
            def proj_fun(x):
                x[:, input_locs] = input_ids
                return x
        else:
            indeces_to_keep = None
            proj_fun = lambda x: x
        
        if hasattr(self.args, 'data'):
            p = data.get_distribution(self.args.data)
        else:
            p = None

        if isinstance(p, np.ndarray):
            p = torch.tensor(p, device=device).float()
        
        if p is not None:
            p_joint = p.clone()
            px = torch.sum(p, axis=1)
            py = torch.sum(p, axis=0)
            pxy_margin = px * py.T
            pxy_margin = pxy_margin.unsqueeze(-1)
            self.marginal_score_fn = lambda x, sigma: graph.get_analytic_score(x, pxy_margin, sigma)
            self.joint_score_fn = lambda x, sigma: graph.get_analytic_score(x, p_joint, sigma)
        else:
            self.marginal_score_fn = None
            self.joint_score_fn = None

        self.sampling_shape = (self.args.training.batch_size // (self.args.ngpus * self.args.training.accum), self.args.model.length)
        self.sampling_fn = sampling.get_sampling_fn(self.args, graph, noise, self.sampling_shape, sampling_eps, device, p)
        self.entropy_estimate_fn = sampling.get_entropy_estimate_fn(self.args, graph, noise, self.sampling_shape, sampling_eps, device, p, proj_fun, indeces_to_keep)
        self.entropy_estimate_montecarlo_fn = sampling.get_entropy_montecarlo_estimate_fn(self.args, graph, noise, self.sampling_shape, sampling_eps, device, p)
        self.entropy_estimate_dynkin_fn = sampling.get_entropy_dynkin_estimate_fn(self.args, graph, noise, self.sampling_shape, sampling_eps, device, p, proj_fun, indeces_to_keep)
        self.mutinfo_estimate_fn = sampling.get_mutinfo_estimate_fn(self.args, graph, noise, self.sampling_shape, sampling_eps, device, p, proj_fun, indeces_to_keep)
        self.mutinfo_estimate_dynkin_fn = sampling.get_mutinfo_dynkin_estimate_fn(self.args, graph, noise, self.sampling_shape, sampling_eps, device, p, proj_fun, indeces_to_keep)

        self.noise = noise
        self.graph = graph

        if hasattr(self.args, 'mutinfo_config'):
            mutinfo_config = self.args.mutinfo_config
        else:
            mutinfo_config = None
        
        self.loss_fn_train = losses.get_loss_fn(self.noise, self.graph, True, sampling_eps, False, mutinfo_config, self.marginal_score_fn, self.joint_score_fn)
        self.loss_fn_test = losses.get_loss_fn(self.noise, self.graph, False, sampling_eps, False, mutinfo_config, self.marginal_score_fn, self.joint_score_fn)

    
    def __call__(self, x: np.ndarray, y: np.ndarray):
        self.args["seq_length"] = x.shape[1] + y.shape[1]

        self.initialize_model()

        data_set = array_to_dataset(x, y)
        train_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=True)

        self.fit(train_loader)

        self.ema.store(self.score_model.parameters())
        self.ema.copy_to(self.score_model.parameters())
        self.estimate_information_quantities(self.score_model, train_loader)
        self.ema.restore(self.score_model.parameters())

        ret_dict = {}

        if self.mutinfo_estimate is not None:
            ret_dict["mi"] = self.mutinfo_estimate
        if self.entropy_estimate is not None:
            ret_dict["entropy"] = self.entropy_estimate

        return ret_dict
    
    def estimate_information_quantities(self, score_model, dataloader):
        if self.args.estimate_entropy:
            if self.args.montecarlo:
                self.entropy_estimate = self.entropy_estimate_montecarlo_fn(score_model, dataloader)
            elif self.args.dynkin:
                self.entropy_estimate = self.entropy_estimate_dynkin_fn(score_model, dataloader)
            else:
                self.entropy_estimate = self.entropy_estimate_fn(score_model)
        if self.args.estimate_mutinfo:
            if self.args.dynkin:
                self.mutinfo_estimate = self.mutinfo_estimate_dynkin_fn(score_model, dataloader)
            else:
                self.mutinfo_estimate = self.mutinfo_estimate_fn(score_model)
    
    def fit(self,train_loader,test_loader=None):

        if test_loader is None:
            test_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=train_loader.num_workers)
        
        args = self.args
        CHECKPOINT_DIR = args.training.checkpoint_dir

        logger=pl.loggers.TensorBoardLogger(save_dir=CHECKPOINT_DIR)
        
        trainer = pl.Trainer(logger=logger,
                         default_root_dir=CHECKPOINT_DIR,
                         accelerator=self.args.training.accelerator,
                         devices=self.args.training.devices,
                         max_steps=self.args.training.max_steps,
                         max_epochs=None,
                         check_val_every_n_epoch=None,
                         val_check_interval=self.args.training.val_check_interval)  
        
        trainer.fit(model=self, train_dataloaders=train_loader,
                val_dataloaders=test_loader)
        
    
    def on_fit_start(self):
        self.entropy_estimate = None
        self.mutinfo_estimate = None
        self.score_model.to(self.device)
        self.noise.to(self.device)
    
    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.loss_fn_train(self.score_model, batch, cond=None).mean()
        if self.joint_score_fn is not None and self.marginal_score_fn is not None and self.debug:
            if np.random.rand() < 1e-3:
                log_score_joint_fn = lambda x, s: self.joint_score_fn(x, s).log()
                min_loss_joint = self.loss_fn_train(log_score_joint_fn, batch, cond=None).mean()
                log_score_marginal_fn = lambda x, s: self.marginal_score_fn(x, s).log()
                min_loss_marginal = self.loss_fn_train(log_score_marginal_fn, batch, cond=None).mean()
                print(f"Analytic loss joint: {min_loss_joint}, Analytic loss marginal: {min_loss_marginal}, Estimated loss: {loss}")
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return {"loss": loss}
    
    def on_after_backward(self):
        # This hook is called after the backward pass
        self.ema.set_device(self.device)
        self.ema.update(self.score_model.parameters())

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            self.ema.store(self.score_model.parameters())
            self.ema.copy_to(self.score_model.parameters())
            loss = self.loss_fn_test(self.score_model, batch, cond=None).mean()
            self.ema.restore(self.score_model.parameters())
            return {"loss": loss}