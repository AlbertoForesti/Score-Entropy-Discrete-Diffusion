import pytorch_lightning as pl
import torch
import numpy as np
import math

try:
    from infosedd import sampling
    from infosedd import graph_lib
    from infosedd import noise_lib
    from infosedd import data
    from infosedd import losses

    from infosedd.model import SEDD
    from infosedd.model.ema import ExponentialMovingAverage
    from infosedd.model.mlp import DiffusionMLP
    from infosedd.model.unetmlp import UnetMLP_simple
    from infosedd.model.two_sedds import DoubleSEDD

    from infosedd.utils import array_to_dataset
except:
    import sampling
    import graph_lib
    import noise_lib
    import data
    import losses

    from model import SEDD
    from model.ema import ExponentialMovingAverage
    from model.mlp import DiffusionMLP
    from model.unetmlp import UnetMLP_simple
    from model.two_sedds import DoubleSEDD

    from utils import array_to_dataset

from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

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
        self.mutinfo_config = None

        args = self.args
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
                         gradient_clip_val=self.args.optim.gradient_clip_val,)  
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.score_model.parameters(), lr=self.args.optim.lr, betas=(self.args.optim.beta1, self.args.optim.beta2), eps=self.args.optim.eps,
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

        data_set = array_to_dataset(x)

        self.args["seq_length"] = data_set[0].shape[0]

        self.train_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=True)

        self.fit(self.train_loader)

        ret_dict = {}

        if self.entropy_estimate is not None:
            ret_dict["entropy"] = self.entropy_estimate

        return ret_dict

    def call_mutinfo(self, x: np.ndarray, y: np.ndarray):
        self.args["seq_length"] = x.shape[1] + y.shape[1]
        if self.args["alphabet_size"] is None:
            self.args["alphabet_size"] = max(np.max(x), np.max(y)) + 1
        self.mutinfo_estimate = None
        self.entropy_estimate = None

        self.mutinfo_config = dict()
        self.mutinfo_config["x_indices"] = list(range(x.shape[1]))
        self.mutinfo_config["y_indices"] = list(range(x.shape[1], x.shape[1] + y.shape[1]))

        data_set = array_to_dataset(x, y)
        self.train_loader = DataLoader(data_set, batch_size=self.args.training.batch_size, shuffle=True)

        self.fit(self.train_loader)

        ret_dict = {}

        if self.mutinfo_estimate is not None:
            ret_dict["mi"] = self.mutinfo_estimate
        if self.entropy_estimate is not None:
            ret_dict["entropy"] = self.entropy_estimate

        return ret_dict
    
    def estimate_information_quantities(self, score_model, dataloader):
        if self.args.estimate_entropy:
            self.entropy_estimate = self.entropy_estimate_dynkin_fn(score_model, dataloader)            
        if self.args.estimate_mutinfo:
            self.mutinfo_estimate = self.mutinfo_estimate_dynkin_fn(score_model, dataloader)
    
    def fit(self,train_loader,test_loader=None):

        if test_loader is None:
            test_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=train_loader.num_workers)
        
        if self.args.resume_training:
            self.trainer.fit(model=self, train_dataloaders=train_loader,
                val_dataloaders=test_loader)
    
    def setup(self, stage=None):
        # build score model
        if self.args.model.name == "mlp":
            score_model = DiffusionMLP(self.args)
        elif self.args.model.name == "unetmlp":
            score_model = UnetMLP_simple(self.args)
        elif "double" in self.args.model.name:
            score_model = DoubleSEDD(self.args)
        else:
            score_model = SEDD(self.args)
        self.score_model = score_model

        if self.args.checkpoint_path is not None:
            try:
                self.load_from_checkpoint(self.args.checkpoint_path)
            except:
                self.score_model = SEDD.from_pretrained(self.args.checkpoint_path).to(self.device)

        # Initialize weights with xavier uniform
        for p in self.score_model.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))


        self.ema = ExponentialMovingAverage(
        score_model.parameters(), decay=self.args.training.ema)

        self.noise = noise_lib.get_noise(self.args)
        device = self.device
        sampling_eps = self.args.sampling.eps

        graph = graph_lib.get_graph(self.args, device)
        noise = noise_lib.get_noise(self.args)

        try:
            seq_length = self.args.seq_length
        except:
            seq_length = self.args.model.seq_length

        if self.args.cond is not None:
            input_ids = torch.tensor(self.args.cond.input_ids, device=device).long()
            input_locs = torch.tensor(self.args.cond.input_locs, device=device).long()
            indeces_to_discard = list(input_locs)
            indeces_to_keep = [i for i in range(seq_length) if i not in indeces_to_discard]
        
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

        self.sampling_shape = (self.args.training.batch_size // (self.args.ngpus * self.args.training.accum), seq_length)
        self.sampling_fn = sampling.get_sampling_fn(self.args, graph, noise, self.sampling_shape, sampling_eps, device, p)
        self.entropy_estimate_dynkin_fn = sampling.get_entropy_dynkin_estimate_fn(self.args, graph, noise, self.sampling_shape, sampling_eps, device, p, proj_fun, indeces_to_keep)
        
        if self.args.estimate_mutinfo:
            try:
                x_indices = self.mutinfo_config["x_indices"]
                y_indices = self.mutinfo_config["y_indices"]
            except:
                raise UserWarning(f"Mutual information config is {self.mutinfo_config}")
            self.mutinfo_estimate_dynkin_fn = sampling.get_mutinfo_dynkin_estimate_fn(self.args, graph, noise, self.sampling_shape, sampling_eps, device, x_indices, y_indices, p, proj_fun, indeces_to_keep)

        self.noise = noise
        self.graph = graph

        if self.args.estimate_entropy and not self.args.estimate_mutinfo:
            p_marginal = 0
            mutinfo_config = None
        else:
            p_marginal = self.args.training.p_marginal
            mutinfo_config = self.mutinfo_config

        self.loss_fn_train = losses.get_loss_fn(self.noise, self.graph, True, sampling_eps, False, mutinfo_config, self.marginal_score_fn, self.joint_score_fn, p_marginal)
        self.loss_fn_test = losses.get_loss_fn(self.noise, self.graph, False, sampling_eps, False, mutinfo_config, self.marginal_score_fn, self.joint_score_fn, p_marginal)

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
    
    def on_save_checkpoint(self, checkpoint):
        # Save EMA state
        checkpoint['ema_state'] = self.ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        # Load EMA state
        if 'ema_state' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state'])

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            self.ema.store(self.score_model.parameters())
            self.ema.copy_to(self.score_model.parameters())
            loss = self.loss_fn_test(self.score_model, batch, cond=None).mean()
            self.ema.restore(self.score_model.parameters())
            return {"loss": loss}
    
    def sample(self, num_samples):
        self.to("cuda")
        self.eval()
        self.setup()
        self.sampling_shape = (num_samples, self.args.model.seq_length)
        self.sampling_fn = sampling.get_sampling_fn(self.args, self.graph, self.noise, self.sampling_shape, self.args.sampling_eps, "cuda")
        with torch.no_grad():
            samples = self.sampling_fn(self.score_model)
        return samples
    
    def on_validation_epoch_end(self):
        self.ema.store(self.score_model.parameters())
        self.ema.copy_to(self.score_model.parameters())
        self.estimate_information_quantities(self.score_model, self.train_loader)
        if self.args.generate_samples:
            samples = self.sampling_fn(self.score_model)
            print(samples)
            if self.args.debug:
                result = samples[:,-2].cpu().numpy()
                to_be_xored = torch.cat((samples[:,0:-2], samples[:,-1:]), dim=1).cpu().numpy()
                other = np.bitwise_xor.reduce(to_be_xored, axis=1)
                correct_generations = np.sum(result == other)
                print(correct_generations)

        self.ema.restore(self.score_model.parameters())
        if self.entropy_estimate is not None:
            self.logger.experiment.add_scalar("val_entropy", self.entropy_estimate, self.global_step)
        if self.mutinfo_estimate is not None:
            self.logger.experiment.add_scalar("val_mutinfo", self.mutinfo_estimate, self.global_step)

    def on_train_end(self):
        self.ema.store(self.score_model.parameters())
        self.ema.copy_to(self.score_model.parameters())
        self.estimate_information_quantities(self.score_model, self.train_loader)
        self.ema.restore(self.score_model.parameters())
        if self.entropy_estimate is not None:
            self.logger.experiment.add_scalar("final_entropy", self.entropy_estimate, self.global_step)
        if self.mutinfo_estimate is not None:
            self.logger.experiment.add_scalar("final_mutinfo", self.mutinfo_estimate, self.global_step)
