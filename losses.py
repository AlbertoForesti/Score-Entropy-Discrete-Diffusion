import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False, mutinfo_config=None, marginal_score_fn = None, joint_score_fn = None):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
        
        # print("1 - Batch example: ", batch[0])
        # t = 1e-3*torch.ones_like(t)
        sigma, dsigma = noise(t)
        # raise UserWarning(f"t is {t}, sigma is {sigma}, batch shape is {batch.shape}")
        
        marginal_step = False

        if mutinfo_config is not None:
            marginal_step = np.random.rand() < 0.5
            if marginal_step:
                var_y_indices = list(mutinfo_config['y_indices'])
                random_batch_permutation = np.random.permutation(batch.shape[0])
                batch[np.arange(batch.shape[0]), var_y_indices] = batch[random_batch_permutation, var_y_indices]

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])
        
        # print("2 - Batch example: ", batch[0])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False, is_marginal=marginal_step)
        log_score = log_score_fn(perturbed_batch, sigma)

        if np.random.rand() < 1e-2:
            if marginal_step:
                if marginal_score_fn is None:
                    score_analytic = None
                else:
                    score_analytic = marginal_score_fn(perturbed_batch, sigma)
            else:
                if joint_score_fn is None:
                    score_analytic = None    
                else:
                    score_analytic = joint_score_fn(perturbed_batch, sigma)
            print(f"Score example - Marginal-{marginal_step}:\n x: {perturbed_batch[0]}\n x0: {batch[0]}\n Estimated score: {log_score[0].exp()}\n True score: {score_analytic[0]}\n t: {t[0]}")
            print(f"Mean Absolute Error: {torch.abs(log_score.exp() - score_analytic).mean()}")
            print(f"Mean Absolute Error with marginal: {torch.abs(log_score.exp() - marginal_score_fn(perturbed_batch, sigma)).mean()}")
        
        
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        # print("3 - Batch example: ", batch[0])

        # print(f"Loss shape before dsigma stuff {loss.shape}")

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        # print(f"Loss shape after dsigma stuff {loss.shape}")

        # print("*****************************")

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum, mutinfo_config=None, marginal_score_fn=None, joint_score_fn=None):
    loss_fn = get_loss_fn(noise, graph, train, mutinfo_config=mutinfo_config, marginal_score_fn=marginal_score_fn, joint_score_fn=joint_score_fn)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']
                
        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']

            # print(f"Batch example: {batch[0]}")

            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn