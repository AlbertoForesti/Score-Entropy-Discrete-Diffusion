import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

try:
    from infosedd.model import utils as mutils
    from infosedd.utils import statistics_batch
except:
    from model import utils as mutils
    from utils import statistics_batch

torch.autograd.set_detect_anomaly(True)

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)

    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False, mutinfo_config=None, marginal_score_fn = None, joint_score_fn = None, p_marginal=0.5):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
        
        # t = compute_density_for_timestep_sampling("logit_normal", batch.shape[0], logit_mean=0.5, logit_std=0.1, mode_scale=0.0).to(batch.device)
        # print("1 - Batch example: ", batch[0])
        # t = 1e-3*torch.ones_like(t)
        sigma, dsigma = noise(t)
        # raise UserWarning(f"t is {t}, sigma is {sigma}, batch shape is {batch.shape}")
        
        marginal_step = False

        if mutinfo_config is not None:
            marginal_step = np.random.rand() < p_marginal
            if marginal_step:
                var_y_indices = list(mutinfo_config['y_indices'])
                device = batch.device
                batch = batch.cpu().numpy()
                random_batch_permutation = np.random.permutation(batch.shape[0])

                shuffled_values = batch[random_batch_permutation[:, None], var_y_indices]
            
                # Update the batch with the shuffled values
                batch[np.arange(batch.shape[0])[:, None], var_y_indices] = shuffled_values
                batch = torch.tensor(batch, device=device)
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])
            if np.random.rand() < 1e-3:
                did_not_change = perturbed_batch == batch
                did_not_change = did_not_change.cpu().numpy()
                did_not_change = np.bitwise_and.reduce(did_not_change, axis=-1)
                print(f"Did not change: {did_not_change.sum() / did_not_change.shape[0]}")
        
        # print("2 - Batch example: ", batch[0])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False, is_marginal=marginal_step)
        log_score = log_score_fn(perturbed_batch, sigma)
        
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


def get_step_fn(noise, graph, train, optimize_fn, accum, mutinfo_config=None, marginal_score_fn=None, joint_score_fn=None, debug=False):
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
            assert not torch.isnan(loss).any(), f"Loss is NaN: {loss}"

            if joint_score_fn is not None and marginal_score_fn is not None and debug:
                if np.random.rand() < 1e-3:
                    log_score_joint_fn = lambda x, s: joint_score_fn(x, s).log()
                    min_loss_joint = loss_fn(log_score_joint_fn, batch, cond=cond).mean()
                    log_score_marginal_fn = lambda x, s: marginal_score_fn(x, s).log()
                    min_loss_marginal = loss_fn(log_score_marginal_fn, batch, cond=cond).mean()
                    print(f"Analytic loss joint: {min_loss_joint}, Analytic loss marginal: {min_loss_marginal}, Estimated loss: {loss}")
                    # print(f"Analytic derivative loss joint: {dloss_joint}, Analytic derivative loss marginal: {dloss_marginal}, Estimated derivative loss: {dloss}")

            
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