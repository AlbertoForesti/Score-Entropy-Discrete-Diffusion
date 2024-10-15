import abc
import torch
import torch.nn.functional as F
import numpy as np
from catsample import sample_categorical
from functools import partial

from model import utils as mutils
from tqdm import tqdm

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass

    def get_ratio_with_uniform(self, score_fn, x, t, step_size, vocab_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)
        uniform_score = torch.ones_like(score)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        new_x = sample_categorical(probs)

        stag_score_uniform = self.graph.staggered_score(uniform_score, dsigma)
        probs_uniform = stag_score_uniform * self.graph.transp_transition(x, dsigma)
        # print("probs ", probs[:5]) # all like [0.01, 0.99]
        # print("numerator ", torch.gather(probs, -1, new_x[...,None])[:5]) # Ends up like [0.99] most of times
        num = torch.gather(probs, -1, new_x[...,None])
        den = torch.gather(probs_uniform, -1, new_x[...,None])

        ratio = num/den # Shape (bs,1,1)
        return ratio


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn

def get_entropy_estimate_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler_for_entropy_estimate(graph=graph,
                                                      noise=noise,
                                                      batch_dims=batch_dims,
                                                      predictor=config.sampling.predictor,
                                                      steps=config.sampling.steps,
                                                      denoise=config.sampling.noise_removal,
                                                      vocab_size=config.tokens,
                                                      eps=eps,
                                                      device=device)
    
    def estimate_entropy_fn(model):
        estimates = []
        for i in tqdm(range(config.paths)):
            kl = sampling_fn(model)
            estimates.append(kl)
        # print("Estimates: ", estimates)
        print("Mean estimate: ", np.log(config.tokens)-np.mean(estimates), "Std estimate: ", np.std(estimates))
        return np.log(config.tokens)-np.mean(estimates)
    
    return estimate_entropy_fn

def get_mutinfo_estimate_fn(config, graph, noise, batch_dims, eps, x_indices, y_indices, device):
    get_pc_sampler_fn = partial(get_pc_sampler_for_entropy_estimate, graph=graph, noise=noise, batch_dims=batch_dims, predictor=config.sampling.predictor, steps=config.sampling.steps, denoise=config.sampling.noise_removal, vocab_size=config.tokens, eps=eps, device=device)

    sampling_joint_fn = get_pc_sampler_fn()

    def get_proj_fun(input_locs, input_ids, random_proj=False):
        def proj_fun(x):
            if random_proj:
                random_ids = torch.randint(0, config.tokens, (len(input_locs),), device=device)
                x[:, input_locs] = random_ids
            else:
                x[:, input_locs] = input_ids
            return x
        return proj_fun

    def estimate_mutinfo_fn(model):
        estimates = []
        kl_cond_x_estimates = []
        kl_cond_y_estimates = []
        kl_joint_estimates = []
        for i in tqdm(range(config.paths)):
            kl_joint = sampling_joint_fn(model)
            
            x_cond_vector = torch.randint(0, config.tokens, (len(x_indices),), device=device)
            y_cond_vector = torch.randint(0, config.tokens, (len(y_indices),), device=device)

            proj_x_fn = get_proj_fun(x_indices, x_cond_vector, True)
            proj_y_fn = get_proj_fun(y_indices, y_cond_vector, True)

            kl_cond_x = get_pc_sampler_fn(proj_fun=proj_x_fn)(model)

            kl_cond_y = get_pc_sampler_fn(proj_fun=proj_y_fn)(model)

            kl_cond_x_estimates.append(kl_cond_x)
            kl_cond_y_estimates.append(kl_cond_y)
            kl_joint_estimates.append(kl_joint)

            estimates.append(kl_cond_x + kl_cond_y - kl_joint)
        
        print("Mean estimate: ", np.mean(estimates), "Std estimate: ", np.std(estimates))
        print("Mean estimate cond x: ", np.mean(kl_cond_x_estimates), "Std estimate cond x: ", np.std(kl_cond_x_estimates))
        print("Mean estimate cond y: ", np.mean(kl_cond_y_estimates), "Std estimate cond y: ", np.std(kl_cond_y_estimates))
        print("Mean estimate joint: ", np.mean(kl_joint_estimates), "Std estimate joint: ", np.std(kl_joint_estimates))
        return np.mean(estimates)
        

    return estimate_mutinfo_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)
            

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        return x
    
    return pc_sampler

def get_pc_sampler_for_entropy_estimate(graph, noise, batch_dims, predictor, steps, vocab_size, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps
        log_radon_sum = 0

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            ratio = predictor.get_ratio_with_uniform(sampling_score_fn, x, t, dt, vocab_size)
            # print("ratio is", ratio[:5]) are all around 2
            # print("log ratio is", torch.log(ratio)[:5])
            log_radon_sum = log_radon_sum + torch.log(ratio)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)
        
        # print("Log Radon sum: ", log_radon_sum)
        return torch.mean(log_radon_sum).item()
    
    return pc_sampler
