import abc
import torch
import torch.nn.functional as F
import numpy as np
from catsample import sample_categorical
from functools import partial
from scipy.linalg import expm

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
    
    def get_ratio_with_uniform(self, score_fn, x, t, step_size, proj_fn = lambda x: x, indeces_to_keep = None):
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x
    
    def get_ratio_with_uniform(self, score_fn, x, t, step_size, proj_fn = lambda x: x, indeces_to_keep = None):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        if indeces_to_keep is not None:
            step_size /= len(indeces_to_keep)
            # print("Step size is ", step_size)
        else:
            step_size /= x.shape[1]

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        new_x = self.graph.sample_rate(x, rev_rate)
        new_x = proj_fn(new_x)

        probs = F.one_hot(x, num_classes=self.graph.dim).to(rev_rate) + rev_rate

        # print(f"probs examples: {probs[:5]}")
        # print(f"probs conditional examples: {probs[:, indeces_to_keep][:5]}")

        # Score shape is (batch_size, seq_len, vocab_size)
        # x shape is (batch_size, seq_len)
        # condition must be sampled according to p_0, not p_T

        # print("score shape ", score.shape)
        uniform_score = torch.ones_like(score)
        # print("uniform score shape ", uniform_score.shape)
        rev_rate_uniform = step_size * dsigma[..., None] * self.graph.reverse_rate(x, uniform_score)
        probs_uniform = F.one_hot(x, num_classes=self.graph.dim).to(rev_rate_uniform) + rev_rate_uniform

        num = torch.gather(probs, -1, new_x[...,None])

        # indeces_to_keep = torch.randint(0, self.graph.dim, (1,))

        if indeces_to_keep is not None:
            num = num[:, indeces_to_keep]
            # print("Indices to keep: ", indeces_to_keep)
        # num_seq = torch.prod(num, dim=1, keepdim=True)
        den = torch.gather(probs_uniform, -1, new_x[...,None])
        if indeces_to_keep is not None:
            den = den[:, indeces_to_keep]
            # print("Indices to keep: ", indeces_to_keep)

        # den_seq = torch.prod(den, dim=1, keepdim=True)

        ratio = num/den # Shape (bs,1,1)
        # ratio = num
        # print("Ratio shape: ", ratio.shape)
        ratio = ratio.prod(dim=1, keepdim=True)
        # print("Ratio shape after sum: ", ratio.shape)
        return ratio

    def get_ratio_with_marginal(self, score_fn_joint, score_fn_marginal, x, t, step_size, proj_fn = lambda x: x, indeces_to_keep = None):
        sigma, dsigma = self.noise(t)
        score_joint = score_fn_joint(x, sigma)

        if indeces_to_keep is not None:
            step_size /= len(indeces_to_keep)
            # print("Step size is ", step_size)
        else:
            step_size /= x.shape[1]

        rev_rate_joint = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score_joint)
        new_x = self.graph.sample_rate(x, rev_rate_joint)
        new_x = proj_fn(new_x)

        probs_joint = F.one_hot(x, num_classes=self.graph.dim).to(rev_rate_joint) + rev_rate_joint

        # print(f"probs examples: {probs[:5]}")
        # print(f"probs conditional examples: {probs[:, indeces_to_keep][:5]}")

        # Score shape is (batch_size, seq_len, vocab_size)
        # x shape is (batch_size, seq_len)
        # condition must be sampled according to p_0, not p_T

        # print("score shape ", score.shape)
        score_marginal = score_fn_marginal(x, sigma)
        # print("score marginal examples: ", score_marginal[:5])
        # print("x examples: ", x[:5])
        # print("uniform score shape ", uniform_score.shape)
        rev_rate_marginal = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score_marginal)
        probs_marginal = F.one_hot(x, num_classes=self.graph.dim).to(rev_rate_marginal) + rev_rate_marginal

        num = torch.gather(probs_joint, -1, new_x[...,None])

        # indeces_to_keep = torch.randint(0, self.graph.dim, (1,))

        if indeces_to_keep is not None:
            num = num[:, indeces_to_keep]
            # print("Indices to keep: ", indeces_to_keep)
        # num_seq = torch.prod(num, dim=1, keepdim=True)
        den = torch.gather(probs_marginal, -1, new_x[...,None])
        if indeces_to_keep is not None:
            den = den[:, indeces_to_keep]
            # print("Indices to keep: ", indeces_to_keep)

        # den_seq = torch.prod(den, dim=1, keepdim=True)

        ratio = num/den # Shape (bs,1,1)
        # ratio = num
        # print("Ratio shape: ", ratio.shape)
        ratio = ratio.prod(dim=1, keepdim=True)
        # print("Ratio shape after sum: ", ratio.shape)
        return ratio

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
    
    def get_ratio_with_uniform(self, score_fn, x, t, step_size, proj_fn = lambda x: x, indeces_to_keep = None):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        # Score shape is (batch_size, seq_len, vocab_size)
        # x shape is (batch_size, seq_len)
        # condition must be sampled according to p_0, not p_T

        # print("score shape ", score.shape)
        uniform_score = torch.ones_like(score)
        # print("uniform score shape ", uniform_score.shape)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        new_x = sample_categorical(probs)
        new_x = proj_fn(new_x)

        stag_score_uniform = self.graph.staggered_score(uniform_score, dsigma)
        probs_uniform = stag_score_uniform * self.graph.transp_transition(x, dsigma)
        num = torch.gather(probs, -1, new_x[...,None])
        if indeces_to_keep is not None:
            num = num[:, indeces_to_keep]
        num_seq = torch.prod(num, dim=1, keepdim=True)
        den = torch.gather(probs_uniform, -1, new_x[...,None])
        if indeces_to_keep is not None:
            den = den[:, indeces_to_keep]
        den_seq = torch.prod(den, dim=1, keepdim=True)

        ratio = num_seq/den_seq # Shape (bs,1,1)
        return ratio

    
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

def get_entropy_estimate_fn(config, graph, noise, batch_dims, eps, device, p=None, proj_fun = lambda x: x, indeces_to_keep=None):
    
    sampling_fn = get_pc_sampler_for_entropy_estimate(graph=graph,
                                                      noise=noise,
                                                      batch_dims=batch_dims,
                                                      predictor=config.sampling.predictor,
                                                      steps=config.sampling.steps,
                                                      denoise=config.sampling.noise_removal,
                                                      vocab_size=config.tokens,
                                                      eps=eps,
                                                      device=device,
                                                      proj_fun=proj_fun,
                                                      p=p,
                                                      indeces_to_keep=indeces_to_keep)
    
    def estimate_entropy_fn(model):
        estimates = []
        for i in tqdm(range(config.paths)):
            kl = sampling_fn(model)
            estimates.append(kl)
        # print("Estimates: ", estimates)
        factor = batch_dims[1] if indeces_to_keep is None else len(indeces_to_keep)
        print("Mean estimate: ", np.log(config.tokens*factor)-np.mean(estimates), "Mean kl: ",np.mean(estimates),"Std estimate: ", np.std(estimates))
        return np.log(config.tokens*factor)-np.mean(estimates)
    
    return estimate_entropy_fn

def get_mutinfo_dynkin_estimate_fn(config, graph, noise, batch_dims, eps, device, p=None, proj_fun = lambda x: x, indeces_to_keep=None):
    
    def estimate_mutinfo_fn(model, data_loader):
        print("batch dims: ", batch_dims)
        if p is None:
            raise NotImplementedError("Score network architecture not implemented yet")
        else:
            joint_score_fn = lambda x, s: graph.get_analytic_score(x, p, s)
            px = torch.sum(p, axis=1)
            py = torch.sum(p, axis=0)
            print("PX: ", px, " with shape ", px.shape)
            print("PY: ", py, " with shape ", py.shape)
            print("PXY: ", p, " with shape ", p.shape)
            pxy_margin = px * py.T
            pxy_margin = pxy_margin.unsqueeze(-1)
            print("PXY MARGINAL: ", pxy_margin, " with shape ", pxy_margin.shape)
            marginal_score_fn = lambda x, s: graph.get_analytic_score(x, pxy_margin, s)
        estimates = []
        with torch.no_grad():
            step = 0
            for batch in tqdm(data_loader, desc="Estimating MI", total=config.paths):
                if step==config.paths:
                    break
                if config.data.valid != "text8" and config.data.valid != "bernoulli":
                    batch = batch['input_ids'].to(device)
                else:
                    if config.data.valid == "bernoulli":
                        batch = batch["feature"].to(device)
                    else:
                        batch = batch.to(device)
                t = torch.rand(batch.shape[0], 1, device=device)
                sigma, dsigma = noise(t)
                # raise UserWarning(f"t is {t}, sigma is {sigma}, batch shape is {batch.shape}")
                perturbed_batch = graph.sample_transition(batch, sigma)
                try:
                    score_joint = joint_score_fn(perturbed_batch, sigma)
                except:
                    raise ValueError(f"Could not compute score for batch with shape {perturbed_batch.shape} and sigma shape {sigma.shape}")
                score_marginal = marginal_score_fn(perturbed_batch, sigma)
                divergence_estimate = graph.score_divergence(score_joint, score_marginal, dsigma, perturbed_batch).mean().item()
                estimates.append(divergence_estimate)
                step += 1
        print("Mean estimate: ", np.mean(estimates), "Mean kl: ",np.mean(estimates),"Std estimate: ", np.std(estimates), "Some stimates: ", estimates[:5])
        return np.mean(estimates)
    return estimate_mutinfo_fn

def get_entropy_dynkin_estimate_fn(config, graph, noise, batch_dims, eps, device, p=None, proj_fun = lambda x: x, indeces_to_keep=None):
    
    def estimate_entropy_fn(model, data_loader):
        if p is None:
            raise NotImplementedError("Score network architecture not implemented yet")
        else:
            score_fn = lambda x, s: graph.get_analytic_score(x, p, s)
        estimates = []
        with torch.no_grad():
            step = 0
            for batch in tqdm(data_loader, desc="Estimating Entropy", total=config.paths):
                if step==config.paths:
                    break
                if config.data.valid != "text8" and config.data.valid != "bernoulli" and config.data.valid != "binomial":
                    batch = batch['input_ids'].to(device)
                else:
                    if config.data.valid == "bernoulli" or config.data.valid == "binomial":
                        batch = batch["feature"].to(device)
                    else:
                        batch = batch.to(device)
                t = torch.rand(batch.shape[0], 1, device=device)
                sigma, dsigma = noise(t)
                # raise UserWarning(f"t is {t}, sigma is {sigma}, batch shape is {batch.shape}")
                perturbed_batch = graph.sample_transition(batch, sigma)
                try:
                    score = score_fn(perturbed_batch, sigma)
                except:
                    raise ValueError(f"Could not compute score for batch with shape {perturbed_batch.shape} and sigma shape {sigma.shape}")
                divergence_estimate = graph.score_logprobability(score, dsigma, perturbed_batch).mean().item()
                estimates.append(divergence_estimate)
                step += 1
        factor = batch_dims[1] if indeces_to_keep is None else len(indeces_to_keep)
        print("Mean estimate: ", np.log(config.tokens*factor) - np.mean(estimates), "Mean kl: ",np.mean(estimates),"Std estimate: ", np.std(estimates), "Some stimates: ", estimates[:5])
        return np.log(config.tokens*factor) - np.mean(estimates)
    return estimate_entropy_fn


def get_mutinfo_estimate_fn(config, graph, noise, batch_dims, eps, device, p=None, proj_fun = lambda x: x, indeces_to_keep=None):
    
    sampling_fn = get_pc_sampler_for_mutinfo_estimate(graph=graph,
                                                      noise=noise,
                                                      batch_dims=batch_dims,
                                                      predictor=config.sampling.predictor,
                                                      steps=config.sampling.steps,
                                                      denoise=config.sampling.noise_removal,
                                                      vocab_size=config.tokens,
                                                      eps=eps,
                                                      device=device,
                                                      proj_fun=proj_fun,
                                                      p=p,
                                                      indeces_to_keep=indeces_to_keep)
    
    def estimate_mutinfo_fn(model):
        estimates = []
        for i in tqdm(range(config.paths)):
            kl = sampling_fn(model)
            estimates.append(kl)
        # print("Estimates: ", estimates)
        print("Mean estimate: ", np.mean(estimates), "Mean kl: ",np.mean(estimates),"Std estimate: ", np.std(estimates))
        return np.mean(estimates)
    
    return estimate_mutinfo_fn

def get_entropy_montecarlo_estimate_fn(config, graph, noise, batch_dims, eps, device, p=None):
        
        def estimate_entropy_fn(model, eval_loader):
            predictor = get_predictor(config.sampling.predictor)(graph, noise)
            if p is None:
                score_fn = mutils.get_score_fn(model, train=False, sampling=True)
            else:
                print("Using analytic score")
                score_fn = lambda x, sigma: graph.get_analytic_score(x, p, sigma)

            n = int(1/eps)

            index_hist = torch.zeros(n, device=device)
            
            with torch.no_grad():
                step = 0
                estimates = []
                for batch in tqdm(eval_loader, desc="Estimating Entropy", total=config.paths):
                    if step==config.paths:
                        break
                    if config.data.valid != "text8" and config.data.valid != "bernoulli":
                        batch = batch['input_ids'].to(device)
                    else:
                        if config.data.valid == "bernoulli":
                            batch = batch["feature"].to(device)
                        else:
                            batch = batch.to(device)
                    time_indeces = torch.randint(1, n+1, (batch.shape[0],), device=device)
                    counts = torch.bincount(time_indeces, minlength=n+1)
                    # unique_indeces_number = torch.unique(time_indeces).shape[0]
                    # print(f"n: {n}, Counts shape: {counts.shape}, index_hist shape: {index_hist.shape}, time indices max: {time_indeces.max()}, time indices min: {time_indeces.min()}, unique indeces number: {unique_indeces_number}")
                    index_hist += counts[1:] # Because we don't have zeros
                    t = time_indeces/n
                    sigma, dsigma = noise(t)
                    # raise UserWarning(f"t is {t}, sigma is {sigma}, batch shape is {batch.shape}")
                    perturbed_batch = graph.sample_transition(batch, sigma[:, None])
                    ratio = predictor.get_ratio_with_uniform(score_fn, perturbed_batch, t, eps)
                    estimates.append(torch.log(ratio).mean().item())
                    if step % 100 == 0:
                        print(sigma[:5])
                        print(t[:5])
                        kl = n*np.mean(estimates)
                        h = np.log(config.tokens*batch_dims[1]) - kl
                        tqdm.write(f"Entropy estimate: {h}, kl: {kl}, kl unnormalised {np.mean(estimates)}")
                    step += 1
                print(f"Entropy estimate: {np.log(config.tokens*batch_dims[1])-n*np.mean(estimates)}")
                print(f"Index probabilities: {index_hist/torch.sum(index_hist)}, with variance {torch.var(index_hist/torch.sum(index_hist))}")
                return np.log(config.tokens*batch_dims[1])-n*np.mean(estimates)
                    
        return estimate_entropy_fn

"""def get_mutinfo_estimate_fn(config, graph, noise, batch_dims, eps, x_indices, y_indices, device):

    get_pc_sampler_for_mutinfo_estimate = lambda bdims, proj_fun, indeces_to_keep: get_pc_sampler_for_entropy_estimate(graph,
                                                      noise,
                                                      bdims,
                                                      config.sampling.predictor,
                                                      config.sampling.steps,
                                                      config.tokens,
                                                      config.sampling.noise_removal,
                                                      eps,
                                                      device,
                                                      proj_fun,
                                                      indeces_to_keep)

    def estimate_mutinfo_fn(model, eval_loader):
        def get_proj_fun(input_ids, input_locs):
            def proj_fun(x):
                x[:, input_locs] = input_ids
                return x
            return proj_fun
        
        with torch.no_grad():
            step = 0
            estimates = []
            estimates_for_entropy = []
            estimate_cond1 = []
            estimate_cond2 = []
            for batch in tqdm(eval_loader, desc="Estimating MI", total=config.paths):
                if step==config.paths:
                    break
                if config.data.valid != "text8" and config.data.valid != "bernoulli":
                    batch = batch['input_ids'].to(device)
                else:
                    if config.data.valid == "bernoulli":
                        batch = batch["feature"].to(device)
                    else:
                        batch = batch.to(device)
                sampling_fn_joint = get_pc_sampler_for_mutinfo_estimate(batch.shape, lambda x: x, None)
                
                proj_1 = get_proj_fun(batch[:, x_indices], x_indices)
                sampling_fn_cond1 = get_pc_sampler_for_mutinfo_estimate(batch.shape, proj_1, x_indices)

                proj_2 = get_proj_fun(batch[:, y_indices], y_indices)
                sampling_fn_cond2 = get_pc_sampler_for_mutinfo_estimate(batch.shape, proj_2, y_indices)
                
                val = sampling_fn_joint(model)
                estimates_for_entropy.append(val)
                estimate_cond1.append(sampling_fn_cond1(model))
                estimate_cond2.append(sampling_fn_cond2(model))
                val -= estimate_cond1[-1] # Entropy estimate must take into account lower dimension for probability distribution, since it's marginalized
                val -= estimate_cond2[-1]
                estimates.append(val)
                dim_2_shape = batch.shape[1]
                step += 1
            print(f"Mutinfo estimates: {np.mean(estimates)}")
            print(f"Entropy estimates: {np.log(config.tokens*dim_2_shape)-np.mean(estimates_for_entropy)}")
            print(f"KL joint estimates: {np.mean(estimates_for_entropy)}")
            print(f"KL cond1 estimates: {np.mean(estimate_cond1)}")
            print(f"KL cond2 estimates: {np.mean(estimate_cond2)}")
            return np.mean(estimates)+np.log(config.tokens/2)
                
    return estimate_mutinfo_fn
"""    

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


def get_pc_sampler_for_entropy_estimate(graph, noise, batch_dims, predictor, steps, vocab_size, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x, indeces_to_keep=None, p=None):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun

    @torch.no_grad()
    def pc_sampler(model):
        if p is None:
            sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        else:
            sampling_score_fn = lambda x, sigma: graph.get_analytic_score(x, p, sigma)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps
        log_radon_sum = 0

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            ratio = predictor.get_ratio_with_uniform(sampling_score_fn, x, t, dt, proj_fn=projector, indeces_to_keep=indeces_to_keep)
            # raise UserWarning(f"Shapes were {x.shape}, {t.shape}, {dt}, {ratio.shape}")
            # print("ratio is", ratio[:5]) are all around 2
            # print("log ratio is", torch.log(ratio)[:5])
            log_radon_sum = log_radon_sum + torch.log(ratio)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)
        
        # print("Log Radon sum: ", log_radon_sum)
        return torch.mean(log_radon_sum).item()
    
    return pc_sampler


def get_pc_sampler_for_mutinfo_estimate(graph, noise, batch_dims, predictor, steps, vocab_size, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x, indeces_to_keep=None, p=None):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun

    @torch.no_grad()
    def pc_sampler(model):
        if p is None:
            sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        else:
            joint_score_fn = lambda x, sigma: graph.get_analytic_score(x, p, sigma)
            px = torch.sum(p, axis=1)
            py = torch.sum(p, axis=0)
            # print("PX: ", px)
            # print("PY: ", py)
            # print("PXY: ", p)
            pxy_margin = px * py.T
            # print("PXY MARGINAL: ", pxy_margin)
            pxy_margin = pxy_margin.unsqueeze(-1)
            marginal_score_fn = lambda x, sigma: graph.get_analytic_score(x, pxy_margin, sigma)
 
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps
        log_radon_sum = 0

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            ratio = predictor.get_ratio_with_marginal(joint_score_fn, marginal_score_fn, x, t, dt, proj_fn=projector, indeces_to_keep=indeces_to_keep)
            
            # raise UserWarning(f"Shapes were {x.shape}, {t.shape}, {dt}, {ratio.shape}")
            # print("ratio is", ratio[:5]) are all around 2
            # print("log ratio is", torch.log(ratio)[:5])
            log_radon_sum = log_radon_sum + torch.log(ratio)
            x = predictor.update_fn(joint_score_fn, x, t, dt)
        
        # print("Log Radon sum: ", log_radon_sum)
        return torch.mean(log_radon_sum).item()
    
    return pc_sampler


"""def get_pc_sampler_for_mutinfo_estimate(graph, noise, batch_dims, predictor, steps, vocab_size, vocab_indices, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
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
            ratio = predictor.get_ratio_with_uniform(sampling_score_fn, x, t, dt, projector)
            log_ratio = log_ratio + torch.log(ratio)
            # print("ratio is", ratio[:5]) are all around 2
            # print("log ratio is", torch.log(ratio)[:5])
            log_radon_sum = log_radon_sum + log_ratio
            x = predictor.update_fn(sampling_score_fn, x, t, dt)
        
        # print("Log Radon sum: ", log_radon_sum)
        return torch.mean(log_radon_sum).item()
    
    return pc_sampler"""
