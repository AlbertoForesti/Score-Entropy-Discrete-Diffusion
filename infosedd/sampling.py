import abc
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

try:
    from infosedd.model import utils as mutils
    from infosedd.catsample import sample_categorical
except:
    from model import utils as mutils
    from catsample import sample_categorical


_PREDICTORS = {}

available_distributions = ["bernoulli", "binomial", "custom_joint", "custom_univariate","categorical", "xor"]

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
        # print("score is_marginal examples: ", score_marginal[:5])
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
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device, p=None):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device,
                                 p=p)
    
    return sampling_fn

def get_mutinfo_dynkin_estimate_fn(config, graph, noise, batch_dims, eps, device, x_indices, y_indices, p=None, proj_fun = lambda x: x, indeces_to_keep=None):
    
    def estimate_mutinfo_fn(model, data_loader):
        if p is None:
            score_fn = mutils.get_score_fn(model, train=False, sampling=True, is_marginal=False)
        else:
            joint_score_fn = lambda x, s: graph.get_analytic_score(x, p, s)
            px = torch.sum(p, axis=1)
            py = torch.sum(p, axis=0)
            print("PX: ", px, " with shape ", px.shape)
            print("PY: ", py, " with shape ", py.shape)
            print("PXY: ", p, " with shape ", p.shape)
            pxy_margin = px * py.T
            pxy_margin = pxy_margin.unsqueeze(-1)
            print("PXY is_marginal: ", pxy_margin, " with shape ", pxy_margin.shape)
            marginal_score_fn = lambda x, s: graph.get_analytic_score(x, pxy_margin, s)
        estimates = []
        with torch.no_grad():
            step = 0
            stop = False
            progress_bar = tqdm(total=config.mc_estimates, desc="Estimating Mutual Information")
            while not stop:
                for batch in data_loader:
                    progress_bar.update(1)
                    batch = batch.to(device)
                    if step==config.mc_estimates:
                        stop = True
                        break
                    t = torch.rand(batch.shape[0], 1, device=device)
                    sigma, dsigma = noise(t)
                    # raise UserWarning(f"t is {t}, sigma is {sigma}, batch shape is {batch.shape}")
                    perturbed_batch = graph.sample_transition(batch, sigma)
                    
                    score_joint = score_fn(perturbed_batch, sigma)

                    perturbed_batch_x = perturbed_batch.clone()
                    perturbed_batch_x[:, y_indices] = graph.dim - 1

                    perturbed_batch_y = perturbed_batch.clone()
                    perturbed_batch_y[:, x_indices] = graph.dim - 1

                    score_marginal_x = score_fn(perturbed_batch_x, sigma)
                    score_marginal_x = score_marginal_x[:, x_indices]

                    score_marginal_y = score_fn(perturbed_batch_y, sigma)
                    score_marginal_y = score_marginal_y[:, y_indices]

                    score_marginal = torch.cat([score_marginal_x, score_marginal_y], dim=1)

                    score_marginal = torch.where(torch.isnan(score_marginal), torch.ones_like(score_marginal), score_marginal)
                    score_joint = torch.where(torch.isnan(score_joint), torch.ones_like(score_joint), score_joint)

                    score_marginal = torch.where(torch.isinf(score_marginal), torch.ones_like(score_marginal), score_marginal)
                    score_joint = torch.where(torch.isinf(score_joint), torch.ones_like(score_joint), score_joint)

                    score_marginal = torch.where(score_marginal<1e-5, 1e-5*torch.ones_like(score_marginal), score_marginal)
                    score_joint = torch.where(score_joint<1e-5, 1e-5*torch.ones_like(score_joint), score_joint)
                    
                    # raise UserWarning(f"Score joint examples {score_joint[:5]}, x examples {perturbed_batch[:5]}")
                    divergence_estimate = graph.score_divergence(score_joint, score_marginal, dsigma, perturbed_batch)
                    estimates.append(divergence_estimate.mean().item())
                    step += 1
            progress_bar.close()
        print("Mean estimate: ", np.mean(estimates), "Mean kl: ",np.mean(estimates),"Std estimate: ", np.std(estimates), "Some stimates: ", estimates[:5], "N estimates: ", len(estimates))
        return np.mean(estimates)
    return estimate_mutinfo_fn

def get_entropy_dynkin_estimate_fn(config, graph, noise, batch_dims, eps, device, p=None, proj_fun = lambda x: x, indeces_to_keep=None):
    
    def estimate_entropy_fn(model, data_loader):
        indeces_to_keep = None
        if p is None:
            score_fn = mutils.get_score_fn(model, train=False, sampling=True, is_marginal=False)
        else:
            score_fn = lambda x, s: graph.get_analytic_score(x, p, s)
        estimates = []
        with torch.no_grad():
            step = 0
            stop = False
            progress_bar = tqdm(total=config.mc_estimates, desc="Estimating Entropy")
            while not stop:
                for batch in data_loader:
                    progress_bar.update(1)
                    batch = batch.to(device)
                    if step==config.mc_estimates:
                        stop = True
                        break
                    t = torch.rand(batch.shape[0], 1, device=device)
                    sigma, dsigma = noise(t)
                    # raise UserWarning(f"t is {t}, sigma is {sigma}, batch shape is {batch.shape}")
                    perturbed_batch = graph.sample_transition(batch, sigma)

                    if hasattr(config, 'conditioning_indices') and config.conditioning_indices is not None:
                        indeces_to_keep = config.conditioning_indices
                        perturbed_batch[:, indeces_to_keep] = batch[:, indeces_to_keep]
                    
                    score = score_fn(perturbed_batch, sigma)

                    if hasattr(config, 'conditioning_indices') and config.conditioning_indices is not None:
                        indeces_to_keep = config.conditioning_indices
                        score = score[:, indeces_to_keep]
                        perturbed_batch = perturbed_batch[:, indeces_to_keep]
                    
                    # raise UserWarning(f"Score joint examples {score_joint[:5]}, x examples {perturbed_batch[:5]}")
                    divergence_estimate = graph.score_logprobability(score, dsigma, perturbed_batch, sigma)
                    estimates.append(divergence_estimate.mean().item())
                    step += 1
            progress_bar.close()
        factor = config.seq_length if indeces_to_keep is None else len(indeces_to_keep)
        print("Mean estimate: ", factor*np.log(config.alphabet_size) - np.mean(estimates), "Mean kl: ",np.mean(estimates),"Std estimate: ", np.std(estimates), "Some stimates: ", estimates[:5])
        return factor*np.log(config.alphabet_size) - np.mean(estimates)
    return estimate_entropy_fn

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x, p=None):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        if p is None:
            sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True, is_marginal=False)
        else:
            sampling_score_fn = lambda x, s: graph.get_analytic_score(x, p, s)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in tqdm(range(steps), desc="Sampling"):
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