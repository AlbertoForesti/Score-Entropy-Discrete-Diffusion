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

def get_mutinfo_step_fn(config, graph, noise, proj_fn = lambda x: x):

    def mutinfo_step_fn(model, batch):
        if config.is_parametric_marginal:
            score_fn_x = mutils.get_score_fn(model, train=False, sampling=True, marginal_flag=0)
            score_fn_y = mutils.get_score_fn(model, train=False, sampling=True, marginal_flag=1)
            score_fn_joint = mutils.get_score_fn(model, train=False, sampling=True, marginal_flag=2)
        else:
            score_fn = mutils.get_score_fn(model, train=False, sampling=True, marginal_flag=None)
        with torch.no_grad():
            t = torch.rand(batch.shape[0], 1).to(batch.device)
            sigma, dsigma = noise(t)
            
            # raise UserWarning(f"t is {t}, sigma is {sigma}, batch shape is {batch.shape}")
            perturbed_batch = graph.sample_transition(batch, sigma)
            perturbed_batch = proj_fn(perturbed_batch)

            perturbed_batch_x = perturbed_batch.clone()
            perturbed_batch_x[:, config.y_indices] = graph.dim - 1

            perturbed_batch_y = perturbed_batch.clone()
            perturbed_batch_y[:, config.x_indices] = graph.dim - 1

            if config.is_parametric_marginal:
                try:
                    score_joint = score_fn_joint(perturbed_batch, sigma)
                except:
                    raise UserWarning(f"Device is {sigma.device}, perturbed batch is {perturbed_batch.device}")
                score_marginal_x = score_fn_x(perturbed_batch_x, sigma)
                score_marginal_y = score_fn_y(perturbed_batch_y, sigma)
            else:
                score_joint = score_fn(perturbed_batch, sigma)
                score_marginal_x = score_fn(perturbed_batch_x, sigma)
                score_marginal_y = score_fn(perturbed_batch_y, sigma)

            score_marginal_x = score_marginal_x[:, config.x_indices]

            score_marginal_y = score_marginal_y[:, config.y_indices]

            score_marginal = torch.cat([score_marginal_x, score_marginal_y], dim=1)

            score_marginal = torch.where(torch.isnan(score_marginal), torch.ones_like(score_marginal), score_marginal)
            score_joint = torch.where(torch.isnan(score_joint), torch.ones_like(score_joint), score_joint)

            score_marginal = torch.where(torch.isinf(score_marginal), torch.ones_like(score_marginal), score_marginal)
            score_joint = torch.where(torch.isinf(score_joint), torch.ones_like(score_joint), score_joint)

            score_marginal = torch.where(score_marginal<1e-5, 1e-5*torch.ones_like(score_marginal), score_marginal)
            score_joint = torch.where(score_joint<1e-5, 1e-5*torch.ones_like(score_joint), score_joint)
            
            # raise UserWarning(f"Score joint examples {score_joint[:5]}, x examples {perturbed_batch[:5]}")
            divergence_estimate = graph.score_divergence(score_joint, score_marginal, dsigma, perturbed_batch)
            return divergence_estimate.mean().item()
    
    return mutinfo_step_fn

def get_entropy_step_fn(config, graph, noise):
    def entropy_step_fn(model, batch):
        score_fn = mutils.get_score_fn(model, train=False, sampling=True, marginal_flag=None)
        with torch.no_grad():
            perturbed_batch = graph.sample_transition(batch, noise)
            t = torch.rand(batch.shape[0], 1).to(batch.device)
            sigma, dsigma = noise(t)
            score = score_fn(perturbed_batch, sigma)
            divergence_estimate = graph.score_logprobability(score, dsigma, perturbed_batch, sigma).mean().item()
            return batch.shape[1]*np.log(config.alphabet_size) - divergence_estimate
    return entropy_step_fn

def get_oinfo_step_fn(config, graph, noise):
    def oinfo_step_fn(model,batch):
        score_fn = mutils.get_score_fn(model, train=False, sampling=True, marginal_flag=None)
        with torch.no_grad():
            perturbed_batch = graph.sample_transition(batch, noise)
            t = torch.rand(batch.shape[0], 1).to(batch.device)
            sigma, dsigma = noise(t)
            score_joint = score_fn(perturbed_batch, sigma)
            all_indices = list(range(batch.shape[1]))
            marginal_scores = []
            s_infos = []

            for i in all_indices:
                all_indices_minus_i = all_indices.copy()
                all_indices_minus_i.remove(i)

                perturbed_batch_single = perturbed_batch.clone()
                perturbed_batch_single[:, all_indices_minus_i] = graph.dim - 1
                score_single = score_fn(perturbed_batch_single, sigma)
                score_single = score_single[:, [i]]

                marginal_scores.append(score_single)
                
                perturbed_batch_all_minus_i = perturbed_batch.clone()
                perturbed_batch_all_minus_i[:, i] = graph.dim - 1
                score_all_minus_i = score_fn(perturbed_batch_all_minus_i, sigma)
                # discard the i-th score
                score_all_minus_i = score_all_minus_i[:, all_indices_minus_i]

                # score_marginal_all_minus_i_and_i = torch.cat([score_single, score_all_minus_i], dim=1) # I think this is wrong, the single should be replaced at its right position
                score_marginal_all_minus_i_and_i = torch.empty_like(score_joint)
                try:
                    score_marginal_all_minus_i_and_i[:, all_indices_minus_i] = score_all_minus_i
                except:
                    raise UserWarning(f"Score all minus i shape {score_all_minus_i.shape}, score marginal all minus i and i shape {score_marginal_all_minus_i_and_i.shape}, all indices minus i {all_indices_minus_i}, all indices {all_indices}")
                score_marginal_all_minus_i_and_i[:, i] = score_single.squeeze()
                
                s_infos.append(graph.score_divergence(score_joint, score_marginal_all_minus_i_and_i, dsigma, perturbed_batch).mean().item())
            
            score_marginal = torch.cat(marginal_scores, dim=1)
            total_correlation = graph.score_divergence(score_joint, score_marginal, dsigma, perturbed_batch).mean().item()
            s_info = np.sum(s_infos)
            o_info = 2*total_correlation - s_info
            return o_info
    return oinfo_step_fn

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
        x = graph.sample_limit(*batch_dims)
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