import torch
import torch.nn.functional as F
import numpy as np

from itertools import cycle

def get_model_fn(model, train=False, marginal_flag=None):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(x, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """

        if train:
            model.train()
        else:
            model.eval()
        
            # otherwise output the raw values (we handle mlm training in losses.py)
        if marginal_flag is None:
            return model(x, sigma, output_hidden_states=False)
        return model(x, sigma, marginal_flag=marginal_flag)

    return model_fn

def get_score_fn(model, train=False, sampling=False, marginal_flag=None, scorify=False):
    if scorify:
        return get_score_from_denoiser_fn(model, train=train, sampling=sampling, marginal_flag=marginal_flag)
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train, marginal_flag=marginal_flag)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        def score_fn(x, sigma):
            
            # sigma = sigma.reshape(-1)

            try:
                score = model_fn(x, sigma)
            except:
                raise UserWarning(f"Devices: {x.device}, {sigma.device}\n\
                    Shapes: {x.shape}, {sigma.shape}\n\
                    Types: {x.dtype}, {sigma.dtype}\n")
            
            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()
                
            return score

    return score_fn

def get_score_from_denoiser_fn(denoiser, train=False, sampling=False, marginal_flag=None):
    model_fn = get_model_fn(denoiser, train=train, marginal_flag=marginal_flag)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        def score_fn(x, sigma):
            logits = model_fn(x, sigma)
            # score(x, t) = p_t(y) / p_t(x)
            # => log score(x, t) = log p_t(y) - log p_t(x)
            
            # case 1: x = masked
            #   (i) y = unmasked
            #     log score(x, t) = log p_\theta(x)|_y + log k
            #     where k = exp(- sigma) / (1 - exp(- sigma))
            #   (ii) y = masked
            #     log score(x, t) = 0

            # case 2: x = unmasked
            #   (i) y != masked, y != x
            #     log score(x_i, t) = - inf
            #   (ii) y = x 
            #     log score(x_i, t) = 0
            #   (iii) y = masked token
            #     log score(x_i, t) = - log k
            #     where k = exp(- sigma) / (1 - exp(- sigma))

            neg_infinity = -1000000000

            logits[:, :, -1] += neg_infinity
    
            # Normalize the logits such that x.exp() is
            # a probability distribution over vocab_size.
            logits = logits - torch.logsumexp(logits, dim=-1,
                                            keepdim=True)

            # Apply updates directly in the logits matrix.
            # For the logits of the unmasked tokens, set all values
            # to -infinity except for the indices corresponding to
            # the unmasked tokens.
            unmasked_indices = (x != 1024)
            logits[unmasked_indices] = neg_infinity
            logits[unmasked_indices, x[unmasked_indices]] = 0
            
            model_output = logits
            
            log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
            assert log_k.ndim == 1
            
            masked_score = model_output + log_k[:, None, None]
            masked_score[:, :, -1] = 0

            unmasked_score = neg_infinity * torch.ones_like(
            model_output)
            unmasked_score = torch.scatter(
            unmasked_score,
            -1,
            x[..., None],
            torch.zeros_like(unmasked_score[..., :1]))
            unmasked_score[:, :, -1] = - (
            log_k[:, None] * torch.ones_like(x))
            
            masked_indices = (x == -1).to(
            model_output.dtype)[:, :, None]
            model_output = (
            masked_score * masked_indices
            + unmasked_score * (1 - masked_indices))
            return model_output.exp()
    return score_fn

