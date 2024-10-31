import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


from catsample import sample_categorical

def get_graph(config, device):
    if config.graph.type == "uniform":
        return Uniform(config.tokens)
    elif config.graph.type == "absorb":
        return Absorbing(config.tokens)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @dim.setter
    def dim(self, value):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass


    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass


    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass


    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    
    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass
    

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass


    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass

    def get_pt(self, p, sigma):
        """
        Computes the diffused probability given sigma.
        """
        # p is of shape (tokens, dim, 1)
        indeces = torch.arange(self.dim, device=p.device).unsqueeze(0).unsqueeze(0)
        indeces = indeces.expand(sigma.shape[0],p.shape[0],-1)
        # indeces = torch.arange(graph.dim).reshape(-1,1)
        exp_qt = self.transition(indeces,sigma[...,None])
        p = p.expand(sigma.shape[0],p.shape[0],-1,-1)
        p_t = exp_qt@p
        return p_t

    def get_analytic_score(self, x, p, sigma):
        """
        Computes the score function given sigma.
        """
        assert len(p.shape) == 3, f"p must be of shape (tokens, dim, 1), instead got shape {p.shape}"
        assert x.shape[1] == p.shape[0], f"p must match x for number of tokens, instead got x={x.shape} and p={p.shape}"
        assert x.shape[0] == sigma.shape[0], "sigma must match x for batch size"
        p_t = self.get_pt(p,sigma)
        score_matrix = p_t.permute(0,1,3,2).expand(-1,-1,self.dim,-1)/p_t.expand(-1,-1,-1,self.dim)
        index_tensor = x[...,None,None].expand(-1,-1,-1,self.dim)
        score = torch.gather(score_matrix,2,index_tensor)
        return score.squeeze(2)
    
    def score_divergence(self, score_p, score_q, dsigma, x):

        # score expected s_theta(x)_y, NOT log s_theta(x)_y

        print(f"Shapes: score_p: {score_p.shape}, score_q: {score_q.shape}, dsigma: {dsigma.shape}, x: {x.shape}")

        print(f"score_p examples: {score_p[:5]}")
        print(f"score_q examples: {score_q[:5]}")
        print(f"dsigma examples: {dsigma[:5]}")
        print(f"x examples: {x[:5]}")

        x = x.unsqueeze(-1)
        
        log_score_p = torch.scatter(score_p.log(), -1, x, torch.zeros_like(score_p))
        score_p = torch.scatter(score_p, -1, x, torch.zeros_like(score_p))

        log_score_q = torch.scatter(score_q.log(), -1, x, torch.zeros_like(score_q))
        score_q = torch.scatter(score_q, -1, x, torch.zeros_like(score_q))

        print(f"Shapes after scatter: score_p: {score_p.shape}, score_q: {score_q.shape}, x: {x.shape}, log_score_p: {log_score_p.shape}, log_score_q: {log_score_q.shape}")
        
        neg_term = log_score_q * score_p

        # constant factor
        const = score_p * (log_score_p - 1)

        #positive term
        pos_term = torch.scatter(score_q, -1, x, torch.zeros_like(score_q))

        print(f"pos term examples: {pos_term[:5]}")
        print(f"neg term examples: {neg_term[:5]}")
        print(f"const examples: {const[:5]}")

        unscaled_ret_value = pos_term - neg_term + const
        print(f"Unscaled ret value examples: {unscaled_ret_value[:5]}")

        print(f"Shapes: unscaled_ret_value: {unscaled_ret_value.shape}, pos_term: {pos_term.shape}, neg_term: {neg_term.shape}, const: {const.shape}")

        x = x.squeeze(-1)

        transp_rate = self.transp_rate(x)
        try:
            scale_factor = torch.scatter(transp_rate, -1, x[...,None], torch.zeros_like(transp_rate))
        except:
            raise ValueError(f"Could not scatter {transp_rate.shape} with {x.shape}")
        scale_factor = scale_factor * dsigma[..., None]

        print(f"Shapes: scale_factor: {scale_factor.shape}, transp_rate: {transp_rate.shape}")

        ret = (scale_factor * unscaled_ret_value).sum(dim=-1)

        print(f"Shapes: ret: {ret.shape}, ret examples: {ret[:5]}")

        raise ValueError("Stop here")
        return ret
    
    def score_logprobability(self, score_p, dsigma, x):

        x = x.unsqueeze(-1)
        
        log_score_p = torch.scatter(score_p.log(), -1, x, torch.zeros_like(score_p))
        score_p = torch.scatter(score_p, -1, x, torch.zeros_like(score_p))

        # constant factor
        const = score_p * (log_score_p - 1) + 1

        unscaled_ret_value = const

        x = x.squeeze(-1)

        transp_rate = self.transp_rate(x)
        try:
            scale_factor = torch.scatter(transp_rate, -1, x[...,None], torch.zeros_like(transp_rate))
        except:
            raise ValueError(f"Could not scatter {transp_rate.shape} with {x.shape}")
        scale_factor = scale_factor * dsigma[..., None]

        ret = (scale_factor * unscaled_ret_value).sum(dim=-1)

        """print(f"Some shapes: ret: {ret.shape}, ret examples: {ret[:5]}, scale_factor: {scale_factor.shape}, unscaled_ret_value: {unscaled_ret_value.shape}")
        print(f"Other shapes: x: {x.shape}, log_score_p: {log_score_p.shape}, score_p: {score_p.shape}")
        print(f"score_p examples: {score_p[:5]}, log_score_p examples: {log_score_p[:5]}")
        print(f"Other shapes: dsigma: {dsigma.shape}")
        raise ValueError("Stop here")"""

        return ret


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """
    def __init__(self, dim):
        self._dim = dim
        Q = torch.ones((dim,dim))
        for i in range(dim):    
            Q[i,i] = 1-dim
        self._Q = Q

    @property
    def dim(self):
        return self._dim
    
    @dim.setter
    def dim(self, value):
        self._dim = value
    
    @property
    def absorb(self):
        return False
    
    @property
    def Q(self):
        return self._Q


    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        try:
            trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        except:
            raise ValueError(f"sigma shape: {sigma[..., None].shape}, i shape: {torch.ones(*i.shape, self.dim, device=i.device).shape}")
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans
    
    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        #positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const


class Absorbing(Graph):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy
    