
from functools import partial
import torch
from torch import nn


# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(sigma, *args, **kwargs):
    return sigma


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        # nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Linear(dim, default(dim_out, dim))
    )


def Downsample(dim, dim_out=None):
    return nn.Linear(dim, default(dim_out, dim))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, shift_scale=True):
        super().__init__()
        # self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.proj = nn.Linear(dim, dim_out)
        self.act = nn.SiLU()
        # self.act = nn.Relu()
        self.norm = nn.GroupNorm(groups, dim)
        # self.norm = nn.BatchNorm1d( dim)
        self.shift_scale = shift_scale

    def forward(self, x, sigma=None):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)

        if exists(sigma):
            if self.shift_scale:
                scale, shift = sigma
                x = x * (scale.squeeze() + 1) + shift.squeeze()
            else:
                x = x + sigma

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, sigma_emb_dim=None, groups=32, shift_scale=False):
        super().__init__()
        self.shift_scale = shift_scale
        self.mlp = nn.Sequential(
            nn.SiLU(),
            # nn.Linear(sigma_emb_dim, dim_out * 2)
            nn.Linear(sigma_emb_dim, dim_out*2 if shift_scale else dim_out)
        ) if exists(sigma_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups,
                            shift_scale=shift_scale)
        self.block2 = Block(dim_out, dim_out, groups=groups,
                            shift_scale=shift_scale)
        # self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.lin_layer = nn.Linear(
            dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):

            time_emb = self.mlp(time_emb)
            scale_shift = time_emb

        h = self.block1(x, sigma=scale_shift)

        h = self.block2(h)

        return h + self.lin_layer(x)


class UnetMLP_simple(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()

        # determine dimensions
        self.sequence_length = config.seq_length
        self.alphabet_size = config.alphabet_size
        self.absorb = config.graph.type == "absorb"
        self.resnet_block_groups = config.model.resnet_block_groups
        self.sigma_dim = config.model.sigma_dim
        self.dim_mults = config.model.dim_mults

        try:
            self.vocab_size = self.alphabet_size + (1 if self.absorb else 0)
        except:
            self.vocab_size = self.alphabet_size + (1 if self.absorb else 0)
        init_dim = default(config.model.init_dim, self.sequence_length+1)
        if init_dim == None:
            init_dim = (self.sequence_length + 1) * self.dim_mults[0]

        dim_in = self.sequence_length + 1 # 1 for marginal flag
        dims = [init_dim, *map(lambda m: init_dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=self.resnet_block_groups)

        self.init_lin = nn.Linear(dim_in, init_dim)

        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, self.sigma_dim),
            nn.GELU(),
            nn.Linear(self.sigma_dim, self.sigma_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            module = nn.ModuleList([block_klass(dim_in, dim_in, sigma_emb_dim=self.sigma_dim),
                                    #        block_klass(dim_in, dim_in, sigma_emb_dim = sigma_dim)
                                    ])

            # module.append( Downsample(dim_in, dim_out) if not is_last else nn.Linear(dim_in, dim_out))
            self.downs.append(module)

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, sigma_emb_dim=self.sigma_dim)

        # self.mid_block2 = block_klass(joint_dim, mid_dim, sigma_emb_dim = sigma_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            module = nn.ModuleList([block_klass(dim_out + dim_in, dim_out, sigma_emb_dim=self.sigma_dim),
                                    #       block_klass(dim_out + dim_in, dim_out, sigma_emb_dim = sigma_dim)
                                    ])
            # module.append( Upsample(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in))
            self.ups.append(module)

        # default_out_dim = channels * (1 if not learned_variance else 2)

        self.out_dim = dim_in

        self.final_res_block = block_klass(
            init_dim * 2, init_dim, sigma_emb_dim=self.sigma_dim)

        self.proj = nn.Linear(init_dim, self.vocab_size*self.sequence_length)

        # self.proj.weight.data.fill_(0.0)
        # self.proj.bias.data.fill_(0.0)

        self.final_lin = nn.Sequential(
            nn.GroupNorm(self.resnet_block_groups, init_dim),
            nn.SiLU(),
            self.proj
        )

    def forward(self, indices, sigma, is_marginal=False, std=None):
        sigma = sigma.reshape(sigma.size(0), 1)

        is_marginal_flag = -torch.ones_like(sigma) if is_marginal else torch.ones_like(sigma)
        x = torch.cat((indices, is_marginal_flag), dim=1)
        try:        
            x = self.init_lin(x.float())
        except:
            raise UserWarning(f"x shape {x.shape} x.float() shape {x.float().shape} sigma shape {sigma.shape} init_lim input dim {self.init_lin.in_features}")

        r = x.clone()

        sigma = self.sigma_mlp(sigma).squeeze()

        h = []

        for blocks in self.downs:

            block1 = blocks[0]

            x = block1(x, sigma)

            h.append(x)
       #     x = downsample(x)

        # x = self.mid_block1(x, sigma)

        # x = self.mid_block2(x, sigma)

        for blocks in self.ups:

            block1 = blocks[0]
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, sigma)

            # x = torch.cat((x, h.pop()), dim = 1)
            # x = block2(x, sigma)

           # x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, sigma)

        if std != None:
            x = self.final_lin(x) / std
        else:
            x = self.final_lin(x)
        
        x = x.view(-1, self.sequence_length, self.vocab_size)
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        return x
