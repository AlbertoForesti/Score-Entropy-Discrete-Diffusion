import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

try:
    from infosedd.model.transformer import SEDD
except:
    from transformer import SEDD

class DoubleSEDD(nn.Module, PyTorchModelHubMixin):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model1 = SEDD(config)
        self.model2 = SEDD(config)
    
    def forward(self, x, sigma, is_marginal=False):
        if is_marginal:
            return self.model1(x, sigma, is_marginal=is_marginal)
        else:
            return self.model2(x, sigma, is_marginal=is_marginal)