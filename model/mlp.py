import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionMLP(nn.Module):
    def __init__(self, config):
        super(DiffusionMLP, self).__init__()
        self.sequence_length = config.seq_length
        self.absorb = config.graph.type == "absorb"
        try:
            vocab_size = config.alphabet_size + (1 if self.absorb else 0)
        except:
            vocab_size = config.tokens + (1 if self.absorb else 0)
        self.vocab_size = vocab_size
        self.hidden_dim = config.model.hidden_size

        # Define the layers
        self.fc1 = nn.Linear(self.sequence_length + 1, self.hidden_dim)  # +1 for sigma
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.sequence_length * self.vocab_size)

    def forward(self, indices, sigma, is_marginal=False):
        # Concatenate x and sigma
        sigma_expanded = sigma.unsqueeze(1)
        x = torch.cat((indices, sigma_expanded), dim=1)

        # Pass through the MLP
        # raise UserWarning(f"x shape is {x.shape}: {x}")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape the output to (batch_size, sequence_length, num_tokens)
        x = x.view(-1, self.sequence_length, self.vocab_size)
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        return x