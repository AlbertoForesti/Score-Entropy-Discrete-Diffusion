import torch
from mutinfo.estimators.neural.mine import _MINE_backbone

def get_backbone_factory(hidden_size, n_layers, activation, print_num_params=True):

    def backbone_factory(x_shape, y_shape):

        layers = []
        layers.append(torch.nn.Linear(x_shape[-1] + y_shape[-1], hidden_size))
        layers.append(activation())

        assert n_layers >= 2, "Number of layers must be at least 2"

        for l in range(n_layers-2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(activation())
        
        layers.append(torch.nn.Linear(hidden_size, 1))
        backbone = torch.nn.Sequential(*layers)

        if print_num_params:
            num_params = sum(p.numel() for p in backbone.parameters())
            print(f"Number of parameters in MINE backbone: {num_params}")

        return _MINE_backbone(backbone)
    
    return backbone_factory
        