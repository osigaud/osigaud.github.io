
# %% [markdown]

# The function below builds a multi-layer perceptron where the size of each layer is given in the `size` list.
# We also specify the activation function of neurons at each layer and optionally a different activation function for the final layer.
# The layers are initialized orthogonal way, which is known to provide better performance in PPO

# %%

import numpy as np
import torch.nn as nn

def ortho_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Function used for orthogonal inialization of the layers
    Taken from here in the cleanRL library: https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

# %%

def build_ortho_mlp(sizes, activation, output_activation=nn.Identity()):
    """Helper function to build a multi-layer perceptron (function from $\mathbb R^n$ to $\mathbb R^p$)
    with orthogonal initialization
    
    Args:
        sizes (List[int]): the number of neurons at each layer
        activation (nn.Module): a PyTorch activation function (after each layer but the last)
        output_activation (nn.Module): a PyTorch activation function (last layer)
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [ortho_init(nn.Linear(sizes[j], sizes[j + 1])), act]
    return nn.Sequential(*layers)
