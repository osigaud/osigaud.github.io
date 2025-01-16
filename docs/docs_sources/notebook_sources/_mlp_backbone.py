
# %% [markdown]

# ### Functions to build networks

# We define a few utilitary functions to build neural networks

# %% tags=["copy"]
from _mlp import *

# %%
def build_backbone(sizes, activation):
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation]
    return layers
