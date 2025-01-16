# %%

from bbrl import get_arguments, get_class
from itertools import chain

def setup_optimizer(cfg_optimizer, *agents):
    """Setup an optimizer for a list of agents"""
    optimizer_args = get_arguments(cfg_optimizer)
    parameters = [agent.parameters() for agent in agents]
    optimizer = get_class(cfg_optimizer)(chain(*parameters), **optimizer_args)
    return optimizer

def copy_parameters(model_a, model_b):
    """Copy parameters from a model a to model_b"""
    for model_a_p, model_b_p in zip(model_a.parameters(), model_b.parameters()):
        model_b_p.data.copy_(model_a_p)
