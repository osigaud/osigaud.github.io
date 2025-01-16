# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]

# # Outlook

# This notebook is designed to understand how to use a gymnasium environment as a BBRL agent in practice, using autoreset=False.
# It is part of the [BBRL documentation](https://github.com/osigaud/bbrl/docs/index.html).

# If this is your first contact with BBRL, you may start be having a look at [this more basic notebook](01-basic_concepts.student.ipynb).
# %% [markdown]
#
# ## Installation and Imports
#
# The BBRL library is [here](https://github.com/osigaud/bbrl).

# Below, we import standard python packages, pytorch packages and gymnasium environments.

# %% tags=["hide-input"]
# Installs the necessary Python and system libraries
try:
    from easypip import easyimport, easyinstall, is_notebook
except ModuleNotFoundError as e:
    get_ipython().run_line_magic("pip", "install easypip")
    from easypip import easyimport, easyinstall, is_notebook

easyinstall("bbrl>=0.2.2")
easyinstall("swig")
easyinstall("bbrl_gymnasium>=0.2.0")
easyinstall("bbrl_gymnasium[classic_control]")

# %% tags=["hide-input"]

import os
import sys
from pathlib import Path
import math

from moviepy.editor import ipython_display as video_display
import time
from tqdm.auto import tqdm
from typing import Tuple, Optional
from functools import partial

from omegaconf import OmegaConf
import torch
import bbrl_gymnasium

import copy
from abc import abstractmethod, ABC
import torch.nn as nn
import torch.nn.functional as F
from time import strftime
OmegaConf.register_new_resolver(
    "current_time", lambda: strftime("%Y%m%d-%H%M%S"), replace=True
)

# %% tags=["hide-input"]

# Imports all the necessary classes and functions from BBRL
from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class
# The workspace is the main class in BBRL, this is where all data is collected and stored
from bbrl.workspace import Workspace

# Agents(agent1, agent2, agent3, ...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent over multiple timesteps in the workspace, 
# or until a given condition is reached

from bbrl.agents import Agents, TemporalAgent
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# %% tags=["hide-input"]

# %% [markdown]

# ## Definition of agents

# We first create an Agent representing [the CartPole-v1 gym environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
# This is done using the [ParallelGymAgent](https://github.com/osigaud/bbrl/blob/40fe0468feb8998e62c3cd6bb3a575fef88e256f/src/bbrl/agents/gymnasium.py#L261) class.

# The ParallelGymAgent is an agent able to execute a batch of gymnasium environments
# with or without auto-resetting. These agents produce multiple variables in the workspace:
# ’env/env_obs’, ’env/reward’, ’env/timestep’, ’env/terminated’,
# 'env/truncated', 'env/done', ’env/cumulated_reward’.
# 
# When called at timestep t=0, the environments are automatically reset. At
# timestep t>0, these agents will read the ’action’ variable in the workspace at
# time t − 1 to generate the next state, by calling the step(action) of the contained gymnasium environment.

# In the example below, we are working with batches (i.e. several episodes at the same time),
# so here our agent uses `n_envs = 3` environments.


# %%
# We run episodes over 3 environments at a time
n_envs = 3
env_agent = ParallelGymAgent(partial(make_env, 'CartPole-v1', autoreset=False), n_envs, reward_at_t=False)
# The random seed is set to 2139
env_agent.seed(2139)

obs_size, action_dim = env_agent.get_obs_and_actions_sizes()
print(f"Environment: observation space in R^{obs_size} and action space {{1, ..., {action_dim}}}")

# %%
# Creates a new workspace
workspace = Workspace() 

# Execute the first step
env_agent(workspace, t=0)

# Our first set of observations. The size of the observation space is 4, and we have 3 environments.
obs = workspace.get("env/env_obs", 0)
print("Observation", obs)

# %% [markdown]

# To generate more steps into the workspace, we need to send actions to the environment.

# ### Random action without agent
#
# We first set an action directly without using an agent

# %%
# Sets the next action
action = torch.randint(0, action_dim, (n_envs, ))
workspace.set("action", 0, action)
print(action)
env_agent(workspace, t=1)

# And perform one step
workspace.get("env/env_obs", 1)

# %% [markdown]

# Let us now look at what's in the workspace. You can see below all the variables it generates.

# %%
for key in workspace.variables.keys():
    print(key, workspace[key])

# %% [markdown]

# You can observe that we have two time steps for each variable that are stored
# within tensors where the first dimension is time.

# You can also see that by convention, all variables written by the environment start with "env/".

# %% [markdown]

# ### Random agent

# The process above can be
# automatized with `Agents` and `TemporalAgent` as shown below - but first we have
# to create an agent that selects the actions (here, randomly).

# %%
class RandomAgent(Agent):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, t: int, choose_action=True, **kwargs):
        """An Agent can use self.workspace"""
        obs = self.get(("env/env_obs", t))
        action = torch.randint(0, self.action_dim, (len(obs), ))
        self.set(("action", t), action)

# Each agent is run in the order given when constructing Agents
agents = Agents(env_agent, RandomAgent(action_dim))

# And the TemporalAgent allows to run through time
t_agents = TemporalAgent(agents)

# %%
# We can now run the agents throught time with a simple call...

workspace = Workspace()
t_agents(workspace, t=0, stop_variable="env/done", stochastic=True)

# %% [markdown]

# ### Termination

# `env/done` tells us whether the episode was finished or not (it is either terminated or truncated)
# here, with NoAutoReset, we wait that all episodes are "done"
# and when the episode is finished, the variables are copied for that environment until all episodes are done.
# So, when an environment is done before the others, its content is copied until the termination of all environments.
# This is convenient for collecting the final reward.

# %%
workspace["env/done"].shape, workspace["env/done"][-10:]

# %% [markdown]

# You can see that the variable is copied until all episodes are done.

# %% [markdown]

# ### Observations

# The resulting tensor of observations, with the last two observations:

# %%
workspace["env/env_obs"].shape, workspace["env/env_obs"][-2:]

# %% [markdown]

# ### Rewards

# The resulting tensor of rewards, with the last 8 rewards:

# %%
workspace["env/reward"].shape, workspace["env/reward"][-8:]

# %% [markdown]

# and the cumulated rewards:

# %%
workspace["env/cumulated_reward"].shape, workspace["env/cumulated_reward"][-8:]

# %% [markdown]

# ### Actions

# The resulting tensor of actions, with the last two actions:

# %%
workspace["action"].shape, workspace["action"][-2:]

# %% [markdown]

# ## Exercise

# Create a stupid agent that always outputs action 1, until the episode stops.
# Watch the content of the resulting workspace.
