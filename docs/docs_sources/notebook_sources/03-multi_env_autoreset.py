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

# This notebook is designed to understand how to use a gymnasium environment as a BBRL agent in practice, using autoreset=True.
# It is part of the [BBRL documentation](https://github.com/osigaud/bbrl/docs/index.html).

# If this is your first contact with BBRL, you may start be having a look at [this more basic notebook](01-basic_concepts.student.ipynb) and [the one using autoreset=False](02-multi_env_noautoreset.student.ipynb).

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

# Replay buffers are useful to store past transitions when training
from bbrl.utils.replay_buffer import ReplayBuffer

# %% tags=["hide-input"]

# %% [markdown]

# ## Definition of agents

# We reuse the RandomAgent already used in the autoreset=False case.


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

# %% [markdown]

# As before, we create an Agent representing [the CartPole-v1 gym environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
# This is done using the [ParallelGymAgent](https://github.com/osigaud/bbrl/blob/40fe0468feb8998e62c3cd6bb3a575fef88e256f/src/bbrl/agents/gymnasium.py#L261) class.

# %% [markdown]

# ### Single environment case

# We start with a single instance of the CartPole environment

# %%
# We deal with 1 environment (random seed 2139)

env_agent = ParallelGymAgent(partial(make_env, env_name='CartPole-v1', autoreset=True), num_envs=1).seed(2139)
obs_size, action_dim = env_agent.get_obs_and_actions_sizes()
print(f"Environment: observation space in R^{obs_size} and action space R^{action_dim}")

# Each agent is run in the order given when constructing Agents

agents = Agents(env_agent, RandomAgent(action_dim))
t_agents = TemporalAgent(agents)

# %% [markdown]

# Let us have a closer look at the content of the workspace

# %%
# Creates a new workspace
workspace = Workspace() 
epoch_size = 6
t_agents(workspace, n_steps=epoch_size)

# We get the transitions: each tensor is transformed so that: 
# - we have the value at time step t and t+1 (so all the tensors first dimension have a size of 2)
# - there is no distinction between the different environments (here, there is just one environment to make it easy)
transitions = workspace.get_transitions()

display("Observations (first 4)", workspace["env/env_obs"][:4])

display("Transitions (first 3)")
for t in range(3):
    display(f'(s_{t}, s_{t+1})')
    # We ignore the first dimension as it corresponds to [t, t+1]
    display(transitions["env/env_obs"][:, t])

# %% [markdown]

# You can see that each transition in the workspace corresponds to a pair of observations.

# ### Transitions as a workspace

# A transition workspace is still a workspace... this is quite
#  handy since each transition can be seen as a mini-episode of two time steps;
#  we can use our agents on it.

# It is often the case in BBRL that we have to apply an agent to an already existing workspace
# as shown below.

# %%
for key in transitions.variables.keys():
    print(key, transitions[key])

t_random_agent = TemporalAgent(RandomAgent(action_dim))
t_random_agent(transitions, t=0, n_steps=2)

# Here, the action tensor will have been overwritten by the new actions
print(f"new action, {transitions['action']}")

# %% [markdown]

# ### Multiple environment case

# Now we are using 3 environments.
# Given the organization of transitions, to find the transitions of a particular environment
# we have to watch in the transition every 3 lines, since transitions are stored one environment after the other.

# %%
# We deal with 3 environments at a time (random seed 2139)

multienv_agent = ParallelGymAgent(partial(make_env, env_name='CartPole-v1', autoreset=True), num_envs=3).seed(2139)
obs_size, action_dim = multienv_agent.get_obs_and_actions_sizes()
print(f"Environment: observation space in R^{obs_size} and action space R^{action_dim}")

agents = Agents(multienv_agent, RandomAgent(action_dim))
t_agents = TemporalAgent(agents)
workspace = Workspace() 
t_agents(workspace, n_steps=epoch_size)
transitions = workspace.get_transitions()

display("Observations (first 4)", workspace["env/env_obs"][:4])

display("Transitions (first 3)")
for t in range(3):
    display(f'(s_{t}, s_{t+1})')
    display(transitions["env/env_obs"][:, t])
            
# %% [markdown]

# You can see how the transitions are organized in the workspace relative to the 3 environments.
# You first get the first transition from the first environment.
# Then the first transition from the second environment.
# Then the first transition from the third environment.
# Then the second transition from the first environment, etc.

# ## The replay buffer
# 
# Differently from the previous case, we use a replace buffer that stores
# a set of transitions $(s_t, a_t, r_t, s_{t+1})$
# Finally, the replay buffer keeps slices [:, i, ...] of the transition
# workspace (here at most 80 transitions)

# %%
rb = ReplayBuffer(max_size=80)

# We add the transitions to the buffer....
rb.put(transitions)

# And sample from them here we get 3 tuples (s_t, s_{t+1})
rb.get_shuffled(3)["env/env_obs"]

# %% [markdown]
# ## Collecting several epochs into the same workspace

# In the code below, the workspace only contains one epoch at a time.
# The content of these different epochs are concatenated into the replay buffer

# %%
nb_steps = 0
max_steps = 100
epoch_size = 10

while nb_steps < max_steps:
    # Execute the agent in the workspace
    if nb_steps == 0:
        # In the first epoch, we start with t=0
        t_agents(workspace, t=0, n_steps=epoch_size)
    else:
        # Clear all gradient graphs from the workspace
        workspace.zero_grad()
        # Here we duplicate the last column of the previous epoch into the first column of the next epoch
        workspace.copy_n_last_steps(1)

        # In subsequent epochs, we start with t=1 so as to avoid overwriting the first column we just duplicated
        t_agents(workspace, t=1, n_steps=epoch_size)

    transition_workspace = workspace.get_transitions()

    # The part below counts the number of steps: it ignores action performed during transition from one episode to the next,
    # as they have been discarded by the get_transitions() function

    action = transition_workspace["action"]
    nb_steps += action[0].shape[0]
    print(f"collecting new epoch, already performed {nb_steps} steps")

    if nb_steps > 0 or epoch_size  > 1:
        rb.put(transition_workspace)
    print(f"replay buffer size: {rb.size()}")

# %% [markdown]

# ## Exercise

# Create a stupid agent that always outputs action 1, run it for 10 epochs of 100 steps over 2 instances of the CartPole-v1 environment.
# Put the data into a replay buffer of size 5000.
#
# Then do the following:
# - Count the number of episodes the agent performed in each environment by counting the number of "done=True" elements in the workspace before applying the `get_transitions()` function
# - Count the total number of episodes performed by the agent by measuring the difference between the size of the replay buffer and the number of steps performed by the agent.
# - Make sure both counts are consistent
#
# Can we count the number of episodes performed in one environment using the second method? Why?
