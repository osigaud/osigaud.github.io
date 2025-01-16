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

# %% [markdown] id="dYfGJCe52lP4"
#
# # Outlook
#
# In this notebook we code the Soft Actor-Critic (SAC) algorithm using BBRL. This algorithm is described in [this paper](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf) and [this paper](https://arxiv.org/pdf/1812.05905.pdf).
# To understand this code, you need [to know more about BBRL](https://colab.research.google.com/drive/1_yp-JKkxh_P8Yhctulqm0IrLbE41oK1p?usp=sharing). You should first have a look at [the BBRL interaction model](https://colab.research.google.com/drive/1gSdkOBPkIQi_my9TtwJ-qWZQS0b2X7jt?usp=sharing), then [a first example](https://colab.research.google.com/drive/1Ui481r47fNHCQsQfKwdoNEVrEiqAEokh?usp=sharing) and, most importantly, [details about the AutoResetGymAgent](https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing).
#
# The algorithm is explained in [this video](https://www.youtube.com/watch?v=U20F-MvThjM) and you can also read [the corresponding slides](http://pages.isir.upmc.fr/~sigaud/teach/ps/12_sac.pdf).

# %% [markdown] id="zJZDcDafp7Uf"
"""
## Installation and Imports

### Installation
"""

# %% id="tPfvqqHyXSvj"

# !pip install easypip


# %% id="j0MaggiOl4KU"
from easypip import easyimport
import time

easyimport("importlib_metadata==4.13.0")
OmegaConf = easyimport("omegaconf").OmegaConf
bbrl_gym = easyimport("bbrl_gym")
bbrl = easyimport("bbrl>=0.1.6")

# %% tags=["teacher"]

import os
import bbrl
import bbrl_gym
from omegaconf import OmegaConf
testing_mode = os.environ.get("TESTING_MODE", None) == "ON"

# %% [markdown] id="m4kV9pWV3wRe"
"""
### Imports

Below, we import standard python packages, pytorch packages and gym environments.

This is OmegaConf that makes it possible that by just defining the `def run_a2c(cfg):` function and then executing a long `params = {...}` variable at the bottom of this colab, the code is run with the parameters without calling an explicit main.

More precisely, the code is run by calling

```py
config=OmegaConf.create(params)
run_a2c(config)
```

at the very bottom of the notebook, after starting tensorboard.

[OpenAI gym](https://gym.openai.com/) is a collection of benchmark environments to evaluate RL algorithms.
"""

# %% id="vktQB-AO5biu"
import os
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

# %% [markdown] id="uWBCaTTKZKCs"
"""
### BBRL imports
"""

# %% id="7oERG6YRZSvx"
from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class

# The workspace is the main class in BBRL, this is where all data is collected and stored
from bbrl.workspace import Workspace

# Agents(agent1,agent2,agent3,...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent over multiple timesteps in the workspace, 
# or until a given condition is reached
from bbrl.agents import Agents, RemoteAgent, TemporalAgent

# AutoResetGymAgent is an agent able to execute a batch of gym environments
# with auto-resetting. These agents produce multiple variables in the workspace: 
# ’env/env_obs’, ’env/reward’, ’env/timestep’, ’env/done’, ’env/initial_state’, ’env/cumulated_reward’, 
# ... When called at timestep t=0, then the environments are automatically reset. 
# At timestep t>0, these agents will read the ’action’ variable in the workspace at time t − 1
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent

from bbrl.utils.replay_buffer import ReplayBuffer

# %% [markdown] id="JVvAfhKm9S8p"
"""
## Definition of agents

### Functions to build networks

We use the same utility functions to build neural networks as usual.
"""


# %% id="HFLn1t5rmIDb"
def build_backbone(sizes, activation):
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation]
    return layers


def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)



# %% [markdown] id="ouwQ5WhxTKNV"
"""
### The SquashedGaussianActor

SAC works better with a Squashed Gaussian policy, which enables the reparametrization trick. Note that our attempts to use a `TunableVarianceContinuousActor` as we did for instance in the [notebook about PPO](https://colab.research.google.com/drive/1KTxeRA3e0Npxa8Fa9y1OMcJCeQa41o_N?usp=sharing) completely failed. Such failure is also documented in the [OpenAI spinning up documentation page about SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html).

The code of the `SquashedGaussianActor` policy is below.

It relies on a specific type of distribution, the `SquashedDiagGaussianDistribution` which is taken from [the Stable Baselines3 library](https://github.com/DLR-RM/stable-baselines3).

The fact that we use the reparametrization trick is hidden inside the code of this distribution. In more details, the key is that the [`sample(self)` method](https://github.com/osigaud/bbrl/blob/5c2b42c2ee30077166f86cc1dd562a3dce6203db/bbrl/utils/distributions.py#L200) calls `rsample()`.
"""

# %% [markdown] id="WCtEKSP19WEz"
"""
If you want to try using SAC with a squashed Gaussian policy but without using the reparametrization trick, you have to rewrite your own class to deal with a squashed Gaussian distribution.
"""

# %% id="5LO_VNaOTeJu"
from bbrl.utils.distributions import SquashedDiagGaussianDistribution

class SquashedGaussianActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        backbone_dim = [state_dim] + list(hidden_layers)
        self.layers = build_backbone(backbone_dim, activation=nn.ReLU())
        self.backbone = nn.Sequential(*self.layers)
        self.last_mean_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.last_std_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        # std must be positive
        self.std_layer = nn.Softplus()

    def dist(self, obs: torch.Tensor):
        """Computes action distributions given observation(s)"""
        backbone_output = self.backbone(obs)
        mean = self.last_mean_layer(backbone_output)
        std_out = self.last_std_layer(backbone_output)
        std = self.std_layer(std_out)
        return self.action_dist.make_distribution(mean, std)


    def forward(self, t, stochastic):
        action_dist = self.dist(self.get(("env/env_obs", t)))
        action = action_dist.sample() if stochastic else action_dist.mode()

        log_prob = action_dist.log_prob(action)
        self.set((f"action", t), action)
        self.set(("action_logprobs", t), log_prob)

    def predict_action(self, obs, stochastic: bool):
        action_dist = self.dist(obs)
        action = action_dist.sample() if stochastic else action_dist.mode()
        return action


# %% [markdown] id="ajqSi5Nbmnxn"

# ### Choosing a specific gym environment
# First, we need to make our gym environment. As usual, this is implemented with the simple function below.


# %% id="Fsb5QRzw7V0o"
def make_gym_env(env_name):
    return gym.make(env_name)


# %% [markdown] id="Din6iU-1DnyH"
# ### CriticAgent

# As critics and target critics, SAC uses several instances of ContinuousQAgent class, as DDPG and TD3. See the [DDPG notebook](https://colab.research.google.com/drive/1APBtDiaFwQHKE2rfTZioGfDM8C41e7Il?usp=sharing) for details.


# %% id="g8y-63nq7Pjo"
class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t, detach_actions=False):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_actions:
            action = action.detach()
        osb_act = torch.cat((obs, action), dim=1)
        q_value = self.model(osb_act)
        self.set(("q_value", t), q_value)

    def predict_value(self, obs, action):
        osb_act = torch.cat((obs, action), dim=0)
        q_value = self.model(osb_act)
        return q_value



# %% [markdown] id="L9BPA7Kht6DL"
# ### Building the complete training and evaluation agents
 
# In the code below we create the Squashed Gaussian actor, two critics and the corresponding target critics. Beforehand, we checked that the environment takes continuous actions (otherwise we would need a different code).


# %% id="UpiApKBfuBCS"
# Create the SAC Agent
def create_sac_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    assert (
        train_env_agent.is_continuous_action()
    ), "SAC code dedicated to continuous actions"

    # Actor
    actor = SquashedGaussianActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )

    # Train/Test agents
    tr_agent = Agents(train_env_agent, actor)
    ev_agent = Agents(eval_env_agent, actor)

    # Builds the critics
    critic_1 = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
    )
    target_critic_1 = copy.deepcopy(critic_1)
    critic_2 = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
    )
    target_critic_2 = copy.deepcopy(critic_2)

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    )



# %% [markdown] id="lU3cO6znHyDc"
"""
### The Logger class
"""

# %% [markdown] id="E4BrXwTLdK0Z"
# The logger class is the same as before, see [this notebook](https://colab.research.google.com/drive/1yAQlrShysj4Q9EBpYM8pBsp2aXInhP7x#scrollTo=lU3cO6znHyDc) for explanations.


# %% id="aOkauz_0H2GA"
class Logger():

  def __init__(self, cfg):
    self.logger = instantiate_class(cfg.logger)

  def add_log(self, log_string, loss, epoch):
    self.logger.add_scalar(log_string, loss.item(), epoch)

  # Log losses
  def log_losses(self, epoch, critic_loss, entropy_loss, actor_loss):
    self.add_log("critic_loss", critic_loss, epoch)
    self.add_log("entropy_loss", entropy_loss, epoch)
    self.add_log("actor_loss", actor_loss, epoch)



# %% [markdown] id="f2vq1OJHWCIE"
# ### Setup the optimizers

# A specificity of SAC is that it can optimize the entropy coefficient named
# $\alpha$. How to tune $\alpha$ is explained in [this
# paper](https://arxiv.org/pdf/1812.05905.pdf).

# Thus we have two functions to set up optimizers, one which deals with the
# actor and the critic as usual, and one which deals with the entropy
# coefficient. We use a single optimizer to tune the parameters of the actor and
# the critic. It would be possible to have two optimizers which would work
# separately on the parameters of each component agent, but it would be more
# complicated because updating the actor requires the gradient of the critic.



# %% id="YFfzXEu2WFWj"
# Configure the optimizer for the actor and critic
def setup_optimizers(cfg, actor, critic_1, critic_2):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = nn.Sequential(critic_1, critic_2).parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer


# %% [markdown] id="uM04csjWBTSB" 

# For the entropy coefficient optimizer, the code is as follows. Note the trick
# which consists in using the log of this entropy coefficient. This trick was
# taken from the Stable baselines3 implementation of SAC, which is explained in
# [this notebook](https://colab.research.google.com/drive/12LER1_ShWOa_UhOL1nlX-LX_t5KQK9LV?usp=sharing).

# Tuning $\alpha$ in SAC is an option. To chose to tune it, the `target_entropy`
# argument in the parameters should be `auto`. The initial value is given
# through the `entropy_coef` parameter. For any other value than `auto`, the
# value of $\alpha$ will stay constant and correspond to the `entropy_coef`
# parameter.


# %% id="Fr6hdgOQ1ODv"
def setup_entropy_optimizers(cfg):
    if cfg.algorithm.target_entropy == "auto":
        entropy_coef_optimizer_args = get_arguments(cfg.entropy_coef_optimizer)
        # Note: we optimize the log of the entropy coefficient which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        # Comment and code taken from the SB3 version of SAC
        log_entropy_coef = torch.log(
            torch.ones(1) * cfg.algorithm.entropy_coef
        ).requires_grad_(True)
        entropy_coef_optimizer = get_class(cfg.entropy_coef_optimizer)(
            [log_entropy_coef], **entropy_coef_optimizer_args
        )
    else:
        log_entropy_coef = 0
        entropy_coef_optimizer = None
    return entropy_coef_optimizer, log_entropy_coef



# %% [markdown] id="YQNvhO_VAJbh"

# ### Compute the critic loss

# With the notations of my slides, the equation corresponding to Eq. (5) and (6)
# in [this paper](https://arxiv.org/pdf/1812.05905.pdf) becomes:

# $$ loss_Q({\boldsymbol{\theta}}) = {\rm I\!E}_{(\mathbf{s}_t, \mathbf{a}_t,
# \mathbf{s}_{t+1}) \sim \mathcal{D}}\left[\left( r(\mathbf{s}_t, \mathbf{a}_t)
# + \gamma {\rm I\!E}_{\mathbf{a} \sim
# \pi_{\boldsymbol{\theta}}(.|\mathbf{s}_{t+1})}\left[\hat{Q}^{\pi_{\boldsymbol{\theta}}}_{\boldsymbol{\phi}}(\mathbf{s}_{t+1},
# \mathbf{a}) - \alpha
# \log{\pi_{\boldsymbol{\theta}}(\mathbf{a}|\mathbf{s}_{t+1})} \right] -
# \hat{Q}^{\pi_{\boldsymbol{\theta}}}_{\boldsymbol{\phi}}(\mathbf{s}_t,
# \mathbf{a}_t) \right)^2 \right] $$

# An important information in the above equation and the one about the actor
# loss below is the index of the expectations. These indexes tell us where the
# data should be taken from. In the above equation, one can see that the index
# of the outer expectation is over samples taken from the replay buffer, whereas
# in the inner expectation we consider actions from the current policy at the
# next state.

# Thus, to compute the inner expectation, one needs to determine what actions
# the current policy would take in the next state of each sample. This is what
# the line 

# `t_actor(rb_workspace, t=1, n_steps=1, stochastic=True)`

# does. The parameter `t=1` (instead of 0) ensures that we consider the next state.

# Once we have determined these actions, we can determine their Q-values and
# their log probabilities, to compute the inner expectation.

# Note that at this stage, we only determine the log probabilities corresponding
# to actions taken at the next time step, by contrast with what we do for the
# actor in the `compute_actor_loss(...)` function later on.

# Finally, once we have computed the $$
# \hat{Q}^{\pi_{\boldsymbol{\theta}}}_{\boldsymbol{\phi}}(\mathbf{s}_{t+1},
# \mathbf{a}) $$ for both critics, we take the min and store it into
# `post_q_values`. By contrast, the Q-values corresponding to the last term of
# the equation are taken from the replay buffer, they are computed in the
# beginning of the function by applying the Q agents to the replay buffer
# *before* changing the action to that of the current policy.

# An important remark is that, if the entropy coefficient $\alpha$ corresponding
# to the `ent_coef` variable is set to 0, then we retrieve exactly the critic
# loss computation function of the TD3 algorithm. As we will see later, this is
# also true of the actor loss computation.

# This remark proved very useful in debugging the SAC code. We have set
# `ent_coef` to 0 and ensured the behavior was strictly the same as the behavior
# of TD3.


# %% id="N9KRFG-PBRtD"
def compute_critic_loss(
    cfg, reward, must_bootstrap,
    t_actor, 
    q_agent_1, q_agent_2, 
    target_q_agent_1, target_q_agent_2, 
    rb_workspace,
    ent_coef
):
    """Computes the critic loss for a set of $S$ transition samples

    Args:
        cfg: The experimental configuration
        reward: _description_
        must_bootstrap: Tensor of indicators (2 x S)
        t_actor: The actor agent (as a TemporalAgent)
        q_agent_1: The first critic (as a TemporalAgent)
        q_agent_2: The second critic (as a TemporalAgent)
        target_q_agent_1: The target of the first critic
        target_q_agent_2: The target of the second critic
        rb_workspace: The transition workspace
        ent_coef: The entropy coefficient

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The two critic losses (scalars)
    """
    # Compute q_values from both critics with the actions present in the buffer:
    # at t, we have Q(s,a) from the (s,a) in the RB
    q_agent_1(rb_workspace, t=0, n_steps=1)
    q_values_rb_1 = rb_workspace["q_value"]
    
    q_agent_2(rb_workspace, t=0, n_steps=1)
    q_values_rb_2 = rb_workspace["q_value"]

    with torch.no_grad():
        # Replay the current actor on the replay buffer to get actions of the
        # current policy
        t_actor(rb_workspace, t=1, n_steps=1, stochastic=True)
        action_logprobs_next = rb_workspace["action_logprobs"]

        # Compute target q_values from both target critics: at t+1, we have
        # Q(s+1,a+1) from the (s+1,a+1) where a+1 has been replaced in the RB

        target_q_agent_1(rb_workspace, t=1, n_steps=1)
        post_q_values_1 = rb_workspace["q_value"]

        target_q_agent_2(rb_workspace, t=1, n_steps=1)
        post_q_values_2 = rb_workspace["q_value"]

    # [[student]] Compute temporal difference

    q_next = torch.min(post_q_values_1[1], post_q_values_2[1]).squeeze(-1)
    v_phi = q_next - ent_coef * action_logprobs_next

    target = (
        reward[-1] + cfg.algorithm.discount_factor * v_phi * must_bootstrap.int()
    )
    td_1 = target - q_values_rb_1[0].squeeze(-1)
    td_2 = target - q_values_rb_2[0].squeeze(-1)
    td_error_1 = td_1**2
    td_error_2 = td_2**2
    critic_loss_1 = td_error_1.mean()
    critic_loss_2 = td_error_2.mean()
    # [[/student]]

    return critic_loss_1, critic_loss_2


# %% [markdown] id="RaRH4rg-HZb5"
# As in DDPG and TD3, we use target critics, thus we need the
# `soft_update_params(...)` function to make sure that the target critics are
# tracking the true critics, using the same equation: $\theta' \leftarrow \tau
# \theta + (1- \tau) \theta'$.


# %% id="gAIbEbNGIdb_"
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# %% [markdown] id="G_OzHrYqYOkR"
# ### Compute the actor Loss


# With the notations of my slides, the equation of the actor loss corresponding
# to Eq. (7) in [this paper](https://arxiv.org/pdf/1812.05905.pdf) becomes:

# $$ loss_\pi({\boldsymbol{\theta}}) = {\rm I\!E}_{\mathbf{s}_t \sim
# \mathcal{D}}\left[ {\rm I\!E}_{\mathbf{a}_t\sim
# \pi_{\boldsymbol{\theta}}(.|\mathbf{s}_t)} \left[ \alpha
# \log{\pi_{\boldsymbol{\theta}}(\mathbf{a}_t|\mathbf{s}_t) -
# \hat{Q}^{\pi_{\boldsymbol{\theta}}}_{\boldsymbol{\phi}}(\mathbf{s}_t,
# \mathbf{a}_t)} \right] \right] $$

# Note that [the paper](https://arxiv.org/pdf/1812.05905.pdf) mistakenly writes
# $Q_\theta(s_t,s_t)$

# As for the critic loss, we have two expectations, one over the states from the
# replay buffer, and one over the actions of the current policy. Thus we need to
# apply again the current policy to the content of the replay buffer.

# But this time, we consider the current state, thus we parametrize it with
# `t=0` and `n_steps=1`. This way, we get the log probabilities and Q-values at
# the current step.

# A nice thing is that this way, there is no overlap between the log probability
# data used to update the critic and the actor, which avoids having to 'retain'
# the computation graph so that it can be reused for the actor and the critic.

# This small trick is one of the features that makes coding SAC the most
# difficult.

# Again, once we have computed the Q values over both critics, we take the min
# and put it into `current_q_values`.

# As for the critic loss, if we set `ent_coef` to 0, we retrieve the actor loss
# function of DDPG and TD3, which simply tries to get actions that maximize the
# Q values (by minimizing -Q).

# %% id="xLq_ZeFzEHON"
def compute_actor_loss(ent_coef, t_actor, q_agent_1, q_agent_2, rb_workspace):
    """Actor loss computation
    
    :param ent_coef: The entropy coefficient $\alpha$
    :param t_actor: The actor agent (temporal agent)
    :param q_agent_1: The first critic (temporal agent)
    :param q_agent_2: The second critic (temporal agent)
    :param rb_workspace: The replay buffer (2 time steps, $t$ and $t+1$)
    """
    # Recompute the q_values from the current policy, not from the actions in the buffer

    # [[student]] Recompute the action with the current policy (at $a_t$)
    t_actor(rb_workspace, t=0, n_steps=1, stochastic=True)
    # [[/student]]
    action_logprobs_new = rb_workspace["action_logprobs"]

    # [[student]] Compute Q-values
    q_agent_1(rb_workspace, t=0, n_steps=1)
    q_values_1 = rb_workspace["q_value"]

    q_agent_2(rb_workspace, t=0, n_steps=1)
    q_values_2 = rb_workspace["q_value"]
    # [[/student]]

    current_q_values = torch.min(q_values_1, q_values_2).squeeze(-1)

    # [[student]] Compute the actor loss
    actor_loss = ent_coef * action_logprobs_new[0] - current_q_values[0]
    # [[/student]]

    return actor_loss.mean()


# %% [markdown] id="Jmi91gANWT4z"

# ## Main training loop


# %% id="sk85_sRWW-5s"

def run_sac(cfg):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9
    ent_coef = cfg.algorithm.entropy_coef

    # 2) Create the environment agent
    train_env_agent = AutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )
    eval_env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.nb_evals,
        cfg.algorithm.seed,
    )

    # 3) Create the A2C Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    ) = create_sac_agent(cfg, train_env_agent, eval_env_agent)

    t_actor = TemporalAgent(actor)
    q_agent_1 = TemporalAgent(critic_1)
    target_q_agent_1 = TemporalAgent(target_critic_1)
    q_agent_2 = TemporalAgent(critic_2)
    target_q_agent_2 = TemporalAgent(target_critic_2)
    train_workspace = Workspace()

    # Creates a replay buffer
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic_1, critic_2)
    entropy_coef_optimizer, log_entropy_coef = setup_entropy_optimizers(cfg)
    nb_steps = 0
    tmp_steps = 0

    # Initial value of the entropy coef alpha. If target_entropy is not auto,
    # will remain fixed
    if cfg.algorithm.target_entropy == "auto":
        target_entropy = -np.prod(train_env_agent.action_space.shape).astype(np.float32)
    else:
        target_entropy = cfg.algorithm.target_entropy

    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps - 1,
                stochastic=True,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps,
                stochastic=True,
            )

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)

        if nb_steps > cfg.algorithm.learning_starts:
            # Get a sample from the workspace
            rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

            done, truncated, reward, action_logprobs_rb = rb_workspace[
                "env/done", "env/truncated", "env/reward", "action_logprobs"
            ]

            # Determines whether values of the critic should be propagated
            # True if the episode reached a time limit or if the task was not done
            # See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing
            must_bootstrap = torch.logical_or(~done[1], truncated[1])

            (
                critic_loss_1, critic_loss_2
            ) = compute_critic_loss(
                cfg, 
                reward, 
                must_bootstrap,
                t_actor,
                q_agent_1,
                q_agent_2,
                target_q_agent_1,
                target_q_agent_2,
                rb_workspace,
                ent_coef
            )

            logger.add_log("critic_loss_1", critic_loss_1, nb_steps)
            logger.add_log("critic_loss_2", critic_loss_2, nb_steps)
            critic_loss = critic_loss_1 + critic_loss_2

            actor_loss = compute_actor_loss(
                ent_coef, t_actor, q_agent_1, q_agent_2, rb_workspace
            )
            logger.add_log("actor_loss", actor_loss, nb_steps)

            # Entropy coef update part #####################################################
            if entropy_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so that we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = torch.exp(log_entropy_coef.detach())
                # See Eq. (17) of the SAC and Applications paper
                entropy_coef_loss = -(
                    log_entropy_coef * (action_logprobs_rb + target_entropy)
                ).mean()
                entropy_coef_optimizer.zero_grad()
                # We need to retain the graph because we reuse the
                # action_logprobs are used to compute both the actor loss and
                # the critic loss
                entropy_coef_loss.backward(retain_graph=True)
                entropy_coef_optimizer.step()
                logger.add_log("entropy_coef_loss", entropy_coef_loss, nb_steps)
                logger.add_log("entropy_coef", ent_coef, nb_steps)

            # Actor update part ###############################
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), cfg.algorithm.max_grad_norm
            )
            actor_optimizer.step()


            # Critic update part ###############################
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic_1.parameters(), cfg.algorithm.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                critic_2.parameters(), cfg.algorithm.max_grad_norm
            )
            critic_optimizer.step()
            ####################################################

            # Soft update of target q function
            tau = cfg.algorithm.tau_target
            soft_update_params(critic_1, target_critic_1, tau)
            soft_update_params(critic_2, target_critic_2, tau)
            # soft_update_params(actor, target_actor, tau)

        # Evaluate ###########################################
        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                stochastic=False,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward/mean", mean, nb_steps)
            logger.add_log("reward/max", rewards.max(), nb_steps)
            logger.add_log("reward/min", rewards.min(), nb_steps)
            logger.add_log("reward/min", rewards.median(), nb_steps)

            print(f"nb_steps: {nb_steps}, reward: {mean}")
            # print("ent_coef", ent_coef)
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = f"./agents/{cfg.gym_env.env_name}/sac_agent/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "sac_" + str(mean.item()) + ".agt"
                actor.save_model(filename)
                


# %% [markdown] id="uo6bc3zzKua_"
# ## Definition of the parameters

# %% id="JB2B8zELNWQd"

params={
  "save_best": True,
  "logger":{
    "classname": "bbrl.utils.logger.TFLogger",
    "log_dir": "./tblogs/CartPoleContinuous-v1/sac-" + str(time.time()),
    "cache_size": 10000,
    "every_n_seconds": 10,
    "verbose": False,    
    },

  "algorithm":{
    "seed": 1,
    "n_envs": 8,
    "n_steps": 32,
    "buffer_size": 1e6,
    "batch_size": 256,
    "max_grad_norm": 0.5,
    "nb_evals":10,
    "eval_interval": 2000,
    "learning_starts": 10000,
    "max_epochs": 8000,
    "discount_factor": 0.98,
    "entropy_coef": 1e-7,
    "target_entropy": "auto",
    "tau_target": 0.05,
# [[remove]]
    # Reduces learning
    "eval_interval": 10 if testing_mode else 1000,
    "max_epochs": 30 if testing_mode else 3000,
# [[/remove]]
    "architecture":{
      "actor_hidden_size": [32, 32],
      "critic_hidden_size": [256, 256],
    },
  },
  "gym_env":{
    "classname": "__main__.make_gym_env",
    "env_name": "CartPoleContinuous-v1",
    },
  "actor_optimizer":{
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
    },
  "critic_optimizer":{
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
    },
  "entropy_coef_optimizer":{
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
    }
}

# %% [markdown] id="jp7jDeGkaoM1"
"""
### Launching tensorboard to visualize the results
"""

# %% id="5VxSP8npnU2Q" tags=["colab"]
# %load_ext tensorboard
# %tensorboard --logdir ./tmp

# %% tags=["not-colab"]
import sys
import os
import os.path as osp
print(f'''Launch tensorboard from the shell:\n{osp.dirname(sys.executable)}/tensorboard --logdir="{params["logger"]["log_dir"]}"''')

# %% id="l42OUoGROlSt"
config=OmegaConf.create(params)
torch.manual_seed(config.algorithm.seed)
run_sac(config)

# %% [markdown] id="paHdoNlz9Lpg"
# Now we can look at the agent

# %%

from bbrl.visu.play import play, load_agent, Path
agent = load_agent(Path(f"agents/{config.gym_env.env_name}/sac_agent"), "sac_")

def play(env: gym.Env, agent: torch.nn.Module):
    """Render the agent"""
    if agent is None:
        print("No agent")
        return

    sum_reward = 0.
    
    try:
        print(agent)
        with torch.no_grad():
            obs = env.reset()
            env.render()
            done = False
            while not done:
                obs = torch.Tensor(obs)
                action = agent.predict_action(obs, False)
                obs, reward, done, info = env.step(action.numpy())
                sum_reward += reward
                env.render()
    finally:
        env.close()

    return reward

play(make_gym_env(config.gym_env.env_name), agent)

# %% [markdown] id="paHdoNlz9Lpg"
# ## Exercises

# - use the same code on the Pendulum-v1 environment. This one is harder to
#   tune. Get the parameters from the
#   [rl-baseline3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) and see if
#   you manage to get SAC working on Pendulum

# %%
