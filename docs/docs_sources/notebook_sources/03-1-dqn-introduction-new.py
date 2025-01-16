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
# In this notebook, using BBRL, we code a simple version of the DQN algorithm
# without a replay buffer nor a target network so as to better understand the
# inner mechanisms.

# To understand this code, you need to know more about 
# [the BBRL interaction model](https://colab.research.google.com/drive/1_yp-JKkxh_P8Yhctulqm0IrLbE41oK1p?usp=sharing).
# Then you should run [a first example](https://colab.research.google.com/drive/1Ui481r47fNHCQsQfKwdoNEVrEiqAEokh?usp=sharing)
# to see how agents interact.
#
# The DQN algorithm is explained in [this
# video](https://www.youtube.com/watch?v=CXwvOMJujZk) and you can also read [the
# corresponding slides](http://pages.isir.upmc.fr/~sigaud/teach/dqn.pdf).
#

# %% tags=["copy"]
from _bbrl_environment import *

# %% [markdown]

# ## Definition of agents
#
# After running the simple BBRL interaction notebook pointed to above, have a
# look at [this more advanced
# notebook](https://colab.research.google.com/drive/1Ui481r47fNHCQsQfKwdoNEVrEiqAEokh?usp=sharing)
# where the agents is a simple random agent and the environment agent is [the
# CartPole-v1 gym
# environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
# We provide just a few details below.
#
# In BBRL, the agent and the environment act within a **workspace** by *reading*
# and *writing* information:
#
# - **reading**: by using `self.get((key, t))` to get the value of the tensor
#   identified by `key` (a string)
# - **writing**: by using `self.set((key, t), y)` to set the value of the tensor
#   identified by `key` to `y`
#
# **WARNING**: we are working with batches (i.e. several episodes at the same
# time)
#
# An episode is depicted in the figure below.

# %% [markdown] 
# 
# ![workspace.png](./images/bbrl_workspace.png)
#
# Note that, as depicted below, the reward indexing scheme we use here consists in getting $r_{t+1}$ after performing action $a_t$ from state $s_t$.
#
# ![data_collection_bbrl.png](./images/data_collection_bbrl.png)
#
# [This notebook](https://colab.research.google.com/drive/1Cld72_FBA1aMS2U4EsyV3LGZIlQC_PsC?usp=sharing#scrollTo=qXvH0jNADUsY) explains that another choice is possible.
#
# To showcase BBRL, let us go through the CartPole environment


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

# ### Random action without agent
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

# Let us now see the workspace

# %%
for key in workspace.variables.keys():
    print(key, workspace[key])

# %% [markdown]
# You can observe that we have two time steps for each variable that are stored
# within tensors where the first dimension is time. 

# %% [markdown]

# ### Random agent

# The process above can be
# automatized with `Agents` and `TemporalAgent` as shown below - but first we have
# to create an agent that selects the actions (here, random).

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

# `env/done` tells us if the episode was finished or not
# here, with NoAutoReset, (1) we wait that all episodes are "done",
# and when an episode is finished the flag remains True.
# Note that when an environment is done before the others, its content is copied until the termination of all environments.
# This is convenient for collecting the final reward.

# %%
workspace["env/done"].shape, workspace["env/done"][-10:]

# %% [markdown]

# The resulting tensor of observations, with the last two observations

# %%
workspace["env/env_obs"].shape, workspace["env/env_obs"][-2:]

# %% [markdown]

# The resulting tensor of rewards, with the last 8 rewards

# %%
workspace["env/reward"].shape, workspace["env/reward"][-8:]

# %% [markdown]

# The resulting tensor of actions, with the last two actions

# %%
workspace["action"].shape, workspace["action"][-2:]

# %% [markdown]
# Some other details are available in the notebooks pointed above.

# %% [markdown]

# ## Definition of agents

# ### The critic agent
# 
# The [DQN](https://daiwk.github.io/assets/dqn.pdf) algorithm is a critic only
# algorithm. Thus we just need a Critic agent (which will also be used to output
# actions) and an Environment agent. We use the `DiscreteQAgent` class also explained in [this
# notebook](https://colab.research.google.com/drive/1Ui481r47fNHCQsQfKwdoNEVrEiqAEokh?usp=sharing).

# %% tags=["copy"]
from _mlp import *

# %% [markdown]

# The `DiscreteQAgent` class implements a critic such as the one used in DQN.
# It has one output neuron per action and its output is the Q-value of these actions given the state.
# As any BBRL agent, it has a `forward()` function that takes a time state as input.
# This `forward()` function outputs the Q-values at the corresponding time step.

# %%
class DiscreteQAgent(Agent):
    """BBRL agent (discrete actions) based on a MLP"""
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t: int, **kwargs):
        """An Agent can use self.workspace"""

        # Retrieves the observation from the environment at time t
        obs = self.get(("env/env_obs", t))

        # Computes the critic (Q) values for the observation
        q_values = self.model(obs)

        # ... and sets the q-values (one for each possible action)
        self.set(("q_values", t), q_values)


class ArgmaxActionSelector(Agent):
    """BBRL agent that selects the best action based on Q(s,a)"""
    def forward(self, t: int, **kwargs):
        q_values = self.get(("q_values", t))
        action = q_values.argmax(1)
        self.set(("action", t), action)

# %% [markdown]

# ### Creating an Exploration method
# 
# As Q-learning, DQN needs some exploration to prevent too early convergence.
# Here we use the simple $\epsilon$-greedy exploration method. The method
# is implemented as an agent which chooses an action based on the Q-values.


# %% 
class EGreedyActionSelector(Agent):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, t: int, **kwargs):
        # Retrieves the q values 
        # (matrix nb. of episodes x nb. of actions)
        q_values = self.get(("q_values", t))
        size, nb_actions = q_values.size()

        # Flag 
        is_random = torch.rand(size).lt(self.epsilon).float()
        random_action = torch.randint(low=0, high=nb_actions, size=(size,))
        max_action = q_values.max(1)[1]

        # Choose the action based on the is_random flag
        action = is_random * random_action + (1 - is_random) * max_action

        # Sets the action at time t
        self.set(("action", t), action.long())

# %% tags=["copy"]
from _logger import *

# %% [markdown]

# ## Heart of the algorithm
# ### Computing the critic loss
# The role of the `compute_critic_loss` function is to implement the Bellman
# backup rule. In Q-learning, this rule was written:
#
# $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [ r(s_t,a_t) + \gamma \max_a
# Q(s_{t+1},a) - Q(s_t,a_t)]$$
#
# In DQN, the update rule $Q \leftarrow Q + \alpha [\delta] $ is replaced by a
# gradient descent step over the Q-network. 
#
# We first compute a target value: $ target = r(s_t,a_t) + \gamma \max_a
# Q(s_{t+1},a)$ from a set of samples.
#
# Then we get a TD error $\delta$ by substracting $Q(s_t,a_t)$ for these samples, 
# and we use the squared TD error as a loss function: $ loss = (target -
# Q(s_t,a_t))^2$.
# 
# To implement the above calculation in BBRL, the difficulty is to
# properly deal with time indexes.
#
# The `compute_critic_loss` function receives rewards, q_values and actions as
# vectors (in practice, pytorch tensors) that have been computed over a complete
# episode.
# 
# We need to take `reward[1:]`, which means all the rewards except the first one,
# as explained in [this
# notebook](https://colab.research.google.com/drive/1Cld72_FBA1aMS2U4EsyV3LGZIlQC_PsC?usp=sharing).
# Similarly, to get $\max_a Q(s_{t+1}, a)$, we need to ignore the first of the
# max_q values, using `max_q[1:]`.
# 
# Note the `max_q[0].detach()` in the computation of the temporal difference
# target. First, the max_q[0] is because the max function returns both the max
# and the indexes of the max. Second, about the .detach(), the idea is that we
# compute this target as a function of $\max_a Q(s_{t+1}, a)$, but **we do not
# want to apply gradient descent on this $\max_a Q(s_{t+1}, a)$**, we only
# apply gradient descent to $Q(s_t, a_t)$ according to this target value. In
# practice, `x.detach()` detaches a computation graph from a tensor, so it
# avoids computing a gradient over this tensor.
# 
# The `must_bootstrap` tensor is used as a trick to deal with terminal states.
# If the state is terminal, $Q(s_{t+1}, a)$ does not make sense. Thus we need to
# ignore this term. So we multiply the term by `must_bootstrap`: if
# `must_bootstrap` is True (converted into a float, it becomes a 1), we get the
# term. If `must_bootstrap` is False (=0), we are at a terminal state, so we
# ignore the term. This trick is used in many RL libraries, e.g. SB3. In [this
# notebook](https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing)
# we explain how to compute `must_bootstrap` so as to properly deal with time
# limits. In this version we use full episodes, thus `must_bootstrap` will
# always be True for all steps but the last one.
# 
# To compute $Q(s_t,a_t)$ we use the `torch.gather()` function. This function is
# a little tricky to use, see [this
# page](https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4)
# for useful explanations.
# 
# In particular, the q_vals output that we get is not properly conditioned,
# hence the need for the `qval[:-1]` (we ignore the last dimension). Finally we
# just need to compute the difference target - qvals, square it, take the mean
# and send it back as the loss.

def compute_critic_loss(cfg, reward: torch.Tensor, must_bootstrap: torch.Tensor, q_values: torch.Tensor, action: torch.LongTensor) -> torch.Tensor:
    """Compute the temporal difference loss from a dataset to 
    update a critic

    For the tensor dimensions:
    
    - T = maximum number of time steps
    - B = number of episodes run in parallel 
    - A = state space dimension

    :param cfg: The configuration
    :param reward: A (T x B) tensor containing the rewards 
    :param must_bootstrap: a (T x B) tensor containing 0 if the episode is
        completed at time $t$ 
    :param q_values: a (T x B x A) tensor containing the Q-values at each
        time step
    :param action: a (T x B) long tensor containing the chosen action

    :return: The DQN loss
    """
    # We compute the max of Q-values over all actions and detach (so that
    # this part of the computation graph is not included in the gradient
    # backpropagation)

    # [[student]] Calculer la loss
    max_q = q_values.max(2)[0].detach()  # Results in a (T x B) tensor

    # To get the max of Q(s_{t+1}, a), we take max_q[1:]
    # The same about must_bootstrap. 
    target = (
        reward[1:] + cfg.algorithm.discount_factor * max_q[1:] * must_bootstrap[1:].int()
    )

    # To get Q(s,a), we use torch.gather along the 3rd dimension (the action)
    qvals = q_values.gather(2, action.unsqueeze(-1)).squeeze(-1)

    # Compute the temporal difference (use must_boostrap as to mask out finished episodes)
    td = (target - qvals[:-1]) * must_bootstrap[:-1].int()

    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    # [[/student]]

    return critic_loss


# %% [markdown]

# ## Main training loop
#
# Note that everything about the shared workspace between all the agents is
# completely hidden under the hood. This results in a gain of productivity, at
# the expense of having to dig into the BBRL code if you want to understand the
# details, change the multiprocessing model, etc.
#
# The next cell defines a `EpisodicDQN` that deals with various part of the training
# loop:
#
# - `__init__` takes care of initializing the train and evaluation policies

# %% tags=["copy"]

from _bbrl_episodic import *

# %% 
class EpisodicDQN(RLEpisodic):
    def __init__(self, cfg):
        super().__init__(cfg)
            
        # Get the observation / action state space dimensions
        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        # Our discrete Q-Agent
        self.q_agent = DiscreteQAgent(obs_size, cfg.algorithm.architecture.hidden_size, act_size)

        # The e-greedy strategy (when training)
        explorer = EGreedyActionSelector(cfg.algorithm.epsilon)

        # The training agent combines the Q agent
        self.train_policy = Agents(self.q_agent, explorer)

        # The optimizer for the Q-Agent parameters
        self.optimizer = setup_optimizer(self.cfg.optimizer, self.q_agent)

        # ...and the evaluation policy (select the most likely action)
        self.eval_policy = Agents(self.q_agent, ArgmaxActionSelector())


# %%
def run(dqn: EpisodicDQN):
    for train_workspace in iter_episodes(dqn):
        q_values, terminated, reward, action = train_workspace[
            "q_values", "env/terminated", "env/reward", "action"
        ]
        
        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj
        must_bootstrap = ~terminated
        
        # Compute critic loss
        critic_loss = compute_critic_loss(dqn.cfg, reward, must_bootstrap, q_values, action)

        # Store the loss for tensorboard display
        dqn.logger.add_log("critic_loss", critic_loss, dqn.nb_steps)

        # Gradient step
        dqn.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            dqn.q_agent.parameters(), dqn.cfg.algorithm.max_grad_norm
        )
        dqn.optimizer.step()

        # Evaluate the current policy (if needed)
        dqn.evaluate()

# %%

# We setup tensorboard before running DQN
setup_tensorboard("./outputs/tblogs")


# %%
params={
  "save_best": False,
  "base_dir": "./outputs/${gym_env.env_name}/dqn-simple-S${algorithm.seed}_${current_time:}",
  "collect_stats": True,
  "logger": {
    "classname": "bbrl.utils.logger.TFLogger",
    "log_dir": "./outputs/tblogs/${gym_env.env_name}/dqn-simple-S${algorithm.seed}_${current_time:}",
    "cache_size": 10000,
    "every_n_seconds": 10,
    "verbose": False,    
    },
  "algorithm":{
    "seed": 3,
    "max_grad_norm": 0.5,
    "epsilon": 0.02,
    "n_envs": 8,
    "eval_interval": 2000,
    "max_epochs": 3_000,
# [[remove]]
    "max_epochs": 20 if testing_mode else 3_000,
    "eval_interval": 1 if testing_mode else 2000,
# [[/remove]]
    "nb_evals": 10,
    "discount_factor": 0.99,
    "architecture":{"hidden_size": [128, 128]},
  },
  "gym_env":{
    "env_name": "CartPole-v1",
  },
  "optimizer":
  {
    "classname": "torch.optim.Adam",
    "lr": 2e-3,
  }
}

dqn = EpisodicDQN(OmegaConf.create(params))


# %%

# Run and visualize the best agent
run(dqn)
dqn.visualize_best()

# %% [markdown] 


# ## What's next?
# 
# To get a full DQN, we need to do the following:
# - Add a replay buffer. We can add a replay buffer independently from the
#   target network. The version with a replay buffer and no target network
#   corresponds to [the NQF
#   algorithm](https://link.springer.com/content/pdf/10.1007/11564096_32.pdf).
#   This will be the aim of the next notebook.
# - Before adding the replay buffer, we will first move to a version of DQN
#   which uses the AutoResetGymAgent. This will be the aim of the next notebook
#   too.
# - We should also add a few extra-mechanisms which are present in the full DQN
#   version: starting to learn once the replay buffer is full enough, decreasing
#   the exploration rate epsilon...
# <!-- - We could also add visualization tools to visualize the learned Q network, by using the `plot_critic` function available in [`bbrl.visu.visu_critics`](https://github.com/osigaud/bbrl/blob/master/src/bbrl/visu/visu_critics.py#L13) -->
