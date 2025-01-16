
# %% [markdown]

# ### Training and evaluation environments

# In actor-critic algorithms relying on a replay buffer, the actor can be
# trained at each step during an episode. Besides, the training signal is the
# reward obtained during these episodes. So it may seem natural to display a
# learning curve corresponding to the performance of the training agent along
# the set of training episodes.
#
# But let us think of it. If the agent is changing during an episode, which
# agent are we truly evaluating? The one in the beginning of the episode? In the
# middle? In the end? We see that such evaluations based on an evolving agent
# makes no sense.
#
# What makes more sense is to train an agent for a number of steps, 
# evaluate it on a few episode to determine the performance of the obtained
# agent, then start again training. With this approach, the learning
# curve makes more sense, it shows the evolving performance of a succession of
# agents obtained after training sequences.
# 
# Separating training and evaluation provides additional opportunities. Often,
# we will train the agent using exploration, but we will evaluate it in a
# greedy, deterministic mode. Indeed, if the problem is truly an MDP, a
# deterministic policy can be optimal.
#
# Thus, in the general case, **we use two environments**: one for training and another one for evaluation.
# The same agent is connected to these two environments in two instances of
# TemporalAgent so that we train and evaluate the same network.
#
# In the context of this notebook, none of the environment agents uses autoreset.
#
# In practice, it is more efficient for training to use an AutoResetGymAgent, as we do not
# want to waste time if the task is done in an environment sooner than in the
# others, but this is more involved so we keep this for the advanced version of this notebook.
#
# By contrast, for evaluation, we just need to perform a fixed number of
# episodes (for statistics), thus it is more convenient to use a
# NoAutoResetGymAgent with a set of environments and just run one episode in
# each environment. Thus we can use the `env/done` stop variable and take
# the average over the cumulated reward of all environments.
# 
# Finally, to keep the story simple, we use a single environment for training.
#
# More details about using autoreset=True versus autoreset=False are given in [this
# notebook](https://colab.research.google.com/drive/1EX5O03mmWFp9wCL_Gb_-p08JktfiL2l5?usp=sharing).

# %% 
from typing import Tuple
from bbrl.agents.gymnasium import make_env, GymAgent, ParallelGymAgent
from functools import partial

def get_env_agents(cfg) -> Tuple[GymAgent, GymAgent]:
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`
    
    # Train environment
    train_env_agent = ParallelGymAgent(
        partial(make_env,  cfg.gym_env.env_name, autoreset=False),
        cfg.algorithm.n_envs
    ).seed(cfg.algorithm.seed)

    # Test environment
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name), 
        cfg.algorithm.nb_evals
    ).seed(cfg.algorithm.seed)

    return train_env_agent, eval_env_agent
