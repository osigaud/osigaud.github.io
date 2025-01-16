# %% [markdown]

# ### Training and evaluation environments
# 
# We build two environments: one for training and another one for evaluation.
# 
# For training, it is more efficient to use an autoreset agent, as we do not
# want to waste time if the task is done in an environment sooner than in the
# others.
# 
# By contrast, for evaluation, we just need to perform a fixed number of
# episodes (for statistics), thus it is more convenient to use a
# noautoreset agent with a set of environments and just run one episode in
# each environment. Thus we can use the `env/done` stop variable and take the
# average over the cumulated reward of all environments.
# 
# See [this
# notebook](https://colab.research.google.com/drive/1Ui481r47fNHCQsQfKwdoNEVrEiqAEokh?usp=sharing)
# for explanations about agents and environment agents.


# %% 
from typing import Tuple
from bbrl.agents.gymnasium import make_env, GymAgent, ParallelGymAgent
from functools import partial

def get_env_agents(cfg, *, autoreset=True, include_last_state=True) -> Tuple[GymAgent, GymAgent]:
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`
    
    # Train environment
    train_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, autoreset=autoreset),
        cfg.algorithm.n_envs, 
        include_last_state=include_last_state
    ).seed(cfg.algorithm.seed)

    # Test environment
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name), 
        cfg.algorithm.nb_evals,
        include_last_state=include_last_state
    ).seed(cfg.algorithm.seed)

    return train_env_agent, eval_env_agent
