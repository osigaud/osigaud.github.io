# %% tags=["copy"]

from _bbrl_learning_env import *

# %% [markdown]
#
# The `RLEpisodic` defines the environment when using episodes. In particular,
# it defines `self.train_env` which is the environment used for training. As
# `algorithm.n_envs` are used in parallel, when a episode ends, we don't stop
# the other episodes. To cater for this, 

# 1. the workspace variable `env/done` is set to `True` for all the next time
# steps
# 2. The variable `env/reward` is set to 0 for all the steps 
#
# The behavior of `RLEpisodic` is controlled by the following configuration
# variables:
#
# - `gym_env.env_name` defines the gymnasium environment
# - `algorithm.n_envs` defines the number of parallel environments
# - `algorithm.seed` defines the random seed used (to initialize the agent and
#   the environment)
# 
# 
# %% tags=["hide-input"]


class RLEpisodic(RLBase):
    """Base class for RL experiments with full episodes"""
    def __init__(self, cfg, autoreset=False):
        super().__init__(cfg)

        self.train_env = ParallelGymAgent(
            partial(make_env, cfg.gym_env.env_name, autoreset=autoreset), 
            cfg.algorithm.n_envs,
        ).seed(cfg.algorithm.seed)


# %% [markdown]

# `iter_episodes` and `iter_partial_episodes` (autoreset) allow
# to iterate over the train workspace by sampling

# %% tags=["hide-input"]

def iter_episodes(rl: RLEpisodic):
    pbar = tqdm(range(rl.cfg.algorithm.max_epochs))

    train_workspace = Workspace()

    for rl.epoch in pbar:
        # Collect samples
        train_workspace = Workspace()
        rl.train_agent(train_workspace, t=0, stop_variable="env/done")

        # Update the number of steps
        rl.nb_steps += int((~train_workspace["env/done"]).sum())

        # Perform a learning step
        yield train_workspace

        # Eval
        pbar.set_description(f"nb_steps: {rl.nb_steps}, best reward: {rl.best_reward:.2f}")


def iter_partial_episodes(rl: RLEpisodic, episode_steps: int):
    pbar = tqdm(range(rl.cfg.algorithm.max_epochs))
    train_workspace = Workspace()

    for rl.epoch in pbar:
        if rl.epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            rl.train_agent(
                train_workspace, t=1, n_steps=episode_steps-1, stochastic=True
            )
        else:
            rl.train_agent(
                train_workspace, t=0, n_steps=episode_steps, stochastic=True
            )

        rl.nb_steps += int((~train_workspace["env/done"]).sum())
        yield train_workspace

        pbar.set_description(f"nb_steps: {rl.nb_steps}, best reward: {rl.best_reward:.2f}")
