# %% tags=["copy"]

from _bbrl_learning_env import *

# %% [markdown]
#
# The `RLTransitions` defines the environment when using replay buffers. In
# particular, it defines `self.train_env` which is the environment used for
# training. This environment uses *autoreset*, i.e. when reaching a terminal
# state, a new environment is created.
#
# `RLTransitions` stores all the transitions $(s_t, a_t, r_t, s_{t+1}, ...)$
# into `self.replay_buffer`.
#
# The behavior of `RLTransitions` is controlled by the following configuration
# variables:
#
# - `gym_env.env_name` defines the gymnasium environment
# - `algorithm.n_envs` defines the number of parallel environments
# - `algorithm.seed` defines the random seed used (to initialize the agent and
#   the environment)
# - `algorithm.buffer_size` is the maximum number of transitions stored into the
# replay buffer
# 
# 


# %% tags=["hide-input"]

class RLTransitions(RLBase):
    """RL environment when using transition buffers"""
    
    train_agent: TemporalAgent
    
    """Base class for RL experiments with full episodes"""
    def __init__(self, cfg):
        super().__init__(cfg)

        # We use a non-autoreset workspace
        self.train_env = ParallelGymAgent(
            partial(make_env, cfg.gym_env.env_name, autoreset=True), 
            cfg.algorithm.n_envs
        ).seed(cfg.algorithm.seed)

        # Configure the workspace to the right dimension
        # Note that no parameter is needed to create the workspace.
        self.replay_buffer = ReplayBuffer(max_size=cfg.algorithm.buffer_size)


# %% [markdown]

# `iter_replay_buffers` provides an easy access to the replay buffer when
# learning. Its behavior depends on several configuration values:
#
# - `cfg.algorithm.max_epochs` defines the number of times the agent is used to
# collect transitions
# - `cfg.algorithm.learning_starts` defines the number of transitions before
# learning starts
#  
# Using `iter_replay_buffers` is simple:
# 
# ```py
#   class MyAgents(RLTransitions):
#       def __init__(self, cfg):
#           super().__init__(cfg)
#
#           # Define the train and evaluation policies
#           # (the agents compute the workspace `action` variable)
#           self.train_policy = ...
#           self.eval_policy = ...
# 
#   rl_env = MyRLAlgo(cfg)
#   for rb in iter_replay_buffers(rl_env):
#       # rb is a workspace containing transitions
#       ...
# ```

# %% tags=["hide-input"]

def iter_replay_buffers(env: RLTransitions):
    """Loop over transition buffers"""
    train_workspace = Workspace()

    epochs_pb = tqdm(range(env.cfg.algorithm.max_epochs))
    for epoch in epochs_pb:
        
        # This is the tricky part with transition buffers. The difficulty lies in the
        # copy of the last step and the way to deal with the n_steps return.
        #
        # The call to `train_agent(workspace, t=1, n_steps=cfg.algorithm.n_timesteps -
        # 1, stochastic=True)` makes the agent run a number of steps in the workspace.
        # In practice, it calls the
        # [`__call__(...)`](https://github.com/osigaud/bbrl/blob/master/src/bbrl/agents/agent.py#L59)
        # function which makes a forward pass of the agent network using the workspace
        # data and updates the workspace accordingly.
        #
        # Now, if we start at the first epoch (`epoch=0`), we start from the first step
        # (`t=0`). But when subsequently we perform the next epochs (`epoch>0`), we must
        # not forget to cover the transition at the border between the previous epoch
        # and the current epoch. To avoid this risk, we copy the information from the
        # last time step of the previous epoch into the first time step of the next
        # epoch. This is explained in more details in [a previous
        # notebook](https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5).
        if epoch == 0:
            # First run: we start from scratch
            env.train_agent(
                train_workspace, t=0, n_steps=env.cfg.algorithm.n_steps, stochastic=True
            )
        else:
            # Other runs: we copy the last step and start from there
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            env.train_agent(
                train_workspace, t=1, n_steps=env.cfg.algorithm.n_steps-1, stochastic=True
            )
        
        env.nb_steps += env.cfg.algorithm.n_steps * env.cfg.algorithm.n_envs

        # Add transitions to buffer
        transition_workspace = train_workspace.get_transitions()
        env.replay_buffer.put(transition_workspace)
        if env.replay_buffer.size() > env.cfg.algorithm.learning_starts:
            yield env.replay_buffer

        # Eval
        epochs_pb.set_description(
            f"nb_steps: {env.nb_steps}, "
            f"best reward: {env.best_reward:.2f}"
        )