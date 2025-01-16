# %% tags=["copy"]

from _bbrl_learning import *
from _logger import *
from _bbrl_environment import *

# %% [markdown]
# %% [markdown]

# # Learning environment
#
# To setup a common learning environment for RL algorithms, we use the `RLBase`
# class. This class:
#
# 1. Initialize the environment (random seed, logger, evaluation environment)
# 2. Define a `evaluate` method that keeps the best agent so far
# 3. Define a `visualize_best` that allows to watch the best agent
# 
# Subclasses need to define `self.train_policy` and `self.eval_policy`, two
# BBRL agents that respectively choose actions when training and evaluating.
# 
# The behavior of `RLBase` is controlled by the following configuration
# variables:
#
# - `base_dir` defines the directory subpath used when outputing losses during
# training as well as other outputs (serialized agent, global statistics, etc.)
# - `algorithm.seed` defines the random seed used (to initialize the agent and
#   the environment)
# - `gym_env` defines the gymnasium environment, and in particular
# `gym_env.env_name` the name of the gymansium environment
# - `logger` defines what type of logger is used to log the different values
# associated with learning
# - `algorithm.eval_interval` defines the number of observed transitions between
# each evaluation of the agent
# 


# %% tags=["hide-input"]

import numpy as np
from typing import Any
import logging
from abc import ABC
from functools import cached_property

class RLBase(ABC):
    """Base class for Reinforcement learning algorithms
    
    This class deals with common processing:

    - defines the logger, the train and evaluation agents
    - defines how to evaluate a policy
    """

    #: The configuration
    cfg: Any

    #: The evaluation environment deals with the last action, and produces a new
    # state of the environment
    eval_env: Agent

    #: The training policy
    train_policy: Agent

    #: The evaluation policy (if not defined, uses the train policy)
    eval_policy: Agent

    def __init__(self, cfg):
        # Basic initialization
        self.cfg = cfg
        torch.manual_seed(cfg.algorithm.seed)

        # Sets the base directory and logger directory
        base_dir = Path(self.cfg.base_dir)
        self.base_dir = Path("outputs") / Path(self.cfg.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the logger class
        if not hasattr(cfg.logger, "log_dir"):
            cfg.logger.log_dir = str(Path("outputs") / "tblogs" / base_dir)
        self.logger = Logger(cfg)

        # Subclasses have to define the train policy
        self.train_policy = None

        # Sets up the evaluation environment
        self.eval_env = ParallelGymAgent(
            partial(make_env, cfg.gym_env.env_name), 
            cfg.algorithm.nb_evals
        ).seed(cfg.algorithm.seed)
        self.eval_policy = None

        # Initialize values
        self.last_eval_step = 0
        self.nb_steps = 0
        self.best_policy = None
        self.best_reward = -torch.inf

        # Records the rewards
        self.eval_rewards = []

    @cached_property
    def train_agent(self):
        """Returns the train agent

        The agent is composed of a policy agent and the train environment.      
        This method supposes that `self.train_policy` has been setup
        """
        assert self.train_policy is not None, "The train_policy property is not defined before the policy is set"
        return TemporalAgent(Agents(self.train_env, self.train_policy))

    @cached_property
    def eval_agent(self):
        """Returns the evaluation agent 
        
        The agent is composed of a policy agent and the evaluation environment
        
        Uses `self.eval_policy` (or `self.train_policy` if not defined)

        """
        assert self.eval_policy is not None or self.train_policy is not None, "eval_agent property is not defined before the policy is set"
        return TemporalAgent(Agents(self.eval_env, self.eval_policy if self.eval_policy is not None else self.train_policy))

    def evaluate(self):
        """Evaluate the current policy `self.eval_policy`
        
        Evaluation is conducted every `cfg.algorithm.eval_interval` steps, and
        we keep a copy of the best agent so far in `self.best_policy`
        
        Returns True if the current policy is the best so far
        """
        if (self.nb_steps - self.last_eval_step) > self.cfg.algorithm.eval_interval:
            self.last_eval_step = self.nb_steps
            eval_workspace = Workspace() 
            self.eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done"
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            self.logger.log_reward_losses(rewards, self.nb_steps)

            if getattr(self.cfg, "collect_stats", False):
                self.eval_rewards.append(rewards)

            rewards_mean = rewards.mean()
            if rewards_mean > self.best_reward:
                self.best_policy = copy.deepcopy(self.eval_policy)
                self.best_reward = rewards_mean
                return True

    def save_stats(self):
        """Save reward statistics into `stats.npy`"""
        if getattr(self.cfg, "collect_stats", False) and self.eval_rewards:
            data = torch.stack(self.eval_rewards, axis=-1) 
            with (self.base_dir / "stats.npy").open("wt") as fp:
                np.savetxt(fp, data.numpy())

    def visualize_best(self):
        """Visualize the best agent"""
        env = make_env(self.cfg.gym_env.env_name, render_mode="rgb_array")
        path = self.base_dir / "best_agent.mp4"
        print(f"Video of best agent recorded in {path}")
        record_video(env, self.best_policy, path)
        return video_display(str(path.absolute()))
