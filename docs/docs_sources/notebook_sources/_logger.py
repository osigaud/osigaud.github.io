
# %% [markdown]

# ### The Logger class
#
# The logger is in charge of collecting statistics during the training
# process.
# 
# Having logging provided under the hood is one of the features allowing you
# to save time when using RL libraries like BBRL.
# 
# In these notebooks, the logger is defined as `bbrl.utils.logger.TFLogger` so as
# to use a tensorboard visualisation (see the parameters part `params = { "logger":{ ...` below).
# 
# Note that the BBRL Logger is also saving the log in a readable format such
# that you can use `Logger.read_directories(...)` to read multiple logs, create
# a dataframe, and analyze many experiments afterward in a notebook for
# instance. The code for the different kinds of loggers is available in the
# [bbrl/utils/logger.py](https://github.com/osigaud/bbrl/blob/master/src/bbrl/utils/logger.py)
# file.
# 
# `instantiate_class` is an inner BBRL mechanism. The
# `instantiate_class`function is available in the
# [`bbrl/__init__.py`](https://github.com/osigaud/bbrl/blob/master/src/bbrl/__init__.py)
# file.

# %% tags=["hide-input"]

from bbrl import instantiate_class

class Logger():

    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, loss, steps):
        self.logger.add_scalar(log_string, loss.item(), steps)

    # A specific function for RL algorithms having a critic, an actor and an entropy losses
    def log_losses(self, critic_loss, entropy_loss, actor_loss, steps):
        self.add_log("critic_loss", critic_loss, steps)
        self.add_log("entropy_loss", entropy_loss, steps)
        self.add_log("actor_loss", actor_loss, steps)

    def log_reward_losses(self, rewards, nb_steps):
        self.add_log("reward/mean", rewards.mean(), nb_steps)
        self.add_log("reward/max", rewards.max(), nb_steps)
        self.add_log("reward/min", rewards.min(), nb_steps)
        self.add_log("reward/median", rewards.median(), nb_steps)
        
