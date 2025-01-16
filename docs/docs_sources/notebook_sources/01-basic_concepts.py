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

# This notebook is designed to understand how to use BBRL in practice, it is part of the [BBRL documentation](https://github.com/osigaud/bbrl/docs/index.html).
# We show of to code a simple agent writing in a simple workspace to implement the Fibonacci sequence of numbers.
# Then some exercises are given to practice on your own.

# %% [markdown]
#
# # Installation and Imports
#
# The BBRL library is [here](https://github.com/osigaud/bbrl).
#
# Below, we import standard python packages, pytorch packages and gymnasium
# environments.

# %% tags=["hide-input"]
# Installs the necessary Python and system libraries
try:
    from easypip import easyimport, easyinstall, is_notebook
except ModuleNotFoundError as e:
    get_ipython().run_line_magic("pip", "install easypip")
    from easypip import easyimport, easyinstall, is_notebook

easyinstall("bbrl>=0.2.2")

# %% tags=["hide-input"]

import os
import sys
from pathlib import Path
import math

import time
from tqdm.auto import tqdm

import copy
from abc import abstractmethod, ABC
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% tags=["hide-input"]

# Imports all the necessary classes and functions from BBRL
from bbrl.agents.agent import Agent
# The workspace is the main class in BBRL, this is where all data is collected and stored
from bbrl.workspace import Workspace

# Agents(agent1, agent2, agent3,...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent over multiple timesteps in the workspace, 
# or until a given condition is reached
from bbrl.agents import Agents, TemporalAgent, PrintAgent

# %% [markdown]

# # Guided example
#
# In this example, we will study the Fibonacci sequence of numbers.
#
# ## Definition of agents

#
# In BBRL, agents interact with each other in a **workspace** by *reading* and *writing* information:
#
# - **reading**: by using `self.get((key, t))` to get the value of the tensor
#   identified by `key` (a string)
# - **writing**: by using `self.set((key, t), y)` to set the value of the tensor
#   identified by `key` to `y`
#
# To initialize the Fibonacci sequence, we need a specific agent to write a 0 at t=0 and a 1 at t=1.
# Note that the written numbers are encapsulated into an array and into a Tensor:
# - The array because when we will interact with several environments, we will have an array of variables at the same time step;
# - The tensor because most often the agent will be pytorch neural networks which read tensors and write tensors.

# %%

class InitAgent(Agent):
    """ The agent to initialize the sequence of numbers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs):
        self.set(("number", 0), torch.Tensor([0]))
        self.set(("number", 1), torch.Tensor([1]))

# %% [markdown]
#
# The Fibonacci agent reads the previous and current numbers, and writes the sum at the next time step
# Note that, as is often the case, the forward function reads a variable using self.get and write another variable (here the same) using self.set.

# %%

class FibonacciAgent(Agent):
    """ An agent to compute the Fibonacci sequence of numbers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, t, **kwargs):
        number = self.get(("number", t))
        prev_number = self.get(("number", t-1))
        next_value =  torch.Tensor([number + prev_number])
        self.set(("number", t+1), next_value)

# %% [markdown]

# ## Running agents in a workspace
#
# Now, let us run the agents into a workspace

# %%

# First we create a new workspace

workspace = Workspace() 

# Then we encapsulate the InitAgent into a Temporal agent which only contains this agent

init_agent = TemporalAgent(Agents(InitAgent()))

# We do the same for the Fibonacci agent

agent = FibonacciAgent()
fib_agent = TemporalAgent(Agents(agent))

# We execute the init_agent into the workspace for just one time step.

init_agent(workspace, t=0, n_steps=1)

# We check the content of the workspace

print("init:", workspace["number"])

# We execute the Fibonacci agent into the workspace for 10 time steps.

fib_agent(workspace, t=1, n_steps=10)

# We get the content of the workspace at a given time step

fib6 = workspace.get("number", 6)
print("6th Fibonacci number : ", fib6)

# %% [markdown]

# Let us now see the content of the workspace. We more systematically print all variables and their value

# %%
for key in workspace.variables.keys():
    print(key, workspace[key])

# %% [markdown]

# ## Composing agents, termination condition

# Now, let's imagine we want to stop the sequence when the number is greater that 200.
# For that, we add a TerminationChecker agent that writes a variable "stop"
# which is False in the number is lower than a threshold and True otherwise.

# Important note: if we want to use the `time_size()` method of a workspace,
# it is important that the time span of all variables is the same.
# Hence the Termination checker has to work over t+1 rather than over t, which is somewhat clumsy.

# %%
class TerminationChecker(Agent):
    """ An agent to check if the current Fibonacci number is above a threshold."""
    
    def __init__(self, threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def forward(self, t, **kwargs):
        number = self.get(("number", t+1))
        if t == 0:
            self.set(("stop", 0), torch.Tensor([False]))
        else:
            if number >= self.threshold:
                self.set(("stop", t+1), torch.Tensor([True]))
            else:
                self.set(("stop", t+1), torch.Tensor([False]))
        

# %% [markdown]

# ## Running the agents in a workspace
#
# Now we compose the agents and run them into a workspace

# %%

# First we create a new workspace

workspace = Workspace() 

# Then we encapsulate the InitAgent into a Temporal agent which only contains this agent

init_agent = TemporalAgent(Agents(InitAgent()))

# We do the same for the Fibonacci agent, but this time we compose it with the TerminationChecker and a PrintAgent,
# which is a utility to debug all steps into a terminal.

fib_agent = TemporalAgent(Agents(FibonacciAgent(), TerminationChecker(200), PrintAgent()))

# We execute the init_agent into the workspace for just one time step.

init_agent(workspace, t=0, n_steps=1)

# We check the content of the workspace

print("init:", workspace["number"])

# We execute the composed agent into the workspace until the stop condition gets True.

fib_agent(workspace, t=1, stop_variable="stop")

# We get the number of steps it took to reach the threshold

print("number of steps:", workspace.time_size())

# %% [markdown]

# # Exercise 1

# ## Question 1

# Similarly to the guided example above, use an agent and a workspace to generate the sequence of powers of 2.
# The sequence should stop once the value is over 10e10.
# You can call the agent computing the powers PowerAgent. It should use a simple product, not the power operator.

# %%

# [[student]]
class PowerAgent(Agent):
    """ An agent to compute the powers of 2."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, t, **kwargs):
        number = self.get(("number", t))
        self.set(("number", t+1), 2 * number)

class InitAgent(Agent):
    """ The agent to initialize the sequence of numbers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs):
        self.set(("number", 0), torch.Tensor([1]))

# First we create a new workspace

workspace = Workspace() 

# Then we encapsulate the InitAgent into a Temporal agent which only contains this agent

init_agent = TemporalAgent(Agents(InitAgent()))

# We do the same for the Power agent, but this time we compose it with the TerminationChecker and a PrintAgent,
# which is a utility to debug all steps into a terminal.

pow_agent = TemporalAgent(Agents(PowerAgent(), TerminationChecker(10e10), PrintAgent()))

# We execute the init_agent into the workspace for just one time step.

init_agent(workspace, t=0, n_steps=1)

# We check the content of the workspace

print("init:", workspace["number"])

# We execute the composed agent into the workspace until the stop condition gets True.

pow_agent(workspace, t=0, stop_variable="stop")

# We get the number of steps it took to reach the threshold

print("number of steps:", workspace.time_size())

# [[/student]]

# ## Question 2

# To stop at the first power over 10e10, you have probably used your PowerAgent and the TerminationChecker from the previous exercise.
# Now, combine both agents into a single PowerAgent. It should write both the powers and the stop variable.

# %%

# [[student]]

# [[/student]]

# %% [markdown]

# # Exercise 2

# The Collatz conjecture asks whether repeating two simple arithmetic operations will eventually transform every positive integer into 1.
# It concerns sequences of integers in which each term is obtained from the previous term as follows:
# - If the previous term is even, the next term is one half of the previous term.
# - If the previous term is odd, the next term is 3 times the previous term plus 1.
# The conjecture is that these sequences always reach 1, no matter which positive integer is chosen to start the sequence (source wikipedia).

# The goal of this exercise is to study the number of steps one needs to reach one starting from every numbers from 1 to 1000.

# ## Question 1

# Write the necessary code to get a CollatzAgent generating in a workspace the sequence of numbers starting from any number until it reaches 1.
# Hint: the CollatzAgent can generate its own stop variable: it is True if the current number is 1, False otherwise.
# Hint: you also need an InitAgent to pass the value from which you want to start.

# %%

# [[student]]
class InitAgent(Agent):
    """ An agent to compute the powers of 2."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, t, init_val, **kwargs):
        self.set(("number", 0), torch.Tensor([init_val]))
        self.set(("stop", 0), torch.Tensor([False]))

        
class CollatzAgent(Agent):
    """ An agent to run the Collatz conjecture."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, t, **kwargs):
        number = self.get(("number", t))
        if number % 2 == 0:
            next = number / 2
        else:
            next = 3 * number + 1
        self.set(("number", t+1), next)
        if next == 1:
            self.set(("stop", t+1), torch.Tensor([True]))
        else:
            self.set(("stop", t+1), torch.Tensor([False]))
            
col_agent = TemporalAgent(Agents(CollatzAgent()))
init_agent = TemporalAgent(Agents(InitAgent()))

for i in range(1000):
    workspace = Workspace() 
    init_agent(workspace, t=0, init_val=i+1, n_steps=1)    
    col_agent(workspace, t=0, stop_variable="stop")
    print (i, workspace.time_size())

# [[/student]]

# %% [markdown]

# ## Question 2

# Write the necessary code to get another agent generating in another workspace the number of steps needed to reach 1 when starting from the 1000 first numbers.
# This agent should use the previous agent.


# %%

# [[student]]

# [[/student]]

# Hint: this second agent should run the first agent in a worskpace in its forward function.

