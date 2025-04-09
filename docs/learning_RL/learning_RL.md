# Learning Reinforcement Learning on your own

## Outlook

The pages below contain teaching material from Olivier Sigaud's reinforcement learning (RL) lectures. They are designed to help you learn RL on your own. Each page contains at least slides, and most often lessons in youtube videos and notebooks corresponding to labs. There might also be additional material to go beyond the basics.

The order of the lessons below matters, if you want to understand a lot about RL and you don't have much prior knowledge, you are encouraged to work on these lessons sequentially. Getting a good grasp on RL from the "central flow of lessons" below may take something like 10 days of work to a motivated beginner.

The labs are based on [the bbrl library](https://github.com/osigaud/bbrl), before studying DQN you should have a look at [the bbrl documentation](https://osigaud.github.io/docs/bbrl_docs/overview.md) and [the bbrl introductory notebooks](../bbrl_docs/notebooks/index.md).

These pages are subject to perpetual improvement, if you have any question or any suggestion to improve the content, send a message to [Olivier.Sigaud@sorbonne-universite.fr](mailto:Olivier.Sigaud@sorbonne-universite.fr).

## Reinforcement Learning: Central flow of lessons

[Overview: the 5 routes to Deep RL](overview.md)

------------

### Tabular reinforcement learning

[Tabular dynamic programming](./Tabular/tabular_dp.md)

[Tabular model-free reinforcement learning](./Tabular/tabular_mfrl.md)

[Reliable evaluation, stats and hyper-parameter tuning](./Tabular/tabular_actor_critic_stats_and_tuning.md)

[Tabular model-based reinforcement learning](./Tabular/tabular_mbrl.md)

------------

### Deep (model-free) reinforcement learning


[Deep Q-Network (DQN) and its successors](./DeepMFRL/dqn.md)

[Deep Deterministic Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3)](./DeepMFRL/ddpg.md)

[On-policy vs Off-policy](./DeepMFRL/onp_ofp.md)

[Policy Gradient approaches](./DeepMFRL/reinforce.md)

[Bias vs Variance](./DeepMFRL/bias_variance.md)

[Advantage Actor-Critic (A2C)](./DeepMFRL/a2c.md)

[Trust Region Policy Optimization (TRPO) and ACKTR](./DeepMFRL/trpo.md)

[Proximal Policy Optimization (PPO)](./DeepMFRL/ppo.md)

[Soft Actor-Critic (SAC)](./DeepMFRL/sac.md)

[High UTD ratio algorithms (TQC, DroQ)](./DeepMFRL/high_UTD.md)

[Deel Model-free RL Wrap-Up](./DeepMFRL/wrap_up.md)

The labs are based on [the bbrl library](https://github.com/osigaud/bbrl), before studying DQN you should have a look at [the bbrl documentation](https://osigaud.github.io/docs/overview.md) and [the bbrl introductory notebooks](https://osigaud.github.io/docs/notebooks).

------------

### Direct policy search and RL

[Direct policy search and RL: introduction](./EvoRL/intro.md)

[Direct policy search methods](./EvoRL/dps_methods.md)

[Policy gradient details](./EvoRL/pg_details.md)

[Direct policy search and RL: comparisons](./EvoRL/comparisons.md)

[Direct policy search and RL: combinations](./EvoRL/combinations.md)

[Population-based training](./EvoRL/pbt.md)

[TD-MPC](./EvoRL/td_mpc.md)

------------

### Goal-conditioned reinforcement learning

[GCRL: introduction](./GCRL/intro.md)

[GCRL: core concepts](./GCRL/core_concepts.md)

[GCRL: typology](./GCRL/typology.md)

[GCRL: skill learners](./GCRL/skill_learners.md)

[GCRL: hindsight experience replay](./her.md)

[GCRL: goal reachers](./GCRL/goal_reachers.md)

------------

## Advanced RL

[RLPD](./ARL/rlpd.md)

<!---
Direct Policy Search (DPS) approaches

Combining DPS and RL
--->
