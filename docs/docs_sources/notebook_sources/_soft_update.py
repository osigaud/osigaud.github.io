
# %% [markdown]

# ### Soft parameter updates
# 
# To update the target critic, one uses the following equation:
# $\theta' \leftarrow \tau \theta + (1- \tau) \theta'$
# where $\theta$ is the vector of parameters of the critic, and $\theta'$ is the vector of parameters of the target critic.
# The `soft_update_params(...)` function is in charge of performing this soft update.


# %% tags=["hide-input"]
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
