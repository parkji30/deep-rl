# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import ale_py
import torch
import gymnasium as gym

gym.register_envs(ale_py)


# %%
def make_env(env_id: str):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

make_env("ALE/SpaceInvaders-v5")



# %%
