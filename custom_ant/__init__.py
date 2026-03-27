from gymnasium.envs.registration import register

register(
    id="CustomAnt-v0",
    entry_point="custom_ant.ant_custom_env:AntCustomEnv",
    max_episode_steps=2000

)

from .ant_custom_env import AntCustomEnv

__all__ = ["AntCustomEnv"]
