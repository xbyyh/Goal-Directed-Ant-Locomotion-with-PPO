import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("Ant-v4", n_envs=16)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=256,        # 每个环境收集 128 步
    batch_size=1024,    # 16*128 = 2048，batch_size 设成它的约数
)

model.learn(total_timesteps=2_000_000)
model.save("ppo_ant_stage1")