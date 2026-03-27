import gymnasium as gym
from stable_baselines3 import SAC
import time
# 创建环境（带渲染）
env = gym.make("Ant-v4", render_mode="human")

# 加载模型
model = SAC.load("sac_ant_stage14", device="auto")

obs, _ = env.reset()

for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.03)
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()[time]