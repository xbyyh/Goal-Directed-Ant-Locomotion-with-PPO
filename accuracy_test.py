import time
import numpy as np
from stable_baselines3 import PPO
from custom_ant.ant_custom_env import AntCustomEnv
from stable_baselines3 import SAC

SUCCESS_RADIUS = 3.0  # 要和你环境里的成功阈值一致
MAX_STEPS = 50000

env = AntCustomEnv(terrain_type="flat", render_mode="human")

# model = SAC.load("sac_ant_stage2_flat_target_20260328_015550", env=env)
# model = PPO.load("ppo_ant_stage2_flat_target_20260328_035157", env=env)
model = PPO.load("93%", env=env)

success_count = 0
episode_count = 0
episode_step = 0

obs, info = env.reset()

for global_step in range(MAX_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.0)

    episode_step += 1

    if terminated or truncated:
        torso_xy = env.data.qpos[0:2].copy()
        dist_to_target = float(np.linalg.norm(env.target_xy - torso_xy))
        reached = dist_to_target < SUCCESS_RADIUS

        episode_count += 1
        if reached:
            success_count += 1

        success_rate = success_count / episode_count

        print(
            f"episode={episode_count}, "
            f"steps={episode_step}, "
            f"target={env.target_xy}, "
            f"torso_xy={torso_xy}, "
            f"dist={dist_to_target:.3f}, "
            f"success={reached}, "
            f"success_rate={success_rate:.2%}"
        )

        obs, info = env.reset()
        episode_step = 0

env.close()

if episode_count > 0:
    print(f"\nFinal success rate: {success_count}/{episode_count} = {success_count / episode_count:.2%}")
else:
    print("\nNo episode finished.")