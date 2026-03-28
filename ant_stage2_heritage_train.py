from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_ant.ant_custom_env import AntCustomEnv
from log_reward_callback import RewardLoggingCallback

callback = RewardLoggingCallback(log_path="logs/reward_log.csv")

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./checkpoints/",
    name_prefix="ppo_ant_stage2"
)

def make_env():
    return AntCustomEnv(terrain_type="flat")

env = make_vec_env(make_env, n_envs=16)

model_path = "93%"
model = PPO.load(model_path, env=env)
model.tensorboard_log = "./tb_logs/"

model.learning_rate = lambda _: 5e-5
model.clip_range = lambda _: 0.1
model.n_epochs = 5
model.ent_coef = 0.0

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

model.learn(
    total_timesteps=500_000,
    reset_num_timesteps=False,
    tb_log_name=f"stage2_flat_target_{run_id}",
    callback=[callback, checkpoint_callback],
)

save_name = f"ppo_ant_stage2_flat_target_{run_id}"
model.save(save_name)

print(f"saved model to: {save_name}.zip")
print(f"tensorboard logs in: ./tb_logs/stage2_flat_target_{run_id}")
print("reward log saved to: reward_log.csv")
print("checkpoints saved in: ./checkpoints/")