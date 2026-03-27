import csv
import os
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_path="reward_log.csv", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.file = None
        self.writer = None
        self.header_written = False

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self.file = open(self.log_path, "w", newline="", encoding="utf-8")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for info in infos:
            #print("align=", info.get("align"), "forward_speed=", info.get("forward_speed"),"dist_to_target=", info.get("dist_to_target"))
            episode_info = info.get("episode", {})

            row = {
                "num_timesteps": self.num_timesteps,
                "num_env_steps": info.get("num_env_steps", None),

                "dist_to_target": info.get("dist_to_target", None),
                "final_dist": info.get("final_dist", None),
                "episode_step_count": info.get("episode_step_count", None),
                "is_success": info.get("is_success", None),

                #"progress": info.get("progress", None),
                "align": info.get("align", None),
                #"speed": info.get("speed", None),
                "forward_speed": info.get("forward_speed", None),
                #"vx": info.get("vx", None),
                #"vy": info.get("vy", None),
                #"vz": info.get("vz", None),
                "z": info.get("z", None),
                #"action_mag": info.get("action_mag", None),

                #"base_reward_raw": info.get("base_reward_raw", None),
                #"r_action_penalty_raw": info.get("r_action_penalty_raw", None),
                #"rew_dist_raw": info.get("rew_dist_raw", None),
                #"r_align_raw": info.get("r_align_raw", None),
                #"r_forward_raw": info.get("r_forward_raw", None),
                #"r_speed_track_raw": info.get("r_speed_track_raw", None),
                #"r_side_penalty_raw": info.get("r_side_penalty_raw", None),
                #"r_hop_penalty_raw": info.get("r_hop_penalty_raw", None),
                "v_norm": info.get("v_norm", None),
                "base_reward_used": info.get("base_reward_used", None),
                "r_action_penalty_used": info.get("r_action_penalty_used", None),
                "rew_dist_used": info.get("rew_dist_used", None),
                "r_align_used": info.get("r_align_used", None),
                "r_forward_used": info.get("r_forward_used", None),
                "r_speed_track_used": info.get("r_speed_track_used", None),
                "r_side_penalty_used": info.get("r_side_penalty_used", None),
                "r_hop_penalty_used": info.get("r_hop_penalty_used", None),
                 "yaw_penalty_used": info.get("yaw_penalty_used",None),

                "combined_reward": info.get("combined_reward", None),
                "done_reason": info.get("done_reason", ""),

                "episode_is_success": episode_info.get("is_success", None),
                "episode_final_dist": episode_info.get("final_dist", None),
                "episode_steps": episode_info.get("steps", None),
                "episode_done_reason": episode_info.get("done_reason", None),
            }

            if not self.header_written:
                self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
                self.writer.writeheader()
                self.header_written = True

            self.writer.writerow(row)

        self.file.flush()
        return True

    def _on_training_end(self) -> None:
        if self.file is not None:
            self.file.close()