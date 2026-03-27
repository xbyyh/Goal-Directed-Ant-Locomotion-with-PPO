# custom_ant/ant_custom_env.py

import os
from typing import Any, Dict, Optional

import mujoco
from gymnasium.envs.mujoco.ant_v4 import AntEnv
import numpy as np
from .terrains import TERRAIN_MAP
#from custom_ant.terrains import TERRAIN_MAP

class AntCustomEnv(AntEnv):
    """
    Custom Ant environment with procedural heightfield terrains.
    Compatible with MuJoCo 3.x and Gymnasium 1.2.2.

    Assumes ant_custom.xml defines:
      - <hfield name="terrain" ... />
      - <geom type="hfield" hfield="terrain" ... />
    """

    def __init__(
        self,
        terrain_type: str = "flat",
        
        **kwargs: Any,
    ):
        self.terrain_type = terrain_type
        # Target at +15, +15 (inside heightfield which spans -20 to +20)
        self.target_xy = np.array((10.0, 10.0), dtype=np.float64) #真正确认reward运动点的地方
        self.prev_dist = None
        self.episode_step = 0
        self.w_forward = 1.0      # <-- tune this
        self.theta_forward = 0.0  # <-- direction angle (0 = +x)
        self.stuck_step_count = 0
        self.stuck_speed_threshold = 0.03   # 低于这个速度算“几乎不动”
        self.stuck_max_steps = 100          # 连续 100 步不动就终止
        self.init_dist_to_target = None
        xml_path = os.path.join(os.path.dirname(__file__), "ant_custom.xml")
        super().__init__(xml_file=xml_path, **kwargs)
        self._apply_terrain()
    # ------------------------------------------------------------------ #
    # Apply terrain heights to the MuJoCo heightfield
    # ------------------------------------------------------------------ #
    
    def _apply_terrain(self) -> None:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        #print("=== APPLY TERRAIN ===")
        #print("target_xy (你设的):", self.target_xy)

        #print("site_pos (实际写入):", self.model.site_pos[site_id])
        # If model has no heightfield, do nothing
        if len(self.model.hfield_data) == 0:
            return

        # Assume the first heightfield is our terrain
        nrow = int(self.model.hfield_nrow[0])
        ncol = int(self.model.hfield_ncol[0])

        if self.terrain_type not in TERRAIN_MAP:
            raise ValueError(
                f"Unknown terrain_type={self.terrain_type}. "
                f"Available: {list(TERRAIN_MAP.keys())}"
            )

        terrain_fn = TERRAIN_MAP[self.terrain_type]
        heightfield = terrain_fn(nrow, ncol)

        if heightfield.shape != (nrow, ncol):
            raise ValueError(
                f"Terrain function '{self.terrain_type}' returned shape "
                f"{heightfield.shape}, expected ({nrow}, {ncol})"
            )

        # Write heights into MuJoCo model
        self.model.hfield_data[:] = heightfield.flatten()

        # Update target site position based on terrain
        z_scale = float(self.model.hfield_size[0, 2])  # z scaling factor (5)

        # For slope terrain, max height is 1.0 * z_scale at the -x,-y corner
        # Just set target high enough to be above max terrain height
        max_terrain_height = float(heightfield.max()) * z_scale
        target_x, target_y = self.target_xy

        # Find the target_site and update its position
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        if site_id >= 0:
            self.model.site_pos[site_id, 0] = target_x
            self.model.site_pos[site_id, 1] = target_y
            self.model.site_pos[site_id, 2] = max_terrain_height + 1.0  # above max terrain

        # Recompute derived data
        mujoco.mj_forward(self.model, self.data)
    # ------------------------------------------------------------------ #
    # Re-generate terrain on reset for stochastic terrain types
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        #self.prev_dist = None
        #self.episode_step = 0

        self._apply_terrain()
        self.prev_dist = None
        self.episode_step = 0
        self.stuck_step_count = 0
        obs, info = super().reset(seed=seed, options=options)

        self.data.qpos[0] = -10.0
        self.data.qpos[1] = -10.0
        self.data.qpos[2] = 0.55

        torso_xy = self.data.qpos[0:2].copy()
        self.init_dist_to_target = float(np.linalg.norm(self.target_xy - torso_xy))

        return obs, info

   

    # ------------------------------------------------------------------ #
    # OVERRIDE step: ignore "health" termination
    # ------------------------------------------------------------------ #
    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)
        self.episode_step += 1
        done_reason = None

        torso_xy = self.data.qpos[0:2].copy()
        x, y = torso_xy
        to_target = self.target_xy - torso_xy
        dist_to_target = float(np.linalg.norm(to_target))

        if dist_to_target > 1e-6:
            dir_target = to_target / dist_to_target
        else:
            dir_target = np.zeros_like(to_target)

        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        vz = float(self.data.qvel[2])
        yaw_rate = float(self.data.qvel[5])
        v_xy = np.array([vx, vy], dtype=np.float64)

        v_norm = np.linalg.norm(v_xy)
        if v_norm > 0.12:
            dir_move = v_xy / v_norm
            align = float(np.dot(dir_move, dir_target))   # [-1, 1]
        else:
            align = 0.0
        
        if v_norm < self.stuck_speed_threshold:
            self.stuck_step_count += 1
        else:
            self.stuck_step_count = 0

        # ------- 所有候选奖励项都先算出来 -------
        progress_ratio = 0.0
        if self.init_dist_to_target is not None and self.init_dist_to_target > 1e-6:
            progress_ratio = np.clip(1.0 - dist_to_target / self.init_dist_to_target, 0.0,1.0)

        #threshold = 0.866#(30°)
        
        # 稠密 align reward：只要更对齐就持续给奖励
        #if align >= 0.0:
            #r_align = 0.25 * (align ** 2) * (1.0 + 3.0 * progress_ratio)
        #else:
            #r_align = 0.08 * align
        if align >= 0.0:
            r_align = 0.25 * align * (1.0 + 3.0 * progress_ratio)#92版本
        else:
            r_align = 0.08 * align
        

        speed = float(np.dot(v_xy, dir_target))
        forward_speed = max(speed, 0.0)
        r_forward = 0.5 * forward_speed
        
        #原speedtrack
        v_target = 0.5
        r_speed_track = -0.1 * (forward_speed - v_target) ** 2

        # align ∈ [-1, 1]
        # forward_speed >= 0

        



        lat_speed = v_xy - (speed * dir_target)
        r_side_penalty = -0.2 * float(np.dot(lat_speed, lat_speed))

        r_hop_penalty = -0.1 * max(vz, 0.0) ** 2
        
        #身体扭转
        r_yaw_penalty = -0.05 * (yaw_rate ** 2)

        action_mag = float((action ** 2).sum())
        r_action_penalty = -0.001 * action_mag

        if self.prev_dist is None:
            self.prev_dist = dist_to_target

        progress = self.prev_dist - dist_to_target
        if progress >= 0:
            rew_dist = 0.9 * progress
        else:   
            rew_dist = 0.3 * progress
        self.prev_dist = dist_to_target

        # ------- 终止条件 -------
        if self.stuck_step_count >= self.stuck_max_steps:
            terminated = True
            done_reason = "stuck"

        if dist_to_target < 3.0:
            terminated = True
            done_reason = "reached_target"

        if abs(torso_xy[0]) > 20 or abs(torso_xy[1]) > 20:
            terminated = True
            done_reason = "out_of_bounds"
        
        z = float(self.data.qpos[2])
        if z < 0.25:
            terminated = True
            done_reason = "fell_down"  

        # ------- 这里只决定哪些项参与总 reward -------
        coef_action_penalty = 1.0
        coef_rew_dist = 4.0
        coef_r_align = 3
        coef_r_forward = 0.0
        coef_r_speed_track = 0.01
        coef_r_side_penalty = 0.005
        coef_r_hop_penalty = 0.0
        coef_r_yaw_penalty = 0.0

        reward = (
            + coef_action_penalty * r_action_penalty
            + coef_rew_dist * rew_dist
            + coef_r_align * r_align
            + coef_r_yaw_penalty * r_yaw_penalty
            + coef_r_forward * r_forward
            + coef_r_speed_track * r_speed_track
            + coef_r_side_penalty * r_side_penalty
            + coef_r_hop_penalty * r_hop_penalty
        )

        # ------- 全部写进 info，包含“没启用的项” -------
        info["num_env_steps"] = self.episode_step
        info["dist_to_target"] = dist_to_target
        #info["progress"] = progress    
        info["align"] = align
        #info["speed"] = speed
        info["forward_speed"] = forward_speed
        info["v_norm"] = v_norm
        #info["vx"] = vx
        #info["vy"] = vy
        #info["vz"] = vz
        #info["z"] = z
        #info["action_mag"] = action_mag
        
        
        #info["r_action_penalty_raw"] = r_action_penalty
        #info["rew_dist_raw"] = rew_dist
        #info["r_align_raw"] = r_align
        #info["r_forward_raw"] = r_forward
        #info["r_speed_track_raw"] = r_speed_track
        #info["r_side_penalty_raw"] = r_side_penalty
        #info["r_hop_penalty_raw"] = r_hop_penalty

      
        #info["coef_action_penalty"] = coef_action_penalty
        #info["coef_rew_dist"] = coef_rew_dist
        #info["coef_r_align"] = coef_r_align
        #info["coef_r_forward"] = coef_r_forward
        #info["coef_r_speed_track"] = coef_r_speed_track
        #info["coef_r_side_penalty"] = coef_r_side_penalty
        #info["coef_r_hop_penalty"] = coef_r_hop_penalty

       
        info["r_action_penalty_used"] = coef_action_penalty * r_action_penalty
        info["yaw_penalty_used"] = coef_r_yaw_penalty * r_yaw_penalty
        info["rew_dist_used"] = coef_rew_dist * rew_dist
        info["r_align_used"] = coef_r_align * r_align
        info["r_forward_used"] = coef_r_forward * r_forward
        info["r_speed_track_used"] = coef_r_speed_track * r_speed_track
        info["r_side_penalty_used"] = coef_r_side_penalty * r_side_penalty
        info["r_hop_penalty_used"] = coef_r_hop_penalty * r_hop_penalty

        info["combined_reward"] = reward
        info["done_reason"] = done_reason if done_reason is not None else ""

        #print("step final info keys:", info.keys())
        #print("step final align:", info.get("align"), "forward_speed:", info.get("forward_speed"))

        return obs, reward, terminated, truncated, info 