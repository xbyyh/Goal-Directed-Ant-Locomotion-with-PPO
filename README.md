# Goal-Directed Ant Locomotion with PPO

Reinforcement learning for goal-directed locomotion using MuJoCo Ant with curriculum learning and reward shaping.

## Demo

<p align="center">
  <img src="demo1.gif" width="700">
</p>
<p align="center">
  <img src="demo2.gif" width="700">
</p>

## Overview
We train a PPO agent in a custom Ant environment to first learn stable locomotion and then navigate toward a target.  
The project focuses on reward design to achieve smooth and directed motion.

## Key Features
- Algorithm PPO
- Two-stage training (locomotion → navigation)
- Custom reward shaping for direction and stability
- Success rate ~90%

## Method

### Training Strategy
- Stage 1 Learn stable locomotion
- Stage 2 Fine-tune for goal-directed movement

### Reward Design
- Progress toward target
- Direction alignment
- Piecewise speed reward (alignment-based)
- Side drift penalty
- Action smoothness penalty

## Results
- Stable locomotion learned in stage 1
- ~90% success rate in reaching target
- Reduced oscillation and drifting
- More direct trajectories after reward tuning

## Environment
Custom Ant environment with
- Target-based navigation
- Direction-aware reward
- Termination for stuck  out-of-bound  fall

## Training pipeline
```bash
python ant_stage1_train.py   ----   python ant_stage2_heritage_train.py

## Evaluation
- locomotion_test(stage1) --ppo_ant_stage1.zip
- python accuracy_test.py(stage2) --93%.zip
- python plot_reward_log.py

## Acknowledgments
Built with Stable-Baselines3 and MuJoCo.
