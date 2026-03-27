import pandas as pd
import matplotlib.pyplot as plt

csv_path = "reward_log.csv"
window_size = 4000   # 你可以改成 5000 / 20000

df = pd.read_csv(csv_path)

main_cols = [
    "rew_dist_used",
    "r_align_used",
    "r_forward_used",
    "r_action_penalty_used",
    "combined_reward",
]

aux_cols = [
    "base_reward_used",
    "r_speed_track_used",
    "r_side_penalty_used",
    "r_hop_penalty_used",
]

state_cols = [
    
    "progress",
    "align",
    "forward_speed",
    "action_mag",
]

episode_cols = [
    "episode_is_success",
    "episode_final_dist",
    "episode_steps",
]

all_cols = ["num_timesteps"] + main_cols + aux_cols + state_cols + episode_cols

for col in all_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["num_timesteps"]).sort_values("num_timesteps")

# -----------------------------
# 原来的 step 级别统计
# -----------------------------
existing_main = [c for c in main_cols if c in df.columns]
existing_aux = [c for c in aux_cols if c in df.columns]
existing_state = [c for c in state_cols if c in df.columns]

group_cols = existing_main + existing_aux + existing_state

grouped = (
    df.groupby("num_timesteps")[group_cols]
    .mean()
    .reset_index()
    .sort_values("num_timesteps")
)

grouped["step_bin"] = (grouped["num_timesteps"] // window_size) * window_size

binned = (
    grouped.groupby("step_bin")[group_cols]
    .mean()
    .reset_index()
)

# 图1：主要参与总奖励的项
plt.figure(figsize=(10, 5))
for col in existing_main:
    plt.plot(binned["step_bin"], binned[col], label=col)
plt.xlabel("num_timesteps")
plt.ylabel("Mean Value")
plt.title("Main Reward Components (binned)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 图2：辅助项
plt.figure(figsize=(10, 5))
for col in existing_aux:
    plt.plot(binned["step_bin"], binned[col], label=col)
plt.xlabel("num_timesteps")
plt.ylabel("Mean Value")
plt.title("Auxiliary Reward Components (binned)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 图3：状态量变化
plt.figure(figsize=(10, 5))
for col in existing_state:
    plt.plot(binned["step_bin"], binned[col], label=col)
plt.xlabel("num_timesteps")
plt.ylabel("Mean Value")
plt.title("State / Behavior Metrics (binned)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 新增：episode 级别统计 hit rate
# -----------------------------
episode_df = df[df["episode_is_success"].notna()].copy()

if len(episode_df) > 0:
    episode_df["step_bin"] = (episode_df["num_timesteps"] // window_size) * window_size

    episode_binned = (
        episode_df.groupby("step_bin")[["episode_is_success", "episode_final_dist", "episode_steps"]]
        .mean()
        .reset_index()
    )

    # 图4：训练过程中的 hit rate / final dist / episode steps
    plt.figure(figsize=(10, 5))
    plt.plot(episode_binned["step_bin"], episode_binned["episode_is_success"], label="hit_rate")
    plt.plot(episode_binned["step_bin"], episode_binned["episode_final_dist"], label="mean_final_dist")
    plt.plot(episode_binned["step_bin"], episode_binned["episode_steps"], label="mean_episode_steps")
    plt.xlabel("num_timesteps")
    plt.ylabel("Mean Value")
    plt.title("Episode Metrics Over Training (binned)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 单独画 hit rate，更清楚
    plt.figure(figsize=(10, 5))
    plt.plot(episode_binned["step_bin"], episode_binned["episode_is_success"], label="hit_rate")
    plt.xlabel("num_timesteps")
    plt.ylabel("Hit Rate")
    plt.title("Hit Rate Over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No episode-level records found. Check whether episode_is_success is being written to CSV.")


# -----------------------------
# 按 align 分段，画 align * forward_speed 的分箱折线图
# x 轴颗粒度 = 4000 step
# -----------------------------
# -----------------------------
# 按 align 分段，画 align * forward_speed 占比折线图
# x 轴颗粒度 = 4000 step
# y 轴 = 每个分箱内的百分比
# -----------------------------
if "align" in df.columns and "forward_speed" in df.columns and "num_timesteps" in df.columns:
    plot_df = df[["num_timesteps", "align", "forward_speed"]].copy()
    plot_df["num_timesteps"] = pd.to_numeric(plot_df["num_timesteps"], errors="coerce")
    plot_df["align"] = pd.to_numeric(plot_df["align"], errors="coerce")
    plot_df["forward_speed"] = pd.to_numeric(plot_df["forward_speed"], errors="coerce")
    plot_df = plot_df.dropna()

    plot_df["align_forward"] = plot_df["align"] * plot_df["forward_speed"]
    plot_df["step_bin"] = (plot_df["num_timesteps"] // 4000) * 4000

    cond1 = plot_df["align"] > 0.9
    cond2 = (plot_df["align"] <= 0.9) & (plot_df["align"] > 0.7)
    cond3 = (plot_df["align"] <= 0.7) & (plot_df["align"] > 0.5)
    cond4 = plot_df["align"] <= 0.5

    line1 = plot_df.loc[cond1].groupby("step_bin")["align_forward"].sum()
    line2 = plot_df.loc[cond2].groupby("step_bin")["align_forward"].sum()
    line3 = plot_df.loc[cond3].groupby("step_bin")["align_forward"].sum()
    line4 = plot_df.loc[cond4].groupby("step_bin")["align_forward"].sum()

    line_df = pd.concat([line1, line2, line3, line4], axis=1).fillna(0)
    line_df.columns = [">0.9", "0.9-0.7", "0.7-0.5", "<=0.5"]

    total = line_df.sum(axis=1).replace(0, 1e-8)
    ratio_df = line_df.div(total, axis=0) * 100
    ratio_df = ratio_df.reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(ratio_df["step_bin"], ratio_df[">0.9"], label=">0.9")
    plt.plot(ratio_df["step_bin"], ratio_df["0.9-0.7"], label="0.9-0.7")
    plt.plot(ratio_df["step_bin"], ratio_df["0.7-0.5"], label="0.7-0.5")
    plt.plot(ratio_df["step_bin"], ratio_df["<=0.5"], label="<=0.5")

    plt.xlabel("num_timesteps")
    plt.ylabel("Percentage (%)")
    plt.title("Share of Sum(align * forward_speed) by Align Range Over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("num_timesteps / align / forward_speed column not found.")

# -----------------------------
# 按 align 分段，画“步数占比”折线图
# -----------------------------
if "align" in df.columns and "num_timesteps" in df.columns:
    plot_df = df[["num_timesteps", "align"]].copy()
    plot_df["num_timesteps"] = pd.to_numeric(plot_df["num_timesteps"], errors="coerce")
    plot_df["align"] = pd.to_numeric(plot_df["align"], errors="coerce")
    plot_df = plot_df.dropna()

    plot_df["step_bin"] = (plot_df["num_timesteps"] // 4000) * 4000

    cond1 = plot_df["align"] > 0.9
    cond2 = (plot_df["align"] <= 0.9) & (plot_df["align"] > 0.7)
    cond3 = (plot_df["align"] <= 0.7) & (plot_df["align"] > 0.5)
    cond4 = plot_df["align"] <= 0.5

    count1 = plot_df.loc[cond1].groupby("step_bin")["align"].count()
    count2 = plot_df.loc[cond2].groupby("step_bin")["align"].count()
    count3 = plot_df.loc[cond3].groupby("step_bin")["align"].count()
    count4 = plot_df.loc[cond4].groupby("step_bin")["align"].count()

    count_df = pd.concat([count1, count2, count3, count4], axis=1).fillna(0)
    count_df.columns = [">0.9", "0.9-0.7", "0.7-0.5", "<=0.5"]

    total = count_df.sum(axis=1).replace(0, 1e-8)
    ratio_df = count_df.div(total, axis=0) * 100
    ratio_df = ratio_df.reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(ratio_df["step_bin"], ratio_df[">0.9"], label=">0.9")
    plt.plot(ratio_df["step_bin"], ratio_df["0.9-0.7"], label="0.9-0.7")
    plt.plot(ratio_df["step_bin"], ratio_df["0.7-0.5"], label="0.7-0.5")
    plt.plot(ratio_df["step_bin"], ratio_df["<=0.5"], label="<=0.5")

    plt.xlabel("num_timesteps")
    plt.ylabel("Percentage (%)")
    plt.title("Align Range Step Ratio Over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("num_timesteps / align column not found.")