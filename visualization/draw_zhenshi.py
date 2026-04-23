import matplotlib.pyplot as plt
import torch
import numpy as np
import os

# 数据路径和保存目录
save_dir = 'vis_resultszhenshi'
os.makedirs(save_dir, exist_ok=True)

# 加载数据：历史轨迹和真实未来轨迹
past_traj = torch.load('data/past1.pt', map_location='cpu')      # [num_scenes, 11, past_len, 2]
future_traj = torch.load('data/future1.pt', map_location='cpu')    # [num_scenes, 11, future_len, 2]
court = plt.imread("court.png")  # 球场背景图

# 如果原始数据坐标范围不一致，则需要缩放
scale = 94 / 28  # 同时用于 x, y

# 球场范围
x_min, x_max = 0, 94
y_min, y_max = 0, 50

def get_hist_color(actor_num):
    """
    历史轨迹颜色（较浅）
    """
    if actor_num < 5:  # 队1
        return "lightblue"
    elif actor_num < 10:  # 队2
        return "lightsalmon"
    else:  # 球
        return "lightgreen"

def get_future_color(actor_num):
    """
    真实未来轨迹颜色（中等深度）
    """
    if actor_num < 5:  # 队1
        return "deepskyblue"
    elif actor_num < 10:  # 队2
        return "salmon"
    else:  # 球
        return "mediumseagreen"

# 获取场景总数
num_scenes = past_traj.shape[0]
print(f"数据集中共有 {num_scenes} 个场景")

# 遍历每个场景
for scene_idx in range(num_scenes):
    # 获取该场景的所有历史轨迹和真实未来轨迹，形状分别为 [11, past_len, 2] 和 [11, future_len, 2]
    traj_scene = past_traj[scene_idx]
    future_scene = future_traj[scene_idx]
    num_actors = traj_scene.shape[0]  # 11（10人+1球）

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 叠加球场背景
    ax.imshow(court, extent=[x_min, x_max, y_max, y_min], alpha=0.6, zorder=0)

    # ------------------ 绘制历史轨迹 ------------------
    last_points = []  # 用于存储每个 actor 历史轨迹的最后一个点
    for actor_num in range(num_actors):
        hist_color = get_hist_color(actor_num)
        hist_traj = traj_scene[actor_num].numpy()  # [past_len, 2]
        x_hist = hist_traj[:, 0] * scale
        y_hist = hist_traj[:, 1] * scale

        ax.scatter(x_hist, y_hist, color=hist_color, s=15, alpha=0.9, zorder=5)
        ax.plot(x_hist, y_hist, color=hist_color, linewidth=2, alpha=0.9, zorder=5)
        last_points.append((x_hist[-1], y_hist[-1]))

    # ------------------ 绘制真实未来轨迹 ------------------
    for actor_num in range(num_actors):
        future_color = get_future_color(actor_num)
        true_future = future_scene[actor_num].numpy()  # [future_len, 2]
        x_future = true_future[:, 0] * scale
        y_future = true_future[:, 1] * scale

        # 连接历史轨迹最后一点和未来轨迹起始点
        last_x_hist, last_y_hist = last_points[actor_num]
        ax.plot([last_x_hist, x_future[0]], [last_y_hist, y_future[0]],
                color=future_color, linewidth=2, alpha=1, zorder=10)
        # 绘制未来轨迹
        ax.plot(x_future, y_future, color=future_color, linewidth=3, alpha=1, marker='o', zorder=10)
        # 标记未来轨迹终点
        ax.scatter(x_future[-1], y_future[-1], color=future_color, marker='*', s=80, zorder=10)

    plt.title(f"Scene {scene_idx}: History & True Future Trajectories", fontsize=12)
    plt.tight_layout()

    # 保存当前场景的图像
    save_path = os.path.join(save_dir, f"scene_{scene_idx}_history_true_future.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
