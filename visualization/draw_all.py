import matplotlib.pyplot as plt
import torch
import numpy as np
import os

# ===================== 路径 =====================
save_dir = 'vis_results_besttraj_selected_scenes'
os.makedirs(save_dir, exist_ok=True)

PAST_PATH = 'data/past1.pt'            # [num_scenes, 11, past_len, 2]
FUTURE_PATH = 'data/future1.pt'        # [num_scenes, 11, future_len, 2]  (没有就改 or 关掉 oracle)
PRED_PATH = 'data/prediction1.pt'      # [num_scenes*11, K, future_len, 2] (K条采样，相对位移)
COURT_IMG = 'court.png'

# ===================== 只画指定场景 =====================
# 例如只画第0、3、7号场景
TARGET_SCENES = [68, 93]

# 如果你只想画一个场景，也可以写成：
# TARGET_SCENES = [5]

# ===================== 选择“最好轨迹”的方式 =====================
USE_ORACLE_BEST = True     # True: 需要 future，按 ADE 选最优；False: 无GT，选 medoid
USE_FDE = False            # True: 用 FDE(终点误差)；False: 用 ADE(平均误差)
SHOW_FIG = True

# ===================== 坐标与场地 =====================
scale = 94 / 28
x_min, x_max = 0, 94
y_min, y_max = 0, 50

# ===================== 颜色 =====================
def get_hist_color(actor_num):
    if actor_num < 5:
        return "lightblue"
    elif actor_num < 10:
        return "lightsalmon"
    else:
        return "lightgreen"

def get_pred_color(actor_num):
    if actor_num < 5:
        return "deepskyblue"
    elif actor_num < 10:
        return "salmon"
    else:
        return "mediumseagreen"

# ===================== 载入数据 =====================
past_traj = torch.load(PAST_PATH, map_location='cpu')     # [S,11,past,2]
court = plt.imread(COURT_IMG)

pred = torch.load(PRED_PATH, map_location='cpu')          # [S*11,K,T,2] (相对)
if pred.ndim != 4:
    raise ValueError(f"Unexpected prediction shape: {pred.shape} (need [S*11,K,T,2])")

S, A, past_len, _ = past_traj.shape
N, K, T, _ = pred.shape
assert N == S * A, f"prediction first dim N={N} must equal num_scenes*actors={S*A}"

# future (oracle) 可选
if USE_ORACLE_BEST:
    fut_traj = torch.load(FUTURE_PATH, map_location='cpu')  # [S,11,T,2]
    assert fut_traj.shape[0] == S and fut_traj.shape[1] == A, "future shape mismatch"
    assert fut_traj.shape[2] == T, f"future_len {fut_traj.shape[2]} must match pred T {T}"

print(f"Scenes={S}, Actors={A}, past_len={past_len}, future_len={T}, K={K}")

# ===================== 检查场景编号是否合法 =====================
valid_target_scenes = []
for s in TARGET_SCENES:
    if 0 <= s < S:
        valid_target_scenes.append(s)
    else:
        print(f"[WARN] scene_idx={s} 超出范围，已跳过。有效范围: [0, {S-1}]")

if len(valid_target_scenes) == 0:
    raise ValueError("TARGET_SCENES 里没有合法的场景编号，请检查。")

print("Will draw scenes:", valid_target_scenes)

# ===================== 预计算：把 pred 转绝对坐标 =====================
# last positions: [S,11,2] -> [N,2]
last_pos = past_traj[:, :, -1, :].contiguous().view(-1, 2)          # [N,2]

# pred_abs: (rel*5 + last_pos) * scale
pred_abs = (pred * 5.0 + last_pos[:, None, None, :]) * scale         # [N,K,T,2]

# GT 绝对坐标（oracle 用）
if USE_ORACLE_BEST:
    gt_abs = (fut_traj.contiguous().view(-1, T, 2)) * scale          # [N,T,2]

# ===================== 选择“最佳轨迹”下标 best_k: [N] =====================
if USE_ORACLE_BEST:
    # 误差: [N,K,T]
    diff = pred_abs - gt_abs[:, None, :, :]                          # [N,K,T,2]
    dist = torch.norm(diff, dim=-1)                                  # [N,K,T]
    if USE_FDE:
        score = dist[:, :, -1]                                       # [N,K]
    else:
        score = dist.mean(dim=-1)                                    # ADE: [N,K]
    best_k = torch.argmin(score, dim=1)                              # [N]
else:
    # medoid：选与其他样本“平均距离最小”的那条（不需要GT）
    mean_traj = pred_abs.mean(dim=1, keepdim=True)                   # [N,1,T,2]
    dist = torch.norm(pred_abs - mean_traj, dim=-1).mean(dim=-1)     # [N,K]
    best_k = torch.argmin(dist, dim=1)                               # [N]

# ===================== 绘图：只画指定场景 =====================
for scene_idx in valid_target_scenes:
    traj_scene = past_traj[scene_idx]  # [11,past,2]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.imshow(court, extent=[x_min, x_max, y_max, y_min], alpha=0.6, zorder=0)

    # ---------- 历史：原始点 + 原始折线 ----------
    last_points = []
    for actor_num in range(A):
        hist_color = get_hist_color(actor_num)
        hist = traj_scene[actor_num].numpy()                         # [past,2]
        x_hist = hist[:, 0] * scale
        y_hist = hist[:, 1] * scale

        # 原始点
        ax.scatter(x_hist, y_hist, color=hist_color, s=15, alpha=0.95, zorder=5)

        # 原始折线（不平滑）
        ax.plot(
            x_hist, y_hist,
            color=hist_color,
            linewidth=2,
            alpha=0.95,
            zorder=4,
            solid_capstyle='round',
            solid_joinstyle='round'
        )

        last_points.append((x_hist[-1], y_hist[-1]))

    # ---------- 预测：best trajectory（原始点 + 原始折线） ----------
    for actor_num in range(A):
        idx = scene_idx * A + actor_num
        k_best = int(best_k[idx].item())

        best_traj = pred_abs[idx, k_best].numpy()                    # [T,2] 已经是 *scale 之后
        x_pred = best_traj[:, 0]
        y_pred = best_traj[:, 1]

        color_pred = get_pred_color(actor_num)

        # 原始预测点
        ax.scatter(x_pred, y_pred, color=color_pred, s=18, alpha=0.95, zorder=11)

        # 将最后一个历史点和未来预测点直接连成折线
        last_x, last_y = last_points[actor_num]
        x_all = np.concatenate([[last_x], x_pred])
        y_all = np.concatenate([[last_y], y_pred])

        ax.plot(
            x_all, y_all,
            color=color_pred,
            linewidth=3,
            alpha=1.0,
            zorder=9,
            solid_capstyle='round',
            solid_joinstyle='round'
        )

        # 终点星
        ax.scatter(x_pred[-1], y_pred[-1], color=color_pred, marker='*', s=90, zorder=12)

    title_best = (
        "OracleBest(ADE)" if (USE_ORACLE_BEST and not USE_FDE)
        else "OracleBest(FDE)" if (USE_ORACLE_BEST and USE_FDE)
        else "MedoidBest(noGT)"
    )

    plt.title(f"Scene {scene_idx}: All Actors - History & BEST Prediction ({title_best})", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"scene_{scene_idx}_besttraj_{title_best}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if SHOW_FIG:
        plt.show()
    plt.close()

print(f"Done. Saved to: {save_dir}")

