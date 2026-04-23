import os
import matplotlib.pyplot as plt
import torch
import numpy as np

# 确保保存文件夹存在
output_dir = './past'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 只加载历史轨迹数据
past_traj = torch.load('data/past1.pt', map_location=torch.device('cpu'))
# 动态获取帧数和实体数量
num_entities = past_traj.size(1)
past_frames = past_traj.size(2)

# 对历史轨迹数据进行缩放（与原代码保持一致）
traj = past_traj * 94 / 28

court = plt.imread("court.png")
# mask = np.zeros_like(court)


class Constant:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 6
    X_MIN = 0
    X_MAX = 100
    Y_MIN = 0
    Y_MAX = 50
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = X_MAX / 2 - DIFF / 1.5 + 0.10
    Y_CENTER = Y_MAX - DIFF / 1.5 - 0.35


# 根据数据情况，调整选取场景和主体的索引（示例）
idx_list = [[0, 0], [12, 1], [3, 0], [5, 0]]

for i in idx_list:
    scene = i[0]
    actor_num = i[1]

    plt.clf()

    ax = plt.axes(xlim=(Constant.X_MIN, Constant.X_MAX),
                  ylim=(Constant.Y_MIN, Constant.Y_MAX))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)  # 去除网格

    # 注意：全局索引计算基于实体数量
    idx = scene * num_entities + actor_num

    colorteam1 = 'dodgerblue'
    colorteam2 = 'dodgerblue'
    colorteam1_pre = 'skyblue'
    colorteam2_pre = 'skyblue'

    # 绘制每个背景球员的历史轨迹
    for actor_num_other in range(num_entities):
        zorder = 5
        if actor_num_other == actor_num:
            zorder = 105
        traj_curr_ = traj[scene, actor_num_other].numpy()
        # 根据实体编号选择颜色
        if actor_num_other < num_entities // 2:
            color_pre = colorteam1_pre
        else:
            color_pre = colorteam2_pre

        # 绘制散点（历史轨迹点）
        for j in range(past_frames):
            (x, y) = (traj_curr_[j, 0], traj_curr_[j, 1])
            plt.scatter(x, y, color=color_pre, s=15, alpha=1, zorder=zorder)

        # 绘制历史轨迹的连接线
        for j in range(past_frames - 1):
            (x_vals, y_vals) = ([traj_curr_[j, 0], traj_curr_[j + 1, 0]],
                                [traj_curr_[j, 1], traj_curr_[j + 1, 1]])
            plt.plot(x_vals, y_vals, color=color_pre, alpha=0.5, linewidth=2, zorder=zorder - 1)

    # 不再绘制预测或未来轨迹，只保留历史轨迹

    plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                        Constant.Y_MAX, Constant.Y_MIN], alpha=0.5)
    # plt.imshow(mask, zorder=90, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
    #                                     Constant.Y_MAX, Constant.Y_MIN], alpha=0.7)

    # 保存图像到指定目录，文件名格式为 scene_场景编号_agent_球员编号.jpg
    save_path = os.path.join(output_dir, 'scene_{}_agent_{}.jpg'.format(scene, actor_num))
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()
