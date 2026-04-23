import torch
import torch.nn as nn

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h

class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='global'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# def make_mlp(dim_list, activation=nn.ReLU, batch_norm=True, dropout=0.0, residual=True):
#     """构造多层感知机（MLP），可选加入残差连接。"""
#     layers = []
#     for i in range(len(dim_list) - 1):
#         in_dim = dim_list[i]
#         out_dim = dim_list[i+1]
#         layers.append(nn.Linear(in_dim, out_dim))
#         # 非最后一层，添加BN、激活和Dropout
#         if i < len(dim_list) - 2:
#             if batch_norm:
#                 layers.append(nn.BatchNorm1d(out_dim))
#             layers.append(activation())
#             if dropout > 0:
#                 layers.append(nn.Dropout(dropout))
#     mlp_seq = nn.Sequential(*layers)
#     if residual:
#         # 改进2: 在MLP中添加残差连接
#         class ResidualMLP(nn.Module):
#             def __init__(self, mlp_seq):
#                 super().__init__()
#                 self.mlp_seq = mlp_seq
#             def forward(self, x):
#                 out = x
#                 prev_input = x  # 保存前一层的输入以供残差连接
#                 for layer in self.mlp_seq:
#                     out = layer(out)
#                     # 在线性层后应用残差连接（如果尺寸匹配）
#                     if isinstance(layer, nn.Linear):
#                         if out.shape == prev_input.shape:
#                             out = out + prev_input
#                         prev_input = out  # 更新残差参考
#                 return out
#         return ResidualMLP(mlp_seq)
#     else:
#         return mlp_seq
#
# class Encoder(nn.Module):
#     """Encoder：使用LSTM提取序列特征，并通过Transformer自注意力层捕获长程依赖。"""
#     def __init__(self, input_dim, embedding_dim, h_dim, num_layers=1, dropout=0.0):
#         super().__init__()
#         self.input_embed = nn.Linear(input_dim, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, h_dim, num_layers, batch_first=True, dropout=dropout)
#         # 改进1: 在Encoder中加入Transformer自注意力层
#         encoder_layer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=4, dropout=dropout, dim_feedforward=h_dim*2)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
#     def forward(self, seq):
#         # seq: (batch, seq_len, input_dim)
#         emb = F.relu(self.input_embed(seq))
#         lstm_out, _ = self.lstm(emb)            # (batch, seq_len, h_dim)
#         # 转换维度以输入TransformerEncoder (seq_len, batch, h_dim)
#         x = lstm_out.transpose(0, 1)
#         x = self.transformer_encoder(x)         # 应用自注意力机制捕获长距离依赖
#         x = x.transpose(0, 1)                   # 转回 (batch, seq_len, h_dim)
#         final_feature = x[:, -1, :]             # 取最后时间步隐藏状态作为特征表示
#         return final_feature
#
# class PoolHiddenNet(nn.Module):
#     """社会池化模块：使用图神经网络 (GNN) 建模多轨迹之间的交互。"""
#     def __init__(self, h_dim, out_dim):
#         super().__init__()
#         # 图卷积层的线性变换（将邻居聚合特征映射到输出维度）
#         self.neighbor_linear = nn.Linear(h_dim, out_dim)
#     def forward(self, h_states, positions=None):
#         """
#         h_states: (N, h_dim) N个行人的隐藏状态
#         positions: (N, 2) 每个行人当前的位置坐标（用于计算邻接矩阵，可选）
#         返回值: agg_feat (N, out_dim) 融合邻居信息后的特征表示
#         """
#         N = h_states.size(0)
#         # 构建邻接矩阵
#         if positions is not None:
#             # 若提供位置，根据距离阈值构建邻接关系
#             dist_matrix = torch.cdist(positions, positions, p=2)            # 计算两两距离
#             adj = (dist_matrix < 2.0).float().to(h_states.device)           # 距离小于阈值视为相邻
#         else:
#             # 未提供位置，则假定完全连通图（所有节点彼此相邻）
#             adj = torch.ones((N, N), device=h_states.device)
#         # 邻接矩阵按行归一化
#         deg = torch.clamp(adj.sum(dim=1, keepdim=True), min=1.0)
#         norm_adj = adj / deg
#         # 聚合邻居特征（包括自身特征）
#         neighbor_agg = norm_adj @ h_states                                   # (N, h_dim)
#         agg_feat = F.relu(self.neighbor_linear(neighbor_agg))                # 线性变换并激活
#         # 改进4: GNN聚合邻居信息，并加入残差连接
#         if agg_feat.shape == h_states.shape:
#             agg_feat = agg_feat + h_states
#         return agg_feat
#
# class TrajectoryDiscriminator(nn.Module):
#     """轨迹判别器：判别输入轨迹是真实数据还是生成数据。"""
#     def __init__(self, seq_len, input_dim=2, embedding_dim=16, h_dim=32, mlp_dim=64, num_layers=1, dropout=0.0, d_type='global'):
#         super().__init__()
#         # 编码器：LSTM + 自注意力Transformer
#         self.encoder = Encoder(input_dim, embedding_dim, h_dim, num_layers, dropout)
#         self.d_type = d_type
#         if d_type == 'global':
#             # 改进4: 使用GNN的 PoolHiddenNet 建模社会交互
#             self.pool_net = PoolHiddenNet(h_dim, h_dim)
#             mlp_input_dim = h_dim + h_dim      # 拼接自身特征和邻居特征作为输入
#             # print("使用了全局！！！")
#         else:
#             self.pool_net = None
#             mlp_input_dim = h_dim
#         # 改进3: 增强判别器的MLP结构（两层隐藏层）并加入残差连接
#         self.real_classifier = make_mlp([mlp_input_dim, mlp_dim, mlp_dim, 1],
#                                         activation=nn.ReLU, batch_norm=True, dropout=dropout, residual=True)
#     def forward(self, traj, positions=None):
#         """
#         traj: (seq_len, batch input_dim) 输入轨迹序列
#         positions: (batch, 2) 每个轨迹最终位 置坐标（用于全局判别的邻接计算，可选）
#         返回: scores (batch, 1) 判别分数（越接近1表示越可能为真实轨迹）
#         """
#         traj = traj.permute(1, 0, 2) #traj: (batch, seq_len, input_dim)
#         # 编码轨迹序列得到隐藏特征
#         final_h = self.encoder(traj)  # (batch, h_dim)
#         if self.d_type == 'global' and self.pool_net is not None:
#             # 全局判别模式：融合邻居轨迹信息
#             pooled_feat = self.pool_net(final_h, positions) if positions is not None else self.pool_net(final_h)
#             final_feat = torch.cat([final_h, pooled_feat], dim=-1)  # 拼接自身特征和邻居特征
#             # print("使用了全局！！！")
#         else:
#             final_feat = final_h
#         # 通过MLP分类器输出判别分数
#         scores = self.real_classifier(final_feat)  # (batch, 1)
#         return scores

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# def make_mlp(dim_list, activation=nn.ReLU, batch_norm=True, dropout=0.0, residual=False):
#     """构造一个多层感知器（MLP），支持可选的残差连接"""
#     layers = []
#     for i in range(len(dim_list) - 1):
#         in_dim = dim_list[i]
#         out_dim = dim_list[i + 1]
#         layers.append(nn.Linear(in_dim, out_dim))
#         if i < len(dim_list) - 2:
#             if batch_norm:
#                 layers.append(nn.BatchNorm1d(out_dim))
#             layers.append(activation())
#             if dropout > 0:
#                 layers.append(nn.Dropout(dropout))
#     mlp_seq = nn.Sequential(*layers)
#     if residual:
#         class ResidualMLP(nn.Module):
#             def __init__(self, mlp_seq):
#                 super().__init__()
#                 self.mlp_seq = mlp_seq
#             def forward(self, x):
#                 out = x
#                 for layer in self.mlp_seq:
#                     out = layer(out)
#                     if isinstance(layer, nn.Linear) and out.shape == x.shape:
#                         out = out + x
#                         x = out
#                 return out
#         return ResidualMLP(mlp_seq)
#     return mlp_seq
#
# class Encoder(nn.Module):
#     """使用LSTM和Transformer的编码器，用于提取序列特征"""
#     def __init__(self, input_dim, embedding_dim, h_dim, num_layers=1, dropout=0.0):
#         super().__init__()
#         self.input_embed = nn.Linear(input_dim, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, h_dim, num_layers, batch_first=True, dropout=dropout)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=4, dropout=dropout, dim_feedforward=h_dim*2)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
#     def forward(self, seq):
#         emb = F.relu(self.input_embed(seq))
#         lstm_out, _ = self.lstm(emb)
#         x = lstm_out.transpose(0, 1)  # (batch, seq_len, h_dim) -> (seq_len, batch, h_dim)
#         x = self.transformer_encoder(x)
#         x = x.transpose(0, 1)  # (seq_len, batch, h_dim) -> (batch, seq_len, h_dim)
#         final_feature = x[:, -1, :]  # 取最后一个时间步的隐藏状态
#         return final_feature
#
# class PoolHiddenNet(nn.Module):
#     """社会池化模块，基于相对位置嵌入和最大池化"""
#     def __init__(self, embedding_dim, h_dim, mlp_dim, bottleneck_dim, dropout=0.0):
#         super().__init__()
#         self.spatial_embedding = nn.Linear(2, embedding_dim)
#         mlp_pre_pool_dims = [embedding_dim + h_dim, mlp_dim, bottleneck_dim]
#         self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=nn.ReLU, batch_norm=True, dropout=dropout)
#     def forward(self, h_states, positions, seq_start_end):
#         pool_h = []
#         for start, end in seq_start_end:
#             num_ped = end - start
#             curr_h_states = h_states[start:end]
#             curr_positions = positions[start:end]
#             # 计算相对位置
#             rel_pos = curr_positions.unsqueeze(1) - curr_positions.unsqueeze(0)  # (num_ped, num_ped, 2)
#             rel_pos_embedding = self.spatial_embedding(rel_pos.view(-1, 2)).view(num_ped, num_ped, -1)
#             # 将h_states重复以匹配每个对
#             h_states_repeated = curr_h_states.unsqueeze(0).repeat(num_ped, 1, 1)  # (num_ped, num_ped, h_dim)
#             # print(rel_pos_embedding.shape, h_states_repeated.shape)
#             # 拼接相对位置嵌入和h_states
#             mlp_input = torch.cat([rel_pos_embedding, h_states_repeated], dim=-1)
#             # mlp_input.shape = [11, 11, 48]
#             mlp_input = mlp_input.view(-1, mlp_input.shape[-1])  # => [11*11, 48] = [121, 48] (num_ped * num_ped, embedding_dim + h_dim)
#
#             # print(mlp_input.shape) #
#             curr_pool_h = self.mlp_pre_pool(mlp_input)
#             curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]  # 最大池化
#             pool_h.append(curr_pool_h)
#         pool_h = torch.cat(pool_h, dim=0)
#         return pool_h
#
# class TrajectoryDiscriminator(nn.Module):
#     """优化后的轨迹判别器，适用于行人轨迹预测"""
#     def __init__(self, seq_len, input_dim=2, embedding_dim=16, h_dim=32, mlp_dim=64, num_layers=1, dropout=0.0, d_type='global'):
#         super().__init__()
#         self.encoder = Encoder(input_dim, embedding_dim, h_dim, num_layers, dropout)
#         self.d_type = d_type
#         if d_type == 'global':
#             self.pool_net = PoolHiddenNet(embedding_dim, h_dim, mlp_dim, h_dim, dropout)
#             mlp_input_dim = h_dim + h_dim  # 拼接编码器输出和池化特征
#         else:
#             self.pool_net = None
#             mlp_input_dim = h_dim
#         self.real_classifier = make_mlp([mlp_input_dim, mlp_dim, mlp_dim, 1],
#                                         activation=nn.ReLU, batch_norm=True, dropout=dropout, residual=True)
#     def forward(self, traj, traj_rel, seq_start_end=None):
#         traj = traj.permute(1, 0, 2)  # (seq_len, batch, input_dim) -> (batch, seq_len, input_dim)
#         traj_rel = traj_rel.permute(1, 0, 2)  # 调整为batch_first
#         final_h = self.encoder(traj_rel)  # (batch, h_dim)
#         if self.d_type == 'global' and self.pool_net is not None:
#             assert seq_start_end is not None, "全局判别器必须提供seq_start_end"
#             positions = traj[:, -1, :]  # 取最后一个时间步的位置
#             pooled_feat = self.pool_net(final_h, positions, seq_start_end)  # (batch, h_dim)
#             final_feat = torch.cat([final_h, pooled_feat], dim=-1)  # (batch, 2 * h_dim)
#             # print("执行了全局的!!!")
#         else:
#             final_feat = final_h
#         scores = self.real_classifier(final_feat)  # (batch, 1)
#         return scores
