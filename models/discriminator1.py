import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTemporalLayer(nn.Module):
    """空间-时间注意力块：包含空间多头注意力、时间多头注意力和前馈网络"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(SpatialTemporalLayer, self).__init__()
        # 多头自注意力层：分别用于空间关系和时间关系
        self.spatial_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 前馈全连接网络两层（隐藏维度dim_feedforward，激活函数ReLU）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # LayerNorm层，用于预归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入 x 维度: (B, T, N, D)
        B = batch size, T = 时间步数, N = 实体数, D = 特征维度(模型隐层维度)
        """
        B, T, N, D = x.shape

        # 1. 空间注意力子层：在每个时间步，对该时间的所有实体执行注意力
        x_res = x  # 残差分支
        # 预LayerNorm归一化
        out = self.norm1(x)
        # 调整形状以应用多头注意力: 将(B, T, N, D)变换为(L=N, N'=B*T, D)
        out_spat = out.contiguous().view(B * T, N, D).permute(1, 0, 2)  # shape: (N, B*T, D)
        # 多头自注意力（键、值、查询都为out_spat本身），仅计算同一时间步(N序列长度)内的相关性
        attn_spat, _ = self.spatial_attn(out_spat, out_spat, out_spat)
        # 将注意力输出变回原形状 (B, T, N, D)
        out_spat = attn_spat.permute(1, 0, 2).view(B, T, N, D)
        # Dropout并加上残差
        x = x_res + self.dropout(out_spat)

        # 2. 时间注意力子层：在每个实体的轨迹序列上执行注意力
        x_res = x  # 残差分支
        out = self.norm2(x)
        # 重排形状: 将(B, T, N, D)变为(L=T, N'=B*N, D)，每个实体（在每个batch）作为一条长度T的序列
        out_temp = out.permute(0, 2, 1, 3).contiguous().view(B * N, T, D).permute(1, 0, 2)  # (T, B*N, D)
        # 多头自注意力：仅计算同一实体轨迹(T序列长度)内的时间相关性
        attn_temp, _ = self.temporal_attn(out_temp, out_temp, out_temp)
        # 恢复回原始形状 (B, T, N, D)
        out_temp = attn_temp.permute(1, 0, 2).view(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
        # Dropout并加残差
        x = x_res + self.dropout(out_temp)

        # 3. 前馈网络子层：逐位置的两层全连接网络
        x_res = x  # 残差分支
        out = self.norm3(x)
        # 全连接层 + ReLU 激活（保持维度不变，逐元素操作）
        ff_hidden = F.relu(self.linear1(out))
        ff_hidden = self.dropout(ff_hidden)
        ff_out = self.linear2(ff_hidden)
        ff_out = self.dropout(ff_out)
        # 加残差
        x = x_res + ff_out

        return x

class TrajectoryDiscriminator1(nn.Module):
    """基于Transformer+GNN的轨迹判别器"""
    def __init__(self, input_dim, model_dim, num_heads=4, num_layers=3,
                 dim_feedforward=128, dropout=0.1, num_entities=None, use_entity_embedding=False, output_dim=1, max_time_steps=500):
        """
        参数:
        - input_dim: 输入特征维度（如每个实体的位置坐标维数）
        - model_dim: 模型隐层维度（Transformer嵌入维度）
        - num_heads: 多头注意力的头数
        - num_layers: 堆叠的空间-时间注意力块层数
        - dim_feedforward: 前馈网络隐藏层维度
        - dropout: Dropout置零比率
        - num_entities: 实体的种类/最大数量（若使用实体嵌入则需提供）
        - use_entity_embedding: 是否使用实体身份嵌入
        - output_dim: 判别器输出维度（默认1表示二分类的logit）
        - max_time_steps: 序列最大时间长度（用于预生成位置编码）
        """
        super(TrajectoryDiscriminator1, self).__init__()
        self.model_dim = model_dim
        # 输入特征投影到模型维度
        self.input_proj = nn.Linear(input_dim, model_dim)
        # 可选：实体嵌入，用于提供每个实体的固定身份表示
        if use_entity_embedding:
            assert num_entities is not None, "使用实体嵌入则需提供num_entities"
            self.entity_emb = nn.Embedding(num_entities, model_dim)
        else:
            self.entity_emb = None
        # 时间位置编码（这里使用正弦位置编码方式）
        pe = torch.zeros(max_time_steps, model_dim)
        position = torch.arange(0, max_time_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)  # shape: (1, max_time_steps, 1, model_dim)
        self.register_buffer('pos_encoding', pe)  # 将位置编码登记为buffer（不作为可训练参数）
        # 堆叠若干层空间-时间注意力块
        self.layers = nn.ModuleList([
            SpatialTemporalLayer(model_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        # 判别器输出层（线性分类层）
        self.classifier = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        """
        前向传播:
        输入 x: 张量，形状 (B, T, N, input_dim)
        输出: 判别结果 (B, output_dim)
        """
        B, T, N, _ = x.shape
        # 1. 输入特征映射到模型维度
        x = self.input_proj(x)  # 得到 (B, T, N, model_dim)
        # 2. 添加实体身份嵌入（如果使用）
        if self.entity_emb is not None:
            # 假设每个序列中的实体索引为 0 ~ N-1
            device = x.device
            # 构造实体索引矩阵 shape: (B, T, N)
            ids = torch.arange(N, device=device).unsqueeze(0).unsqueeze(1).expand(B, T, N)
            # 获取对应的实体embedding并相加
            ent_embed = self.entity_emb(ids)  # (B, T, N, model_dim)
            x = x + ent_embed
        # 3. 添加时间位置编码，以包含顺序信息
        # 从预计算的位置编码buffer中取前T步并广播到(N)实体维度
        pos_enc = self.pos_encoding[:, :T, :, :]  # 形状 (1, T, 1, model_dim)
        x = x + pos_enc  # 广播加到x上 (B,T,N,D)
        # 4. 通过多层空间-时间注意力块编码特征
        for layer in self.layers:
            x = layer(x)
        # 5. 聚合全局特征用于判别输出
        # 我们对时间维度和实体维度求平均 (也可采用其他聚合方式如max或注意力池化)
        # 得到每个序列的全局表示 (B, model_dim)
        global_rep = x.mean(dim=(1, 2))
        # 6. 分类输出
        logits = self.classifier(global_rep)
        # 若为二分类任务，可在训练时对logits应用 sigmoid 激活计算概率
        return logits


