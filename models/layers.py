import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear

"""
	用于diffusion模块
	这段代码实现了一个基于位置编码 (Positional Encoding) 的类 PositionalEncoding，
	用于为输入序列数据引入位置信息。位置编码是 Transformer 模型中的一个重要部分，它为输入序列的每个时间步添加位置信息，
	以便网络能够区分不同时间步的顺序和位置。
"""
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
		)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer("pe", pe)

	def forward(self, x):
		# print(f"x shape: {x.shape}")
		# print(f"self.pe shape: {self.pe.shape}")
		x = x + self.pe[: x.size(0), :]
		return self.dropout(x)

"""
	用于diffusion模块
	类 ConcatSquashLinear
	这个类继承自 torch.nn.Module，包含一个线性层和两个依赖上下文的辅助线性层，
	能够在计算时灵活地结合输入和上下文信息。
"""
class ConcatSquashLinear(Module):
	def __init__(self, dim_in, dim_out, dim_ctx):
		super(ConcatSquashLinear, self).__init__()
		self._layer = Linear(dim_in, dim_out)
		self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
		self._hyper_gate = Linear(dim_ctx, dim_out)

	def forward(self, ctx, x):
		# ctx: (B, 1, F+3)
		# x: (B, T, 2)
		gate = torch.sigmoid(self._hyper_gate(ctx))
		bias = self._hyper_bias(ctx)
		# if x.dim() == 3:
		#     gate = gate.unsqueeze(1)
		#     bias = bias.unsqueeze(1)
		ret = self._layer(x) * gate + bias
		return ret
	
	def batch_generate(self, ctx, x):
		# ctx: (B, n, 1, F+3)
		# x: (B, n, T, 2)
		gate = torch.sigmoid(self._hyper_gate(ctx))
		bias = self._hyper_bias(ctx)
		# if x.dim() == 3:
		#     gate = gate.unsqueeze(1)
		#     bias = bias.unsqueeze(1)
		ret = self._layer(x) * gate + bias
		return ret
	

"""
	仅用于此类当中
	GAT 类
	GAT 类实现了多头图注意力网络层。GAT 是一种图神经网络，通过自注意力机制为图中每个节点分配注意力权重，
	能够更灵活地学习图结构中节点之间的相互关系。
"""
class GAT(nn.Module):
	def __init__(self, in_feat=2, out_feat=64, n_head=4, dropout=0.1, skip=True):
		super(GAT, self).__init__()
		self.in_feat = in_feat
		self.out_feat = out_feat
		self.n_head = n_head
		self.skip = skip
		self.w = nn.Parameter(torch.Tensor(n_head, in_feat, out_feat))
		self.a_src = nn.Parameter(torch.Tensor(n_head, out_feat, 1))
		self.a_dst = nn.Parameter(torch.Tensor(n_head, out_feat, 1))
		self.bias = nn.Parameter(torch.Tensor(out_feat))

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
		self.softmax = nn.Softmax(dim=-1)
		self.dropout = nn.Dropout(dropout)

		nn.init.xavier_uniform_(self.w, gain=1.414)
		nn.init.xavier_uniform_(self.a_src, gain=1.414)
		nn.init.xavier_uniform_(self.a_dst, gain=1.414)
		nn.init.constant_(self.bias, 0)

	def forward(self, h, mask):
		h_prime = h.unsqueeze(1) @ self.w
		attn_src = h_prime @ self.a_src
		attn_dst = h_prime @ self.a_dst
		attn = attn_src @ attn_dst.permute(0, 1, 3, 2)
		attn = self.leaky_relu(attn)
		attn = self.softmax(attn)
		attn = self.dropout(attn)
		attn = attn * mask if mask is not None else attn
		out = (attn @ h_prime).sum(dim=1) + self.bias
		if self.skip:
			out += h_prime.sum(dim=1)
		return out, attn


class MLP(nn.Module):
	def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), activation=None, dropout=-1):
		super(MLP, self).__init__()
		dims = (in_feat, ) + hid_feat + (out_feat, )

		self.layers = nn.ModuleList()
		for i in range(len(dims) - 1):
			self.layers.append(nn.Linear(dims[i], dims[i + 1]))

		self.activation = activation if activation is not None else lambda x: x
		self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.activation(x)
			x = self.dropout(x)
			x = self.layers[i](x)
		return x

#用于led模块
class social_transformer(nn.Module):
	def __init__(self, past_len):
		super(social_transformer, self).__init__()
		self.encode_past = nn.Linear(past_len*6, 256, bias=False)
		self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

	def forward(self, h, mask):
		'''
		h: batch_size, t, 2
		'''
		# print(f"h shape: {h.shape}")

		h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
		# print(h_feat.shape)
		# n_samples, 1, 64
		h_feat_ = self.transformer_encoder(h_feat, mask)
		h_feat = h_feat + h_feat_

		return h_feat

"""
	用于LED模块
	该类 st_encoder 用于将一个时间序列输入转换为一个固定长度的向量（即特征嵌入），可以用于后续的任务，如分类、回归或预测。
	主要组成部分：
		空间特征提取： 通过 Conv1d 卷积层提取空间特征，卷积处理特征维度中的信息。
		时间特征提取： 通过 GRU 处理时间维度中的信息，得到输入序列的时间嵌入表示。
		最终嵌入表示： 返回序列编码后的隐藏状态，用于后续模型或任务。
"""
class st_encoder(nn.Module):
	def __init__(self):
		super().__init__()
		channel_in = 6
		channel_out = 32
		dim_kernel = 3
		self.dim_embedding_key = 256
		self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
		self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

		self.relu = nn.ReLU()

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_normal_(self.spatial_conv.weight)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
		nn.init.zeros_(self.spatial_conv.bias)
		nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
		nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

	def forward(self, X):
		'''
		X: b, T, 2

		return: b, F
		'''
		X_t = torch.transpose(X, 1, 2)
		X_after_spatial = self.relu(self.spatial_conv(X_t))
		X_embed = torch.transpose(X_after_spatial, 1, 2)

		output_x, state_x = self.temporal_encoder(X_embed)
		state_x = state_x.squeeze(0)

		return state_x

