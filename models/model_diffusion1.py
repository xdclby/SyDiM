import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class STEncoder(nn.Module):
    def __init__(self, channel_in=2, channels=(32, 64), embedding_dim=256):
        super().__init__()

        self.conv1 = nn.Conv1d(channel_in, channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels[0])

        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels[1])

        self.relu = nn.ReLU(inplace=True)

        self.temporal_encoder = nn.GRU(
            input_size=channels[1],
            hidden_size=embedding_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: [B, T_obs, 2]
        return:
            x_st: [B, T_obs, d_m]
        """
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)

        x_st, _ = self.temporal_encoder(x)
        return x_st


def build_all_social_features(obs_traj, neighbor_mask=None, eps=1e-6):
    """
    obs_traj: [B, N, T_obs, 2] in a shared/global coordinate frame.
    return:   [B, N, T_obs, 6]
    """
    B, N, T_obs, _ = obs_traj.shape
    device = obs_traj.device
    dtype = obs_traj.dtype

    vel = torch.zeros_like(obs_traj)
    vel[:, :, 1:, :] = obs_traj[:, :, 1:, :] - obs_traj[:, :, :-1, :]

    x_i = obs_traj.unsqueeze(2)
    x_j = obs_traj.unsqueeze(1)

    v_i = vel.unsqueeze(2)
    v_j = vel.unsqueeze(1)

    r_ij = x_j - x_i
    u_ij = v_j - v_i

    dist_r = torch.norm(r_ij, dim=-1, keepdim=True)
    dist_u = torch.norm(u_ij, dim=-1, keepdim=True)

    g_ij = torch.cat([r_ij, u_ij, dist_r, dist_u], dim=-1)

    if neighbor_mask is None:
        mask = torch.ones(B, N, N, device=device, dtype=dtype)
        eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
        mask = mask - eye
    else:
        mask = neighbor_mask.to(dtype)
        eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
        mask = mask * (1.0 - eye)

    mask = mask.unsqueeze(-1).unsqueeze(-1)

    num = (g_ij * mask).sum(dim=2)
    den = mask.sum(dim=2).clamp_min(eps)
    pooled = num / den
    return pooled


class SocialTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=256, nhead=4, num_layers=3, max_len=64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, d_model),
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=max_len)
        self.learnable_pos = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.context_adapter = nn.Linear(d_model, d_model)

    def forward(self, src, src_mask=None):
        """
        src: [B, T_obs, 6]
        src_mask:
            - None
            - [B, T_obs] as padding mask
            - [T_obs, T_obs] as attention mask
        return:
            [B, d_model]
        """
        padding_mask = None
        attn_mask = None

        if src_mask is not None:
            if src_mask.dim() == 2 and src_mask.shape[0] == src_mask.shape[1]:
                attn_mask = src_mask
            else:
                padding_mask = src_mask if src_mask.dtype == torch.bool else (src_mask == 0)

        x = self.embed(src)
        B, T_obs, _ = x.shape

        x = self.pos_encoder(x)
        x = x + self.learnable_pos[:, :T_obs, :]

        x = x.permute(1, 0, 2)
        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)

        out = x.mean(dim=1)
        return self.context_adapter(out)


class CrossContextFusionTransformer(nn.Module):
    def __init__(self, context_dim=256, nheads=4, max_len=64):
        super().__init__()

        self.st_proj = nn.Linear(context_dim, context_dim)
        self.st_ln = nn.LayerNorm(context_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=context_dim, num_heads=nheads)

        self.fuse = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.ReLU(inplace=True),
        )

        self.pos_embed = nn.Embedding(max_len, context_dim)
        self.raw_proj = nn.Linear(2, context_dim)

    def forward(self, x_st, x_context, raw_xy):
        """
        x_st: [B, T_obs, d_m]
        x_context: [B, d_m]
        raw_xy: [B, T_obs, 2]
        return:
            h_out: [B, T_obs, d_m]
        """
        B, T_obs, _ = x_st.shape

        q = self.st_ln(self.st_proj(x_st))
        kv = x_context.unsqueeze(1)

        q = q.permute(1, 0, 2)
        kv = kv.permute(1, 0, 2)

        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = attn_out.permute(1, 0, 2)

        x_fusion = self.fuse(torch.cat([x_st, attn_out], dim=-1))

        pos_ids = torch.arange(T_obs, device=x_st.device).unsqueeze(0)
        pos = self.pos_embed(pos_ids).expand(B, -1, -1)

        raw_res = self.raw_proj(raw_xy)

        h_out = x_fusion + pos + raw_res
        return h_out


class TransformerDenoisingModel(nn.Module):
    def __init__(self, context_dim=256, tf_layers=4, needs_legacy=False, max_seq_len=64):
        super().__init__()
        self.context_dim = context_dim
        self.max_seq_len = max_seq_len

        self.st_encoder = STEncoder(embedding_dim=context_dim)
        self.social_encoder = SocialTransformer(
            input_dim=6,
            d_model=context_dim,
            nhead=4,
            num_layers=3,
            max_len=max_seq_len,
        )
        self.context_fusion = CrossContextFusionTransformer(
            context_dim=context_dim,
            nheads=4,
            max_len=max_seq_len,
        )

        self.time_embed = nn.Sequential(
            nn.Linear(2, context_dim),
            nn.ReLU(inplace=True),
        )
        self.legacy_adapter = nn.Linear(2, context_dim) if needs_legacy else None

        self.beta_embed = nn.Sequential(
            nn.Linear(3, context_dim),
            nn.LayerNorm(context_dim),
        )
        self.cond_proj = nn.Linear(context_dim, context_dim)
        self.pos_embed = nn.Embedding(max_seq_len, context_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=4,
            dim_feedforward=context_dim * 4,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        self.output_net = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim // 2, 2),
        )

        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def load_state_dict(self, state_dict, strict=True):
        own_state = super().state_dict()
        compatible_keys = [
            key for key, value in state_dict.items()
            if key in own_state and own_state[key].shape == value.shape
        ]
        legacy_keys = (
            "encoder_context.",
            "concat1.",
            "concat3.",
            "concat4.",
            "linear.",
            "transformer_encoder.",
            "pos_emb.",
        )

        if compatible_keys and "output_scale" not in state_dict:
            compatible_ratio = len(compatible_keys) / max(len(own_state), 1)
            if compatible_ratio > 0.5:
                self.output_scale.data.fill_(1.0)

        if not compatible_keys:
            if any(key.startswith(legacy_keys) for key in state_dict):
                warnings.warn(
                    "Detected a legacy model_diffusion checkpoint. "
                    "Its generator weights are not structurally compatible with "
                    "model_diffusion1, so the denoiser would otherwise stay random. "
                    "The model keeps a small output_scale for stability; for best "
                    "results, retrain model_diffusion1 or convert the checkpoint.",
                    RuntimeWarning,
                )
            else:
                warnings.warn(
                    "No compatible parameters were found for model_diffusion1; "
                    "the denoiser is staying near its randomly initialized state.",
                    RuntimeWarning,
                )

        return super().load_state_dict(state_dict, strict=strict)

    def _align_tokens(self, tokens, target_len):
        if tokens.size(1) == target_len:
            return tokens

        x = tokens.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        x = x.transpose(1, 2)
        return x

    def _encode_context(self, context, mask=None):
        if context.dim() == 3:
            if context.size(-1) == 6:
                return self.social_encoder(context, src_mask=mask)
            if context.size(-1) == self.context_dim:
                return context.mean(dim=1)
        elif context.dim() == 2:
            return context
        raise ValueError(f"Unexpected context shape: {tuple(context.shape)}")

    def forward(self, x, beta, context, mask=None, hist=None):
        """
        x: [B, T_pred, 2]
        beta: [B] or [B,1,1]
        context:
            - [B, T_obs, 6]  : social feature sequence
            - [B, d_m]       : pre-encoded global context
        hist:
            - [B, T_obs, 2]  : observed history (required)
        """
        if hist is None:
            raise ValueError("hist must be provided for the new denoiser.")

        B, T_pred, _ = x.shape
        hist_xy = hist

        x_st = self.st_encoder(hist_xy)
        context_feat = self._encode_context(context, mask=mask)

        h_out = self.context_fusion(x_st=x_st, x_context=context_feat, raw_xy=hist_xy)
        h_cond = self._align_tokens(h_out, T_pred)
        h_cond = self.cond_proj(h_cond)

        if self.legacy_adapter is not None:
            u_t = self.legacy_adapter(x)
        else:
            u_t = self.time_embed(x)

        beta = beta.view(B, 1, 1)
        beta_feat = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1).expand(-1, T_pred, -1)
        beta_emb = self.beta_embed(beta_feat)

        pos_ids = torch.arange(T_pred, device=x.device).unsqueeze(0)
        pos = self.pos_embed(pos_ids).expand(B, -1, -1)

        trans_in = u_t + h_cond + beta_emb + pos
        trans_in = trans_in.permute(1, 0, 2)
        trans_out = self.transformer(trans_in)
        trans_out = trans_out.permute(1, 0, 2)

        out = self.output_net(trans_out) * self.output_scale
        return out

    def generate_accelerate(self, x, beta, context, mask=None, hist=None):
        """
        Compatible with the original trainer.

        x: [N, K, T_pred, 2]
            N = agents in the current flattened scene batch
            K = proposal count

        context:
            - [N, T_obs, 6]
            - [N, d_model]

        hist:
            - [N, T_obs, 2]
        """
        if hist is None:
            raise ValueError("hist must be provided in generate_accelerate().")

        N, K, T_pred, _ = x.shape
        x_flat = x.reshape(N * K, T_pred, 2)

        if beta.dim() == 0:
            beta_expand = beta.view(1).expand(N * K)
        else:
            beta = beta.reshape(N, -1)[:, 0]
            beta_expand = beta.unsqueeze(1).expand(N, K).reshape(-1)

        if context.dim() == 3 and context.size(-1) == 6:
            ctx_feat = self.social_encoder(context, src_mask=None)
        elif context.dim() == 2:
            ctx_feat = context
        elif context.dim() == 3 and context.size(-1) == self.context_dim:
            ctx_feat = context.mean(dim=1)
        else:
            raise ValueError(f"Unexpected context shape for accelerate: {tuple(context.shape)}")
        ctx_feat = ctx_feat.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)

        if hist.dim() != 3:
            raise ValueError(f"hist should be [N, T_obs, 2], got {tuple(hist.shape)}")
        hist_flat = hist.unsqueeze(1).expand(N, K, -1, -1).reshape(N * K, hist.size(1), 2)

        out_flat = self.forward(
            x_flat,
            beta_expand,
            ctx_feat,
            mask=None,
            hist=hist_flat,
        )
        return out_flat.reshape(N, K, T_pred, 2)
