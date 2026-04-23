import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config
from utils.utils import print_log, abs_to_relative, convert_to_4d
from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate

from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion1 import TransformerDenoisingModel as CoreDenoisingModel
from models.model_diffusion1 import build_all_social_features
from models.discriminator1 import TrajectoryDiscriminator1

try:
    from models.compute_loss import gan_d_loss as imported_gan_d_loss
except Exception:
    imported_gan_d_loss = None

NUM_Tau = 15


class Trainer:
    def __init__(self, config):
        if torch.cuda.is_available():
            torch.cuda.set_device(config.gpu)
        self.device = torch.device('cuda') if config.cuda else torch.device('cpu')
        self.cfg = Config(config.cfg, config.info)

        train_dset = NBADataset(
            obs_len=self.cfg.past_frames,
            pred_len=self.cfg.future_frames,
            training=True,
        )
        self.train_loader = DataLoader(
            train_dset,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=seq_collate,
            pin_memory=True,
        )

        test_dset = NBADataset(
            obs_len=self.cfg.past_frames,
            pred_len=self.cfg.future_frames,
            training=False,
        )
        self.test_loader = DataLoader(
            test_dset,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=seq_collate,
            pin_memory=True,
        )

        self.traj_mean = torch.FloatTensor(self.cfg.traj_mean).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.traj_scale = self.cfg.traj_scale

        self.n_steps = self.cfg.diffusion.steps
        self.betas = self.make_beta_schedule(
            schedule=self.cfg.diffusion.beta_schedule,
            n_timesteps=self.n_steps,
            start=self.cfg.diffusion.beta_start,
            end=self.cfg.diffusion.beta_end,
        ).to(self.device)

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        self.model = CoreDenoisingModel().to(self.device)
        self.discriminator = TrajectoryDiscriminator1(
            input_dim=4,
            model_dim=128,
            num_heads=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1,
            num_entities=11,
            use_entity_embedding=True,
            output_dim=1,
        ).to(self.device)
        self.model_initializer = InitializationModel(t_h=10, d_h=6, t_f=20, d_f=2, k_pred=20).to(self.device)

        if hasattr(self.cfg, 'pretrained_core_denoising_model') and self.cfg.pretrained_core_denoising_model:
            model_cp = torch.load(self.cfg.pretrained_core_denoising_model, map_location='cpu')
            if 'generator_dict' in model_cp:
                self.model.load_state_dict(model_cp['generator_dict'], strict=False)
            if 'model_initializer_dict' in model_cp:
                self.model_initializer.load_state_dict(model_cp['model_initializer_dict'])
            if 'discriminator_dict' in model_cp:
                self.discriminator.load_state_dict(model_cp['discriminator_dict'])
        else:
            self.initialize_weights()

        self.opt = torch.optim.AdamW(self.model_initializer.parameters(), lr=config.learning_rate)
        self.optimizer_G = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate * 0.5)
        self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters(), lr=config.learning_rate)

        self.scheduler_model = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma
        )
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma
        )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_D, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma
        )

        self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
        self.print_model_param(self.discriminator, name='discriminator')
        self.print_model_param(self.model, name='Core Denoising Model')
        self.print_model_param(self.model_initializer, name='Initialization Model')

        # loss settings
        self.temporal_reweight = (
            torch.FloatTensor([self.cfg.future_frames + 1 - i for i in range(1, self.cfg.future_frames + 1)])
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            / 10.0
        )
        self.lambda_dist = float(getattr(self.cfg, 'lambda_dist', 50.0))
        self.lambda_uncertainty = float(getattr(self.cfg, 'lambda_uncertainty', 1.0))
        self.reweight_temperature = float(getattr(self.cfg, 'reweight_temperature', 0.5))
        self.num_discriminator_steps = int(getattr(self.cfg, 'num_discriminator_steps', 1))

    # ---------------------------------------------------------------------
    # basic utils
    # ---------------------------------------------------------------------
    def initialize_weights(self):
        for m in self.discriminator.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def print_model_param(self, model: nn.Module, name: str = 'Model') -> None:
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_log(f'[{name}] Trainable/Total: {trainable_num}/{total_num}', self.log)

    def make_beta_schedule(self, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == 'quad':
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == 'sigmoid':
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        else:
            raise ValueError(f'Unknown beta schedule: {schedule}')
        return betas

    def extract(self, arr, t, x):
        out = torch.gather(arr, 0, t.to(arr.device))
        reshape = [t.shape[0]] + [1] * (len(x.shape) - 1)
        return out.reshape(*reshape)

    def _gan_d_loss(self, scores_real, scores_fake):
        scores_real = scores_real.reshape(-1)
        scores_fake = scores_fake.reshape(-1)
        if imported_gan_d_loss is not None:
            try:
                return imported_gan_d_loss(scores_real, scores_fake)
            except Exception:
                pass
        loss_real = F.binary_cross_entropy_with_logits(scores_real, torch.ones_like(scores_real))
        loss_fake = F.binary_cross_entropy_with_logits(scores_fake, torch.zeros_like(scores_fake))
        return loss_real + loss_fake

    def _reshape_agent_proposal(self, x, batch_size, num_agents):
        # [B*N, K] -> [B, N, K]
        return x.view(batch_size, num_agents, -1)

    def _scores_to_weights(self, proposal_scores):
        # stop-gradient reweighting, as in paper-style proposal weighting
        return torch.softmax(proposal_scores.detach() / self.reweight_temperature, dim=1)

    # ---------------------------------------------------------------------
    # data preprocess
    # ---------------------------------------------------------------------
    def data_preprocess(self, data):
        batch_size = data['pre_motion_3D'].shape[0]
        num_agents = data['pre_motion_3D'].shape[1]

        traj_mask = torch.zeros(batch_size * num_agents, batch_size * num_agents, device=self.device)
        for i in range(batch_size):
            traj_mask[i * num_agents:(i + 1) * num_agents, i * num_agents:(i + 1) * num_agents] = 1.0

        initial_pos = data['pre_motion_3D'].to(self.device)[:, :, -1:]
        past_traj_abs = ((data['pre_motion_3D'].to(self.device) - self.traj_mean) / self.traj_scale).contiguous().view(-1, self.cfg.past_frames, 2)
        past_traj_rel = ((data['pre_motion_3D'].to(self.device) - initial_pos) / self.traj_scale).contiguous().view(-1, self.cfg.past_frames, 2)
        past_traj_vel = torch.cat(
            (past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1
        )
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

        fut_traj = ((data['fut_motion_3D'].to(self.device) - initial_pos) / self.traj_scale).contiguous().view(-1, self.cfg.future_frames, 2)
        return batch_size, num_agents, traj_mask, past_traj, fut_traj

    def data_preprocess_denoiser(self, data):
        """
        New denoiser input:
            hist_local_scene : [B, N, T_obs, 2]
            social_scene     : [B, N, T_obs, 6]
            hist_local_flat  : [B*N, T_obs, 2]
            social_flat      : [B*N, T_obs, 6]
        """
        obs = data['pre_motion_3D'].to(self.device)
        B, N, T_obs, _ = obs.shape

        last_pos = obs[:, :, -1:, :]
        hist_local_scene = (obs - last_pos) / self.traj_scale

        obs_global_scene = (obs - self.traj_mean) / self.traj_scale
        social_scene = build_all_social_features(obs_global_scene)

        hist_local_flat = hist_local_scene.reshape(B * N, T_obs, 2)
        social_flat = social_scene.reshape(B * N, T_obs, 6)
        return hist_local_scene, social_scene, hist_local_flat, social_flat

    def data_preprocess_1(self, data):
        batch_size = data['pre_motion_3D'].shape[0]
        num_agents = data['pre_motion_3D'].shape[1]

        seq_start_end = torch.LongTensor([
            [i * num_agents, (i + 1) * num_agents]
            for i in range(batch_size)
        ]).to(self.device)

        past_traj_abs = (data['pre_motion_3D'].to(self.device) - self.traj_mean) / self.traj_scale
        fut_traj_abs = (data['fut_motion_3D'].to(self.device) - self.traj_mean) / self.traj_scale

        def get_relative(traj):
            rel = traj[:, :, 1:] - traj[:, :, :-1]
            rel = torch.cat([torch.zeros_like(traj[:, :, :1]), rel], dim=2)
            return rel / self.traj_scale

        past_traj_rel = get_relative(data['pre_motion_3D'].to(self.device))
        fut_traj_rel = get_relative(data['fut_motion_3D'].to(self.device))

        def reshape_permute(tensor, seq_len):
            return tensor.view(-1, seq_len, 2).permute(1, 0, 2)

        obs_traj = reshape_permute(past_traj_abs, self.cfg.past_frames)
        pred_traj = reshape_permute(fut_traj_abs, self.cfg.future_frames)
        obs_traj_rel = reshape_permute(past_traj_rel, self.cfg.past_frames)
        pred_traj_rel = reshape_permute(fut_traj_rel, self.cfg.future_frames)

        return obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, seq_start_end

    # ---------------------------------------------------------------------
    # diffusion forward / reverse
    # ---------------------------------------------------------------------
    def noise_estimation_loss(self, social_flat, hist_flat, y_0):
        batch_size = y_0.shape[0]
        t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,), device=y_0.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]

        a = self.extract(self.alphas_bar_sqrt, t, y_0)
        beta = self.extract(self.betas, t, y_0)
        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
        e = torch.randn_like(y_0)
        y = y_0 * a + e * am1
        output = self.model(y, beta, social_flat, mask=None, hist=hist_flat)
        return (e - output).square().mean()

    def p_sample(self, social_flat, hist_flat, cur_y, t):
        z = torch.zeros_like(cur_y) if t == 0 else torch.randn_like(cur_y)

        t_tensor = torch.full((cur_y.size(0),), t, device=cur_y.device, dtype=torch.long)
        eps_factor = (
            (1 - self.extract(self.alphas, t_tensor, cur_y))
            / self.extract(self.one_minus_alphas_bar_sqrt, t_tensor, cur_y)
        )
        beta = self.extract(self.betas, t_tensor, cur_y)
        eps_theta = self.model(cur_y, beta, social_flat, mask=None, hist=hist_flat)
        mean = (1 / self.extract(self.alphas, t_tensor, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        sigma_t = self.extract(self.betas, t_tensor, cur_y).sqrt()
        sample = mean + sigma_t * z
        return sample

    def p_sample_accelerate(self, social_ctx, hist_ctx, cur_y, t):
        z = torch.zeros_like(cur_y) if t == 0 else torch.randn_like(cur_y)

        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        alpha_bar_gap_t = self.one_minus_alphas_bar_sqrt[t]
        eps_factor = (1 - alpha_t) / alpha_bar_gap_t

        beta_batch = beta_t.expand(cur_y.size(0))
        eps_theta = self.model.generate_accelerate(
            cur_y,
            beta_batch,
            social_ctx,
            mask=None,
            hist=hist_ctx,
        )
        mean = (1 / alpha_t.sqrt()) * (cur_y - (eps_factor * eps_theta))
        sigma_t = beta_t.sqrt()
        sample = mean + sigma_t * z * 0.00001
        return sample

    def p_sample_loop(self, social_flat, hist_flat, shape):
        self.model.eval()
        prediction_total = torch.empty(0, device=self.device)
        for _ in range(20):
            cur_y = torch.randn(shape, device=self.device)
            for i in reversed(range(self.n_steps)):
                cur_y = self.p_sample(social_flat, hist_flat, cur_y, i)
            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
        return prediction_total

    def p_sample_loop_mean(self, social_flat, hist_flat, loc):
        prediction_total = torch.empty(0, device=self.device)
        cur_y = loc
        for i in reversed(range(NUM_Tau)):
            cur_y = self.p_sample(social_flat, hist_flat, cur_y, i)
        prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
        return prediction_total

    def p_sample_loop_accelerate(self, social_ctx, hist_ctx, loc):
        """
        social_ctx: [B*N, T_obs, 6]
        hist_ctx  : [B*N, T_obs, 2]
        loc       : [B*N, 20, T_pred, 2]
        """
        cur_y_left = loc[:, :10]
        for i in reversed(range(NUM_Tau)):
            cur_y_left = self.p_sample_accelerate(social_ctx, hist_ctx, cur_y_left, i)

        cur_y_right = loc[:, 10:]
        for i in reversed(range(NUM_Tau)):
            cur_y_right = self.p_sample_accelerate(social_ctx, hist_ctx, cur_y_right, i)

        prediction_total = torch.cat((cur_y_right, cur_y_left), dim=1)
        return prediction_total

    # ---------------------------------------------------------------------
    # proposal-level discriminator helpers
    # ---------------------------------------------------------------------
    def _pack_scene_trajectory(self, obs_traj, obs_traj_rel, pred_future_bn_t_2, seq_start_end):
        """
        obs_traj:      [T_obs, B*N, 2]
        obs_traj_rel:  [T_obs, B*N, 2]
        pred_future:   [B*N, T_pred, 2]
        return:        [B, N, T_all, 4]
        """
        pred_traj = pred_future_bn_t_2.permute(1, 0, 2)
        pred_traj_rel = abs_to_relative(pred_traj)

        traj = torch.cat([obs_traj, pred_traj], dim=0)
        traj_rel = torch.cat([obs_traj_rel, pred_traj_rel], dim=0)
        traj_concat = torch.cat([traj, traj_rel], dim=-1)

        traj_4d = convert_to_4d(traj_concat, seq_start_end).permute(0, 2, 1, 3)
        return traj_4d

    def _score_real_future(self, obs_traj, obs_traj_rel, pred_traj_gt, seq_start_end):
        real_future_bn_t_2 = pred_traj_gt.permute(1, 0, 2).contiguous()
        real_4d = self._pack_scene_trajectory(obs_traj, obs_traj_rel, real_future_bn_t_2, seq_start_end)
        scores_real = self.discriminator(real_4d).view(-1)
        return scores_real

    def _score_fake_proposals(self, obs_traj, obs_traj_rel, generated_y, seq_start_end, detach_future=False):
        """
        generated_y: [B*N, K, T_pred, 2]
        return: [B, K]
        """
        B = seq_start_end.size(0)
        K = generated_y.size(1)
        score_list = []

        for k in range(K):
            fake_future_bn_t_2 = generated_y[:, k]
            if detach_future:
                fake_future_bn_t_2 = fake_future_bn_t_2.detach()
            fake_4d = self._pack_scene_trajectory(obs_traj, obs_traj_rel, fake_future_bn_t_2, seq_start_end)
            score_k = self.discriminator(fake_4d).view(B)
            score_list.append(score_k)

        return torch.stack(score_list, dim=1)

    def _proposal_losses(self, generated_y, fut_traj, variance_estimation, batch_size, num_agents):
        """
        generated_y: [B*N, K, T_pred, 2]
        fut_traj:    [B*N, T_pred, 2]
        variance_estimation: [B*N] or [B*N,1]

        Returns:
            dist_scene_prop: [B, K]
            unc_scene_prop:  [B, K]
        """
        error_l2 = (generated_y - fut_traj.unsqueeze(1)).norm(p=2, dim=-1)               # [B*N, K, T]
        dist_per_agent_prop = (error_l2 * self.temporal_reweight).mean(dim=-1)           # [B*N, K]
        mean_err_per_agent_prop = error_l2.mean(dim=-1)                                  # [B*N, K]

        var = variance_estimation.reshape(-1, 1)
        unc_per_agent_prop = torch.exp(-var) * mean_err_per_agent_prop + var             # [B*N, K]

        dist_scene_prop = self._reshape_agent_proposal(dist_per_agent_prop, batch_size, num_agents).mean(dim=1)
        unc_scene_prop = self._reshape_agent_proposal(unc_per_agent_prop, batch_size, num_agents).mean(dim=1)
        return dist_scene_prop, unc_scene_prop

    # ---------------------------------------------------------------------
    # training / evaluation
    # ---------------------------------------------------------------------
    def fit(self):
        for epoch in range(0, self.cfg.num_epochs):
            loss_total, loss_distance, loss_uncertainty, d_loss = self._train_single_epoch(epoch)
            print_log(
                '[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}\tDiscriminator Loss: {:.6f}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    epoch,
                    loss_total,
                    loss_distance,
                    loss_uncertainty,
                    d_loss,
                ),
                self.log,
            )

            if (epoch + 1) % self.cfg.test_interval == 0:
                performance, samples = self._test_single_epoch()
                for time_i in range(4):
                    print_log(
                        '--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(
                            time_i + 1,
                            performance['ADE'][time_i] / samples,
                            time_i + 1,
                            performance['FDE'][time_i] / samples,
                        ),
                        self.log,
                    )
                cp_path = self.cfg.model_path % (epoch + 1)
                model_cp = {
                    'model_initializer_dict': self.model_initializer.state_dict(),
                    'generator_dict': self.model.state_dict(),
                    'discriminator_dict': self.discriminator.state_dict(),
                }
                torch.save(model_cp, cp_path)

            self.scheduler_model.step()
            self.scheduler_G.step()
            self.scheduler_D.step()

    def _train_single_epoch(self, epoch):
        self.model.train()
        self.discriminator.train()
        self.model_initializer.train()

        loss_total, loss_dt, loss_dc, d_loss, count = 0, 0, 0, 0, 0

        for data in self.train_loader:
            batch_size, num_agents, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
            _, _, hist_flat, social_flat = self.data_preprocess_denoiser(data)
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end = self.data_preprocess_1(data)

            # -------------------------------------------------------------
            # 1) generator forward: initializer + accelerated denoiser
            # -------------------------------------------------------------
            sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
            sample_prediction = (
                torch.exp(variance_estimation / 2)[..., None, None]
                * sample_prediction
                / sample_prediction.std(dim=1).mean(dim=(1, 2)).clamp_min(1e-6)[:, None, None, None]
            )
            loc = sample_prediction + mean_estimation[:, None]
            generated_y = self.p_sample_loop_accelerate(social_flat, hist_flat, loc)  # [B*N, K, T, 2]

            # -------------------------------------------------------------
            # 2) proposal scores from D -> proposal weights (stop-grad)
            # -------------------------------------------------------------
            with torch.no_grad():
                proposal_scores = self._score_fake_proposals(
                    obs_traj=obs_traj,
                    obs_traj_rel=obs_traj_rel,
                    generated_y=generated_y,
                    seq_start_end=seq_start_end,
                    detach_future=False,
                )  # [B, K]
                proposal_weights = self._scores_to_weights(proposal_scores)  # [B, K]

            # -------------------------------------------------------------
            # 3) proposal-weighted generator objective
            # -------------------------------------------------------------
            dist_scene_prop, unc_scene_prop = self._proposal_losses(
                generated_y=generated_y,
                fut_traj=fut_traj,
                variance_estimation=variance_estimation,
                batch_size=batch_size,
                num_agents=num_agents,
            )

            loss_dist = (proposal_weights * dist_scene_prop).sum(dim=1).mean()
            loss_uncertainty = (proposal_weights * unc_scene_prop).sum(dim=1).mean()
            gen_loss = self.lambda_dist * loss_dist + self.lambda_uncertainty * loss_uncertainty

            self.opt.zero_grad()
            self.optimizer_G.zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.optimizer_G.step()

            # -------------------------------------------------------------
            # 4) discriminator update with all fake proposals
            # -------------------------------------------------------------
            d_running_loss = torch.zeros(1, device=self.device)
            for _ in range(self.num_discriminator_steps):
                scores_real = self._score_real_future(
                    obs_traj=obs_traj,
                    obs_traj_rel=obs_traj_rel,
                    pred_traj_gt=pred_traj_gt,
                    seq_start_end=seq_start_end,
                )  # [B]

                scores_fake = self._score_fake_proposals(
                    obs_traj=obs_traj,
                    obs_traj_rel=obs_traj_rel,
                    generated_y=generated_y.detach(),
                    seq_start_end=seq_start_end,
                    detach_future=True,
                )  # [B, K]

                data_loss = self._gan_d_loss(
                    scores_real=scores_real,
                    scores_fake=scores_fake.reshape(-1),
                )
                d_running_loss = d_running_loss + data_loss

                self.optimizer_D.zero_grad()
                data_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.optimizer_D.step()

            loss_total += gen_loss.item()
            loss_dt += (self.lambda_dist * loss_dist).item()
            loss_dc += (self.lambda_uncertainty * loss_uncertainty).item()
            d_loss += d_running_loss.item() / max(self.num_discriminator_steps, 1)
            count += 1
            if getattr(self.cfg, 'debug', False) and count == 2:
                break

        return loss_total / count, loss_dt / count, loss_dc / count, d_loss / count

    def _test_single_epoch(self):
        performance = {'FDE': [0, 0, 0, 0], 'ADE': [0, 0, 0, 0]}
        samples = 0

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        with torch.no_grad():
            for data in self.test_loader:
                batch_size, num_agents, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
                _, _, hist_flat, social_flat = self.data_preprocess_denoiser(data)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
                sample_prediction = (
                    torch.exp(variance_estimation / 2)[..., None, None]
                    * sample_prediction
                    / sample_prediction.std(dim=1).mean(dim=(1, 2)).clamp_min(1e-6)[:, None, None, None]
                )
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(social_flat, hist_flat, loc)

                fut_traj_rep = fut_traj.unsqueeze(1).repeat(1, pred_traj.size(1), 1, 1)
                distances = torch.norm(fut_traj_rep - pred_traj, dim=-1) * self.traj_scale
                for time_i in range(1, 5):
                    ade = (distances[:, :, :5 * time_i]).mean(dim=-1).min(dim=-1)[0].sum()
                    fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
                    performance['ADE'][time_i - 1] += ade.item()
                    performance['FDE'][time_i - 1] += fde.item()
                samples += distances.shape[0]
        return performance, samples

    # ---------------------------------------------------------------------
    # optional utilities
    # ---------------------------------------------------------------------
    def save_data(self):
        model_path = './results/checkpoints/d_gan.p'
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        if 'model_initializer_dict' in model_dict:
            self.model_initializer.load_state_dict(model_dict['model_initializer_dict'])
        if 'generator_dict' in model_dict:
            self.model.load_state_dict(model_dict['generator_dict'], strict=False)

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        root_path = './visualization/data/'

        with torch.no_grad():
            for data in self.test_loader:
                _, _, traj_mask, past_traj, _ = self.data_preprocess(data)
                _, _, hist_flat, social_flat = self.data_preprocess_denoiser(data)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
                torch.save(sample_prediction, root_path + 'p_var1.pt')
                torch.save(mean_estimation, root_path + 'p_mean1.pt')
                torch.save(variance_estimation, root_path + 'p_sigma1.pt')

                sample_prediction = (
                    torch.exp(variance_estimation / 2)[..., None, None]
                    * sample_prediction
                    / sample_prediction.std(dim=1).mean(dim=(1, 2)).clamp_min(1e-6)[:, None, None, None]
                )
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(social_flat, hist_flat, loc)
                pred_mean = self.p_sample_loop_mean(social_flat, hist_flat, mean_estimation)

                torch.save(data['pre_motion_3D'], root_path + 'past1.pt')
                torch.save(data['fut_motion_3D'], root_path + 'future1.pt')
                torch.save(pred_traj, root_path + 'prediction1.pt')
                torch.save(pred_mean, root_path + 'p_mean_denoise1.pt')
                raise ValueError

    def test_single_model(self):
        model_path = './results/checkpoints/d_gan.p'
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        if 'model_initializer_dict' in model_dict:
            self.model_initializer.load_state_dict(model_dict['model_initializer_dict'])
        if 'generator_dict' in model_dict:
            self.model.load_state_dict(model_dict['generator_dict'], strict=False)
        if 'discriminator_dict' in model_dict:
            self.discriminator.load_state_dict(model_dict['discriminator_dict'])

        performance = {'FDE': [0, 0, 0, 0], 'ADE': [0, 0, 0, 0]}
        samples = 0
        print_log(model_path, log=self.log)

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        with torch.no_grad():
            for data in self.test_loader:
                batch_size, num_agents, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
                _, _, hist_flat, social_flat = self.data_preprocess_denoiser(data)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
                sample_prediction = (
                    torch.exp(variance_estimation / 2)[..., None, None]
                    * sample_prediction
                    / sample_prediction.std(dim=1).mean(dim=(1, 2)).clamp_min(1e-6)[:, None, None, None]
                )
                loc = sample_prediction + mean_estimation[:, None]
                pred_traj = self.p_sample_loop_accelerate(social_flat, hist_flat, loc)

                fut_traj_rep = fut_traj.unsqueeze(1).repeat(1, pred_traj.size(1), 1, 1)
                distances = torch.norm(fut_traj_rep - pred_traj, dim=-1) * self.traj_scale
                for time_i in range(1, 5):
                    ade = (distances[:, :, :5 * time_i]).mean(dim=-1).min(dim=-1)[0].sum()
                    fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
                    performance['ADE'][time_i - 1] += ade.item()
                    performance['FDE'][time_i - 1] += fde.item()
                samples += distances.shape[0]

        for time_i in range(4):
            line = '--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(
                time_i + 1,
                performance['ADE'][time_i] / samples,
                time_i + 1,
                performance['FDE'][time_i] / samples,
            )
            print(line)
            print_log(line, self.log)
