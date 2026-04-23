import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from utils.config import Config
from utils.utils import print_log, abs_to_relative, convert_to_4d
from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate

from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion1 import TransformerDenoisingModel as CoreDenoisingModel
from models.model_diffusion1 import build_all_social_features
from models.discriminator1 import TrajectoryDiscriminator1
from models.compute_loss import gan_d_loss

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

        self.temporal_reweight = torch.FloatTensor([21 - i for i in range(1, 21)]).to(self.device).unsqueeze(0).unsqueeze(0) / 10

    def initialize_weights(self):
        for m in self.discriminator.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def print_model_param(self, model: nn.Module, name: str = 'Model') -> None:
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_log(f"[{name}] Trainable/Total: {trainable_num}/{total_num}", self.log)

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

    def data_preprocess(self, data):
        batch_size = data['pre_motion_3D'].shape[0]
        num_agents = data['pre_motion_3D'].shape[1]

        traj_mask = torch.zeros(batch_size * num_agents, batch_size * num_agents, device=self.device)
        for i in range(batch_size):
            traj_mask[i * num_agents:(i + 1) * num_agents, i * num_agents:(i + 1) * num_agents] = 1.0

        initial_pos = data['pre_motion_3D'].to(self.device)[:, :, -1:]
        past_traj_abs = ((data['pre_motion_3D'].to(self.device) - self.traj_mean) / self.traj_scale).contiguous().view(-1, 10, 2)
        past_traj_rel = ((data['pre_motion_3D'].to(self.device) - initial_pos) / self.traj_scale).contiguous().view(-1, 10, 2)
        past_traj_vel = torch.cat(
            (past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1
        )
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

        fut_traj = ((data['fut_motion_3D'].to(self.device) - initial_pos) / self.traj_scale).contiguous().view(-1, 20, 2)
        return batch_size, traj_mask, past_traj, fut_traj

    def data_preprocess_denoiser(self, data):
        """
        New denoiser input:
            hist_local_scene : [B, N, T_obs, 2]  agent-local history for ST encoder
            social_scene     : [B, N, T_obs, 6]  shared-coordinate social context
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

        obs_traj = reshape_permute(past_traj_abs, 10)
        pred_traj = reshape_permute(fut_traj_abs, 20)
        obs_traj_rel = reshape_permute(past_traj_rel, 10)
        pred_traj_rel = reshape_permute(fut_traj_rel, 20)

        return obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, seq_start_end

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
        if t == 0:
            z = torch.zeros_like(cur_y)
        else:
            z = torch.randn_like(cur_y)

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
        if t == 0:
            z = torch.zeros_like(cur_y)
        else:
            z = torch.randn_like(cur_y)

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
            batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
            _, _, hist_flat, social_flat = self.data_preprocess_denoiser(data)
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end = self.data_preprocess_1(data)

            d_running_loss = torch.zeros(1, device=self.device)

            sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
            sample_prediction = (
                torch.exp(variance_estimation / 2)[..., None, None]
                * sample_prediction
                / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
            )
            loc = sample_prediction + mean_estimation[:, None]

            generated_y = self.p_sample_loop_accelerate(social_flat, hist_flat, loc)

            loss_dist = (
                (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1)
                * self.temporal_reweight
            ).mean(dim=-1).min(dim=1)[0].mean()

            loss_uncertainty = (
                torch.exp(-variance_estimation)
                * (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2))
                + variance_estimation
            ).mean()

            gen_loss = loss_dist * 50 + loss_uncertainty

            self.opt.zero_grad()
            self.optimizer_G.zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.optimizer_G.step()

            pred_traj_fake = generated_y.detach().mean(dim=1).permute(1, 0, 2)
            pred_traj_fake_rel = abs_to_relative(pred_traj_fake)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            traj_concat_real = torch.cat([traj_real, traj_real_rel], dim=-1)
            traj_concat_fake = torch.cat([traj_fake, traj_fake_rel], dim=-1)

            traj_real_4d = convert_to_4d(traj_concat_real, seq_start_end).permute(0, 2, 1, 3)
            traj_fake_4d = convert_to_4d(traj_concat_fake, seq_start_end).permute(0, 2, 1, 3)

            scores_fake = self.discriminator(traj_fake_4d)
            scores_real = self.discriminator(traj_real_4d)

            data_loss = gan_d_loss(scores_real, scores_fake)
            d_running_loss = d_running_loss + data_loss

            self.optimizer_D.zero_grad()
            d_running_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.optimizer_D.step()

            loss_total += gen_loss.item()
            loss_dt += loss_dist.item() * 50
            loss_dc += loss_uncertainty.item()
            d_loss += d_running_loss.item()
            count += 1
            if self.cfg.debug and count == 2:
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
                batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
                _, _, hist_flat, social_flat = self.data_preprocess_denoiser(data)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
                sample_prediction = (
                    torch.exp(variance_estimation / 2)[..., None, None]
                    * sample_prediction
                    / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
                )
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(social_flat, hist_flat, loc)

                fut_traj_rep = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
                distances = torch.norm(fut_traj_rep - pred_traj, dim=-1) * self.traj_scale
                for time_i in range(1, 5):
                    ade = (distances[:, :, :5 * time_i]).mean(dim=-1).min(dim=-1)[0].sum()
                    fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
                    performance['ADE'][time_i - 1] += ade.item()
                    performance['FDE'][time_i - 1] += fde.item()
                samples += distances.shape[0]
        return performance, samples

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
                _, traj_mask, past_traj, _ = self.data_preprocess(data)
                _, _, hist_flat, social_flat = self.data_preprocess_denoiser(data)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
                torch.save(sample_prediction, root_path + 'p_var1.pt')
                torch.save(mean_estimation, root_path + 'p_mean1.pt')
                torch.save(variance_estimation, root_path + 'p_sigma1.pt')

                sample_prediction = (
                    torch.exp(variance_estimation / 2)[..., None, None]
                    * sample_prediction
                    / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
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
                batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
                _, _, hist_flat, social_flat = self.data_preprocess_denoiser(data)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
                sample_prediction = (
                    torch.exp(variance_estimation / 2)[..., None, None]
                    * sample_prediction
                    / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
                )
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(social_flat, hist_flat, loc)

                fut_traj_rep = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
                distances = torch.norm(fut_traj_rep - pred_traj, dim=-1) * self.traj_scale
                for time_i in range(1, 5):
                    ade = (distances[:, :, :5 * time_i]).mean(dim=-1).min(dim=-1)[0].sum()
                    fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
                    performance['ADE'][time_i - 1] += ade.item()
                    performance['FDE'][time_i - 1] += fde.item()
                samples += distances.shape[0]

        for time_i in range(4):
            print_log(
                '--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(
                    time_i + 1,
                    performance['ADE'][time_i] / samples,
                    time_i + 1,
                    performance['FDE'][time_i] / samples,
                ),
                log=self.log,
            )
