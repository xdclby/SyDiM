# #
# train_batch_size             : 5
# test_batch_size              : 100
import os
import time
import torch
import random
import numpy as np
import torch.nn as nn

from tqdm import tqdm

from utils.config import Config
from utils.utils import print_log, abs_to_relative, convert_to_4d


from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate


from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
from models.dicriminator import TrajectoryDiscriminator
from models.discriminator1 import TrajectoryDiscriminator1
from models.compute_loss import gan_d_loss

import pdb
NUM_Tau = 15  # 设置扩散模型的时间步长

class Trainer:
	def __init__(self, config):
		# 初始化 Trainer 类
		if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)  # 设置 GPU 设备
		self.device = torch.device('cuda') if config.cuda else torch.device('cpu')  #选择计算设备
		self.cfg = Config(config.cfg, config.info)  #加载配置文件

		# ------------------------- prepare train/test data loader -------------------------
		# 准备训练和测试数据加载器
		train_dset = NBADataset(
			obs_len=self.cfg.past_frames,
			pred_len=self.cfg.future_frames,
			training=True)

		self.train_loader = DataLoader(
			train_dset,
			batch_size=self.cfg.train_batch_size,
			shuffle=True,
			num_workers=4,
			collate_fn=seq_collate,
			pin_memory=True)

		test_dset = NBADataset(
			obs_len=self.cfg.past_frames,
			pred_len=self.cfg.future_frames,
			training=False)

		self.test_loader = DataLoader(
			test_dset,
			batch_size=self.cfg.test_batch_size,
			shuffle=False,
			num_workers=4,
			collate_fn=seq_collate,
			pin_memory=True)


		# data normalization parameters
		# 数据归一化参数
		self.traj_mean = torch.FloatTensor(self.cfg.traj_mean).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 轨迹均值
		self.traj_scale = self.cfg.traj_scale  #轨迹的缩放因子

		# ------------------------- define diffusion parameters -------------------------
		# 定义扩散模型参数
		self.n_steps = self.cfg.diffusion.steps # define total diffusion steps 总扩散步数

		# make beta schedule and calculate the parameters used in denoising process.
		# 生成 beta 调度，并计算降噪过程中的相关参数
		self.betas = self.make_beta_schedule(
			schedule=self.cfg.diffusion.beta_schedule, n_timesteps=self.n_steps,
			start=self.cfg.diffusion.beta_start, end=self.cfg.diffusion.beta_end).cuda()

		self.alphas = 1 - self.betas  # 计算 alpha 值（数据保留比例）
		self.alphas_prod = torch.cumprod(self.alphas, 0)  # 计算每一步的 alpha 连乘积
		self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)  # 计算累计 alpha 的平方根
		self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)  # 计算 (1 - alpha) 的平方根


		# ------------------------- define models -------------------------
		# 定义模型
		self.model = CoreDenoisingModel().cuda()  # 核心去噪模型
		# self.discriminator = TrajectoryDiscriminator(obs_len=self.cfg.past_frames, pred_len=self.cfg.future_frames).cuda()
		# self.discriminator = TrajectoryDiscriminator(seq_len=self.cfg.past_frames+self.cfg.future_frames).cuda()
		self.discriminator = TrajectoryDiscriminator1(input_dim=4, model_dim=128, num_heads=8, num_layers=4, dim_feedforward=256, dropout=0.1, num_entities=11, use_entity_embedding=True, output_dim=1).cuda()
		# load pretrained models
		self.model_initializer = InitializationModel(t_h=10, d_h=6, t_f=20, d_f=2, k_pred=20).cuda()
		# 加载预训练模型
		# if hasattr(self.cfg, 'pretrained_core_denoising_model') and self.cfg.pretrained_core_denoising_model:
		# 	model_cp1 = torch.load(self.cfg.pretrained_core_denoising_model, map_location='cpu')
		# 	model_cp2 = torch.load(self.cfg.model_initializer, map_location='cpu')
		# 	self.model_initializer.load_state_dict(model_cp2['model_initializer_dict'])
		# 	self.model.load_state_dict(model_cp1['model_dict'])
		# else:
		# 	self.initializer_weights()
		if hasattr(self.cfg, 'pretrained_core_denoising_model') and self.cfg.pretrained_core_denoising_model:
			model_cp = torch.load(self.cfg.pretrained_core_denoising_model, map_location='cpu')
			self.model.load_state_dict(model_cp['generator_dict'])
			self.model_initializer.load_state_dict(model_cp['model_initializer_dict'])
			self.discriminator.load_state_dict(model_cp['discriminator_dict'])
		else:
			self.initialize_weights()

		# 初始化模型  **************************************************************************


		# 优化器和学习率调度器
		"""
			这行代码使用 AdamW 优化器对 model_initializer 的参数进行优化。
			self.model_initializer.parameters() 获取模型的可训练参数。
			lr=config.learning_rate 设置优化器的学习率。
			AdamW 是 Adam 优化器的改进版，它使用权重衰减来抑制过拟合，比标准的 Adam 更适合用于训练深层神经网络。
		"""
		self.opt = torch.optim.AdamW(self.model_initializer.parameters(), lr=config.learning_rate)
		self.optimizer_G = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate*0.5)
		self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters(), lr=config.learning_rate)
		"""
			StepLR 是一个学习率调度器，用于在训练过程中调整学习率。
			step_size=self.cfg.decay_step 表示每隔 decay_step 个 epoch，学习率就会发生变化。
			gamma=self.cfg.decay_gamma 是学习率更新的乘数因子，更新时新学习率会乘以这个因子，使得学习率逐渐降低。
		"""
		self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)
		self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)
		self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)

		# ------------------------- prepare logs -------------------------
		# 准备日志
		self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
		self.print_model_param(self.discriminator, name='discriminator')
		self.print_model_param(self.model, name='Core Denoising Model')  # 打印模型参数信息
		self.print_model_param(self.model_initializer, name='Initialization Model')

		# temporal reweight in the loss, it is not necessary.
		# 在损失函数中使用的时间权重参数
		self.temporal_reweight = torch.FloatTensor([21 - i for i in range(1, 21)]).cuda().unsqueeze(0).unsqueeze(0) / 10

	def initialize_weights(self):
		# 使用Xavier初始化模型权重
		for m in self.model.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
		for m in self.discriminator.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def print_model_param(self, model: nn.Module, name: str = 'Model') -> None:
		'''
		Count the trainable/total parameters in `model`.
		'''
		total_num = sum(p.numel() for p in model.parameters())
		trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print_log("[{}] Trainable/Total: {}/{}".format(name, trainable_num, total_num), self.log)
		return None


	def make_beta_schedule(self, schedule: str = 'linear',
			n_timesteps: int = 1000,
			start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
		'''
		Make beta schedule.

		Parameters
		----
		schedule: str, in ['linear', 'quad', 'sigmoid'],
		n_timesteps: int, diffusion steps,
		start: float, beta start, `start<end`,
		end: float, beta end,

		Returns
		----
		betas: Tensor with the shape of (n_timesteps)

		'''
		if schedule == 'linear':
			betas = torch.linspace(start, end, n_timesteps)
		elif schedule == "quad":
			betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
		elif schedule == "sigmoid":
			betas = torch.linspace(-6, 6, n_timesteps)
			betas = torch.sigmoid(betas) * (end - start) + start
		return betas


	def extract(self, input, t, x):
		shape = x.shape
		out = torch.gather(input, 0, t.to(input.device))
		reshape = [t.shape[0]] + [1] * (len(shape) - 1)
		return out.reshape(*reshape)

	def noise_estimation_loss(self, x, y_0, mask):
		batch_size = x.shape[0]
		# Select a random step for each example
		t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,)).to(x.device)
		t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]
		# x0 multiplier
		a = self.extract(self.alphas_bar_sqrt, t, y_0)
		beta = self.extract(self.betas, t, y_0)
		# eps multiplier
		am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
		e = torch.randn_like(y_0)
		# model input
		y = y_0 * a + e * am1
		output = self.model(y, beta, x, mask)
		# batch_size, 20, 2
		return (e - output).square().mean()



	def p_sample(self, x, mask, cur_y, t):
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		# Factor to the model output
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		# Model output
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		eps_theta = self.model(cur_y, beta, x, mask)
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		# Generate z
		z = torch.randn_like(cur_y).to(x.device)
		# Fixed sigma
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z
		return (sample)

	def p_sample_accelerate(self, x, mask, cur_y, t):
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		# Factor to the model output
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		# Model output
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask)
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		# Generate z
		z = torch.randn_like(cur_y).to(x.device)
		# Fixed sigma
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z * 0.00001
		return (sample)



	def p_sample_loop(self, x, mask, shape):
		self.model.eval()
		prediction_total = torch.Tensor().cuda()
		for _ in range(20):
			cur_y = torch.randn(shape).to(x.device)
			for i in reversed(range(self.n_steps)):
				cur_y = self.p_sample(x, mask, cur_y, i)
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total

	def p_sample_loop_mean(self, x, mask, loc):
		prediction_total = torch.Tensor().cuda()
		for loc_i in range(1):
			cur_y = loc
			for i in reversed(range(NUM_Tau)):
				cur_y = self.p_sample(x, mask, cur_y, i)
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total

	def p_sample_loop_accelerate(self, x, mask, loc):
		'''
		Batch operation to accelerate the denoising process.

		x: [11, 10, 6]
		mask: [11, 11]
		cur_y: [11, 10, 20, 2]
		'''
		prediction_total = torch.Tensor().cuda()
		cur_y = loc[:, :10]
		for i in reversed(range(NUM_Tau)):
			cur_y = self.p_sample_accelerate(x, mask, cur_y, i)
		cur_y_ = loc[:, 10:]
		for i in reversed(range(NUM_Tau)):
			cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)
		# shape: B=b*n, K=10, T, 2
		prediction_total = torch.cat((cur_y_, cur_y), dim=1)
		return prediction_total



	"""
		实现了一个包含训练、评估、保存模型的完整训练过程。具体地，它执行以下步骤：
		在每个 epoch 中训练模型，并计算损失。
		每隔一定的 epoch 对模型进行评估，打印评估指标。
		保存模型检查点，以便后续加载。
		调整学习率，以提高模型的训练效果。
	"""
	def fit(self):
		# Training loop
		for epoch in range(0, self.cfg.num_epochs):
			loss_total, loss_distance, loss_uncertainty, d_loss = self._train_single_epoch(epoch)
			print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}\tDiscriminator Loss: {:.6f}'.format(
				time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
				epoch, loss_total, loss_distance, loss_uncertainty, d_loss), self.log)

			if (epoch + 1) % self.cfg.test_interval == 0:
				performance, samples = self._test_single_epoch()
				for time_i in range(4):
					print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(
						time_i+1, performance['ADE'][time_i]/samples,
						time_i+1, performance['FDE'][time_i]/samples), self.log)
				cp_path = self.cfg.model_path % (epoch + 1)
				# model_cp = {'model_initializer_dict': self.model_initializer.state_dict()}
				model_cp = {
					'model_initializer_dict': self.model_initializer.state_dict(),
					'generator_dict': self.model.state_dict(),
					'discriminator_dict': self.discriminator.state_dict()
				}
				torch.save(model_cp, cp_path)

				torch.save(model_cp, cp_path)
			self.scheduler_model.step()


	def data_preprocess(self, data):
		"""
			pre_motion_3D: torch.Size([32, 11, 10, 2]), [batch_size, num_agent, past_frame, dimension]
			fut_motion_3D: torch.Size([32, 11, 20, 2])
			fut_motion_mask: torch.Size([32, 11, 20])
			pre_motion_mask: torch.Size([32, 11, 10])
			traj_scale: 1
			pred_mask: None
			seq: nba
		"""
		batch_size = data['pre_motion_3D'].shape[0]

		traj_mask = torch.zeros(batch_size*11, batch_size*11).cuda()
		for i in range(batch_size):
			traj_mask[i*11:(i+1)*11, i*11:(i+1)*11] = 1.

		initial_pos = data['pre_motion_3D'].cuda()[:, :, -1:]
		# augment input: absolute position, relative position, velocity
		past_traj_abs = ((data['pre_motion_3D'].cuda() - self.traj_mean)/self.traj_scale).contiguous().view(-1, 10, 2)
		past_traj_rel = ((data['pre_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, 10, 2)
		past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1)
		past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

		fut_traj = ((data['fut_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, 20, 2)
		return batch_size, traj_mask, past_traj, fut_traj

	def data_preprocess_1(self, data):
		batch_size = data['pre_motion_3D'].shape[0]
		num_agents = data['pre_motion_3D'].shape[1]

		# 生成 seq_start_end（与 seq_collate 格式对齐）
		seq_start_end = torch.LongTensor([
			[i * num_agents, (i + 1) * num_agents]
			for i in range(batch_size)
		]).cuda()

		# 计算标准化绝对坐标（对齐 seq_collate 的 obs_traj/pred_traj）
		past_traj_abs = (data['pre_motion_3D'].cuda() - self.traj_mean) / self.traj_scale
		fut_traj_abs = (data['fut_motion_3D'].cuda() - self.traj_mean) / self.traj_scale

		# 计算相对坐标（对齐 seq_collate 的 obs_traj_rel/pred_traj_rel）
		def get_relative(traj):
			""" 计算相邻时间步的相对位移 Δx = x_t - x_{t-1}，首帧补零 """
			rel = traj[:, :, 1:] - traj[:, :, :-1]
			rel = torch.cat([torch.zeros_like(traj[:, :, :1]), rel], dim=2)
			return rel / self.traj_scale

		past_traj_rel = get_relative(data['pre_motion_3D'].cuda())
		fut_traj_rel = get_relative(data['fut_motion_3D'].cuda())

		# 维度调整（对齐 seq_collate 的格式）
		def reshape_permute(tensor, seq_len):
			return tensor.view(-1, seq_len, 2).permute(1, 0, 2)  # [seq_len, batch*agent, 2]

		obs_traj = reshape_permute(past_traj_abs, 10)  # [10, batch*agent, 2]
		pred_traj = reshape_permute(fut_traj_abs, 20)  # [20, batch*agent, 2]
		obs_traj_rel = reshape_permute(past_traj_rel, 10)  # [10, batch*agent, 2]
		pred_traj_rel = reshape_permute(fut_traj_rel, 20)  # [20, batch*agent, 2]

		# 处理其他参数（假设 data 中存在对应字段）
		# non_linear_ped = data['non_linear_ped'].cuda().view(-1)  # [batch*agent]
		# loss_mask = data['fut_motion_mask'].cuda().view(-1, 20)  # [batch*agent, 20]

		# 返回与 seq_collate 完全一致的格式
		return (
			obs_traj,  # [10, batch*agent, 2]
			pred_traj,  # [20, batch*agent, 2]
			obs_traj_rel,  # [10, batch*agent, 2]
			pred_traj_rel,  # [20, batch*agent, 2]
			# non_linear_ped,  # [batch*agent]
			# loss_mask,  # [batch*agent, 20]
			seq_start_end  # [batch_size, 2]
		)


	def _train_single_epoch(self, epoch):

		self.model.train()
		self.discriminator.train()
		self.model_initializer.train()
		loss_total, loss_dt, loss_dc, d_loss, count = 0, 0, 0, 0, 0

		for data in self.train_loader:
			batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
			obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end = self.data_preprocess_1(data)
			# print("obs_traj", obs_traj.size())

			losses = {}
			loss = torch.zeros(1).to(pred_traj_gt)

			sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
			sample_prediction = torch.exp(variance_estimation / 2)[
									..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
				dim=(1, 2))[:, None, None, None]
			loc = sample_prediction + mean_estimation[:, None]

			# print("生成轨迹之前的时间：", time.ctime(time.time()))
			generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
			# print("生成轨迹之后的时间：", time.ctime(time.time()))


			# 计算生成器损失
			loss_dist = ((generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1)
						 *
						 self.temporal_reweight
						 ).mean(dim=-1).min(dim=1)[0].mean()
			loss_uncertainty = (torch.exp(-variance_estimation)
								*
								(generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2))
								+
								variance_estimation
								).mean()

			gen_loss = loss_dist * 50 + loss_uncertainty

			#生成器优化
			self.opt.zero_grad()
			gen_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
			self.opt.step()

			#更新判别器
			pred_traj_fake = generated_y.detach()
			pred_traj_fake = pred_traj_fake.mean(dim=2).permute(1, 0, 2)
			# print(pred_traj_fake.shape)
			pred_traj_fake_rel = abs_to_relative(pred_traj_fake)
			# pred_traj_fake = pred_traj_fake.permute(1, 0, 2, 3)
			# seq_len1, batch1, num_agents1, _ = pred_traj_fake.shape
			# pred_traj_fake = pred_traj_fake.reshape(seq_len1, batch1 * num_agents1, 2)
			# print(pred_traj_fake_rel.shape)

			traj_real = torch.cat([obs_traj, pred_traj_gt],dim=0)
			# traj_real_last_position = traj_real[-1]
			traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel],dim=0)
			traj_fake = torch.cat([obs_traj, pred_traj_fake],dim=0)
			# traj_fake_last_position = traj_fake[-1]
			traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel],dim=0)

			# scores_fake = self.discriminator(traj_fake, traj_fake_rel, seq_start_end)
			# scores_real = self.discriminator(traj_real, traj_real_rel, seq_start_end)
			# scores_fake = self.discriminator(traj_fake, traj_fake_last_position)
			# scores_real = self.discriminator(traj_real, traj_real_last_position)

			#******************************************************************************************
			traj_concat_real = torch.cat([traj_real, traj_real_rel], dim=-1)  # (T, batch, 4)
			traj_concat_fake = torch.cat([traj_fake, traj_fake_rel], dim=-1)  # (T, batch, 4)
			# print(traj_concat_real.shape, traj_concat_fake.shape)

			traj_real_4d = convert_to_4d(traj_concat_real, seq_start_end).permute(0, 2, 1, 3)
			traj_fake_4d = convert_to_4d(traj_concat_fake, seq_start_end).permute(0, 2, 1, 3)
			# print(traj_real_4d.shape, traj_fake_4d.shape)

			scores_fake = self.discriminator(traj_fake_4d)
			scores_real = self.discriminator(traj_real_4d)
			# print(scores_fake, scores_real)

			# 用梯度惩罚计算损失
			data_loss = gan_d_loss(scores_real, scores_fake)
			losses['D_data_loss'] = data_loss.item()
			loss += data_loss
			losses['D_total_loss'] = loss.item()

			self.optimizer_D.zero_grad()
			loss.backward()
			#进行梯度裁剪与优化
			torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
			self.optimizer_D.step()


			loss_total += gen_loss.item()
			loss_dt += loss_dist.item() * 50
			loss_dc += loss_uncertainty.item()
			d_loss += loss.item()
			count += 1
			if self.cfg.debug and count == 2:
				break

		return loss_total / count, loss_dt / count, loss_dc / count, d_loss / count


	def _test_single_epoch(self):
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0]}
		samples = 0
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		count = 0
		with torch.no_grad():
			for data in self.test_loader:
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]

				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

				fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
				for time_i in range(1, 5):
					ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
				samples += distances.shape[0]
				count += 1
				# if count==100:
				# 	break
		return performance, samples


	def save_data(self):
		'''
		Save the visualization data.
		'''
		model_path = './results/checkpoints/d_gan.p'
		model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		self.model_initializer.load_state_dict(model_dict)
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

				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				torch.save(sample_prediction, root_path+'p_var1.pt')
				torch.save(mean_estimation, root_path+'p_mean1.pt')
				torch.save(variance_estimation, root_path+'p_sigma1.pt')

				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]

				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
				pred_mean = self.p_sample_loop_mean(past_traj, traj_mask, mean_estimation)

				torch.save(data['pre_motion_3D'], root_path+'past1.pt')
				torch.save(data['fut_motion_3D'], root_path+'future1.pt')
				torch.save(pred_traj, root_path+'prediction1.pt')
				torch.save(pred_mean, root_path+'p_mean_denoise1.pt')

				raise ValueError



	def test_single_model(self):
		model_path = './results/checkpoints/d_gan.p'
		model_dict1 = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		# model_dict2 = torch.load(model_path, map_location=torch.device('cpu'))['generator_dict']
		# model_dict3 = torch.load(model_path, map_location=torch.device('cpu'))['discriminator_dict']
		self.model_initializer.load_state_dict(model_dict1)
		# self.model.load_state_dict(model_dict2)
		# self.discriminator.load_state_dict(model_dict3)
		print("!!!")
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0]}
		samples = 0
		print_log(model_path, log=self.log)
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		count = 0
		with torch.no_grad():
			for data in self.test_loader:
				batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]

				pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

				fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
				# b*n, K, T, 2
				distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
				for time_i in range(1, 5):
					ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
				samples += distances.shape[0]
				count += 1
					# if count==2:
					# 	break
		for time_i in range(4):
			print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(time_i+1, performance['ADE'][time_i]/samples, \
				time_i+1, performance['FDE'][time_i]/samples), log=self.log)
