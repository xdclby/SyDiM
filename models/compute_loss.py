import torch
import random


def bce_loss(input, target):
	neg_abs = -input.abs()
	loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
	return loss.mean()

def gan_d_loss(scores_real, scores_fake):
	"""
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
	y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
	y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
	loss_real = bce_loss(scores_real, y_real)
	loss_fake = bce_loss(scores_fake, y_fake)
	return loss_real + loss_fake