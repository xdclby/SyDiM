import argparse
# from trainer import train as sydim
from trainer import train_modified as sydim

def parse_config():
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", default=True)
	parser.add_argument("--learning_rate", type=int, default=0.002)
	parser.add_argument("--max_epochs", type=int, default=128)

	parser.add_argument('--cfg', default='sydimnba')
	parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use.')
	parser.add_argument('--train', type=int, default=0, help='Whether train or evaluate.')
	
	parser.add_argument("--info", type=str, default='', help='Name of the experiment. ' 'It will be used in file creation.')

	#dicriminator
	parser.add_argument('--d_type', default='local', type=str)
	parser.add_argument('--encoder_h_dim_d', default=64, type=int)
	parser.add_argument('--d_learning_rate', default=5e-4, type=float)
	parser.add_argument('--d_steps', default=2, type=int)
	parser.add_argument('--clipping_threshold_d', default=0, type=float)


	return parser.parse_args()


def main(config):
	t = sydim.Trainer(config)
	# if config.train==1:
	t.fit()
	# else:
	# t.save_data()
	# t.test_single_model()


if __name__ == "__main__":
	config = parse_config()
	main(config)
