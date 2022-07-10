import argparse


def get_args():
	parser = argparse.ArgumentParser('GF')

	parser.add_argument('--model_name', type=str, default='c3d',
						help='The action recognition')
	parser.add_argument('--dataset_name', type=str, default='hmdb51',
						help='The dataset: hmdb51/ucf101')
	parser.add_argument('--gpus', nargs='+', type=int, required=False, default=[0],
						help='The gpus to use')
	parser.add_argument('--train_num', type=int, default=20,
						help='The number of testing')
	parser.add_argument('--max_iter', type=int, default=15000,
						help='The max number of iter')

	parser.add_argument('--test_num', type=int, default=40,
						help='The number of testing')
	# cnn model
	parser.add_argument('--policy_rl', type=float, default=0.003)
	parser.add_argument('--num_segments', type=int, default=16)
	parser.add_argument('--k', type=int, default=3)
	parser.add_argument('--dropout', type=float, default=0.5)
	parser.add_argument('--num_classes', type=int, default=51)    # 120
	parser.add_argument('--backbone_lr', type=float, default=0.01)
	parser.add_argument('--fc_lr', type=float, default=0.005)

	# dataset
	parser.add_argument('--weight_decay', type=float, default='0.0001')
	parser.add_argument('--patch_size', type=int, default=66)  # 15以下 容易fail 和炸

# ------------------------SUPER TEST---------------------------

	parser.add_argument('--train_stage', type=int, default=2)
	parser.add_argument('--cuda', type=bool, default=True)
	parser.add_argument('--random_patch', type=bool, default=False)
	parser.add_argument('--policy_conv', type=bool, default=True)
	parser.add_argument('--seed', type=int, default=1007)
	parser.add_argument('--glance_size', type=int, default=112)
	parser.add_argument('--policy_lr ', type=float, default='0.0003')
	parser.add_argument('--feature_map_channels', type=int, default=1280)  # 1500
	parser.add_argument('--action_dim', type=int, default=49)
	parser.add_argument('--hidden_state_dim', type=int, default=512)
	parser.add_argument('--penalty ', type=float, default='0.5')
	parser.add_argument('--gamma', type=float, default='0.7')
	parser.add_argument('--gpu', type=int, default=0)

	args = parser.parse_args()
	return args
	
















