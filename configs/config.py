import argparse

from mmcv import DictAction


def get_args():
	parser = argparse.ArgumentParser('AST')

	parser.add_argument('--model_name', type=str, default='c3d',
						help='The action recognition')
	parser.add_argument('--dataset_name', type=str, default='hmdb51',
						help='The dataset: hmdb51/ucf101')
	parser.add_argument('--gpus', nargs='+', type=int, required=False, default=[0],
						help='The gpus to use')
	parser.add_argument('--train_num', type=int, default=20,
						help='The number of testing')

	parser.add_argument('--test_num', type=int, default=40,
						help='The number of testing')
	# cnn model
	parser.add_argument('--policy_rl', type=float, default=0.003)
	parser.add_argument('--num_segments', type=int, default=25)
	parser.add_argument('--k', type=int, default=3)
	parser.add_argument('--dropout', type=float, default=0.5)
	parser.add_argument('--num_classes', type=int, default=51)
	parser.add_argument('--backbone_lr', type=float, default=0.01)
	parser.add_argument('--fc_lr', type=float, default=0.005)

	# dataset
	parser.add_argument('--weight_decay', type=float, default='0.0001')
	parser.add_argument('--patch_size', type=int, default=65)
	parser.add_argument('--train_stage', type=int, default=2)
	parser.add_argument('--cuda', type=bool, default=True)
	parser.add_argument('--random_patch', type=bool, default=False)
	parser.add_argument('--policy_conv', type=bool, default=True)
	parser.add_argument('--seed', type=int, default=1007)
	parser.add_argument('--glance_size', type=int, default=112)
	parser.add_argument('--policy_lr ', type=float, default='0.0003')
	parser.add_argument('--feature_map_channels', type=int, default=1280)
	parser.add_argument('--action_dim', type=int, default=49)
	parser.add_argument('--hidden_state_dim', type=int, default=512)
	parser.add_argument('--penalty ', type=float, default='0.5')
	parser.add_argument('--gamma', type=float, default='0.7')
	parser.add_argument('--gpu', type=int, default=0)



	parser.add_argument('--config',
						default='C:/Users/86173/Desktop/mmaction2-master//configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py',
						help='test config file path')
	parser.add_argument('--checkpoint',
						default='C:/Users/86173/Desktop/mmaction2-master/checkpoints/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth',
						help='checkpoint file')




	parser.add_argument(
		'--out',
		default=None,
		help='output result file in pkl/yaml/json format')
	parser.add_argument(
		'--fuse-conv-bn',
		action='store_true',
		help='Whether to fuse conv and bn, this will slightly increase'
			 'the inference speed')
	parser.add_argument(
		'--eval',
		default='top_k_accuracy',
		type=str,
		nargs='+',
		help='evaluation metrics, which depends on the dataset, e.g.,'
			 ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
	parser.add_argument(
		'--gpu-collect',
		action='store_true',
		help='whether to use gpu to collect results')
	parser.add_argument(
		'--tmpdir',
		help='tmp directory used for collecting results from multiple '
			 'workers, available when gpu-collect is not specified')
	parser.add_argument(
		'--options',
		nargs='+',
		action=DictAction,
		default={},
		help='custom options for evaluation, the key-value pair in xxx=yyy '
			 'format will be kwargs for dataset.evaluate() function (deprecate), '
			 'change to --eval-options instead.')
	parser.add_argument(
		'--eval-options',
		nargs='+',
		action=DictAction,
		default={},
		help='custom options for evaluation, the key-value pair in xxx=yyy '
			 'format will be kwargs for dataset.evaluate() function')
	parser.add_argument(
		'--cfg-options',
		nargs='+',
		action=DictAction,
		default={},
		help='override some settings in the used config, the key-value pair '
			 'in xxx=yyy format will be merged into config file. For example, '
			 "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
	parser.add_argument(
		'--average-clips',
		choices=['score', 'prob', None],
		default=None,
		help='average type when averaging test clips')
	parser.add_argument(
		'--launcher',
		choices=['none', 'pytorch', 'slurm', 'mpi'],
		default='none',
		help='job launcher')
	parser.add_argument('--local_rank', type=int, default=0)
	parser.add_argument(
		'--onnx',
		action='store_true',
		help='Whether to test with onnx model or not')
	parser.add_argument(
		'--tensorrt',
		action='store_true',
		help='Whether to test with TensorRT engine or not')



	args = parser.parse_args()
	return args
	
















