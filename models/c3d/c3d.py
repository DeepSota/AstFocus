#from .hmdb51_opts import parse_opts as hmdb51_c3d_opts
#from .ucf101_opts import parse_opts as ucf101_c3d_opts

from models.c3d.generate_models import generate_model as c3d_gen_model
import pickle
from models.c3d_k400_model.opts import parse_opts
from models.c3d_k400_model.model import generate_model
from models.c3d_k400_model.mean import get_mean
import torch

class DictToAttr(object):
    def __init__(self, args):
        for i in args.keys():
            setattr(self, i, args[i])

            
def generate_model_c3d(dataset):
    assert dataset in ['hmdb51','ucf101','k400']
    if dataset == 'hmdb51':
        with open('./models/c3d/hmdb51_params.pkl', 'rb') as ipt:
            model_opt = pickle.load(ipt)
        model_opt = DictToAttr(model_opt)
        model, parameters = c3d_gen_model(model_opt)
    elif dataset == 'ucf101':
        with open('./models/c3d/ucf101_params.pkl', 'rb') as ipt:
            model_opt = pickle.load(ipt)
            model, parameters = c3d_gen_model(model_opt)
    elif dataset == 'k400':
        opt = parse_opts()
        opt.mean = get_mean()
        opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        opt.sample_size = 112
        opt.sample_duration = 16
        opt.n_classes = 400

        model = generate_model(opt)
        print('loading model {}'.format(opt.model))
        model_data = torch.load(opt.model)
        assert opt.arch == model_data['arch']
        model.load_state_dict(model_data['state_dict'])

    return model