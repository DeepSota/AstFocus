import time
from Utils.utils import *
from attack.attackSTRE import targeted_video_attack
from Adaselection.config import*
from model_wrapper.vid_model_top_k import C3D_K_Model, LRCN_K_Model
import numpy as np

def main(args):

    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])
    model_name = 'c3d'
    dataset_name = 'hmdb51'
    print('load {} dataset'.format(dataset_name))
    # test_data = generate_dataset(model_name, dataset_name)
    # train_data = generate_train_dataset(model_name,dataset_name)
    print('load {} model'.format(model_name))
    model = generate_model(model_name, dataset_name)
    print('Initialize model')

    try:
        model.cuda()

    except:
        pass
    if model_name == 'c3d':
        vid_model = C3D_K_Model(model)
    else:
        vid_model = LRCN_K_Model(model)

    # 获取需要攻击的样本的下坐标
    # attacked_ids = get_samples(model_name, dataset_name)
    attacked_ids = np.arange(0, 50)
    def GetPairs( idx):
        x0 = torch.from_numpy(np.load('{}_{}_np_vid/{}.npy'.format(model_name, dataset_name, idx)))
        x0 = image_to_vector(model_name, x0)
        print('----vid{}-----'.format(idx))
        return x0.cuda(), x0

    test = [0]
    t_s = time.time()
    TEST_OBJ = test
    name = str('test ')
    method = str('AST')
    for m in range(len(TEST_OBJ)):
        XX = TEST_OBJ[m]
        result_root = 'T_results/{}/{}_{}_{}_{}'.format(method, model_name,dataset_name,name, XX)
        #os.mkdir(result_root)
        total_iternum_average = 0
        total_pertubation_average = 0
        NUM = 0
        success_num = 0
        max_iter =30000
        metric_path = os.path.join(result_root, 'metric.txt')
        for idx in range(0, len(attacked_ids)):
            vid, x0 = GetPairs(attacked_ids[idx])
            '--------------------Attack-----------------------'
            print('THE {}th Attacking.....'.format(attacked_ids[idx]))
            ori_vid_batch = vid[None, :]
            top_val, label, logits = vid_model(ori_vid_batch)
            _, h = logits.sort()
            target_label = h[0, -5]
            ori_vid = vid.clone()
            t1 = time.time()
            res, iter_num, adv_vid = targeted_video_attack(XX, vid_model, ori_vid,  x0, label, target_label,  args)
            t2 = time.time()
            print('using {} time'.format(t2-t1))
            NUM += 1
            '--------------------complete-----------------------'
            AP = pertubation(vid, adv_vid)
            print('The average pertubation of video is: {}'.format(AP.cpu()))
            total_pertubation_average += AP.cpu()
            if res:
                # 成功
                print('untargeted attack succeed using {} quries'.format(iter_num))
                total_iternum_average += iter_num
                success_num += 1
            else:
                # 失败
                total_iternum_average += max_iter
                print('--------------------Attack Fails-----------------------')

            print('total iternum average is {} '.format(total_iternum_average))
            print('total pertubation average is {} '.format(total_pertubation_average))
            print('fail number is {} '.format(NUM-success_num))

        t_e = time.time()
        print('TOTAL TIME {} '.format(t_e - t_s))
        total_iternum_averages = total_iternum_average /(NUM)
        total_pertubation_averages = total_pertubation_average /(NUM)
        print('total iternum average is {} '.format(total_iternum_averages))
        print('total pertubation average is {} '.format(total_pertubation_averages*255))
        print('total success rate  is {} '.format(success_num*100 / NUM))
        f = open(metric_path, 'a')
        f.write('\n')
        f.write(str('********************** total results ***********************'))
        f.write('\n')
        f.write(str('XX is {} '.format(XX)))
        f.write('\n')
        f.write(str('total iternum average is {} '.format(total_iternum_averages)))
        f.write('\n')
        f.write(str('total pertubation average is {} '.format(total_pertubation_averages*255)))
        f.write('\n')
        f.write(str('success rate is {} '.format(success_num*100 / NUM)))

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main(args)


class Obj:
    def __init__(self, info):
        self.info = info