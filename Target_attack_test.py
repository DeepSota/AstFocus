from Utils.utils import *
from configs.config import*
from model_wrapper.vid_model_top_k import C3D_K_Model, SLOWFAST_K_Model, TSN_K_Model, TSM_K_Model, C3D_K_Model_k400

import numpy as np
from attack.attackZERO import targeted_video_attack


def main(args):

        model_name = args.model_name
        dataset_name = args.dataset_name
        gpus = args.gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])
        print('load {} dataset'.format(dataset_name))
        print('load {} model'.format(model_name))
        model = generate_model(model_name, dataset_name)
        print('Initialize model')
        attacked_ids = [1, 2, 3]

        try:
            model.cuda()
        except:
            pass

        if model_name == 'c3d':
            if dataset_name == 'k400':
                vid_model = C3D_K_Model_k400(model)
            else:
                vid_model = C3D_K_Model(model)
        if model_name == 'slowfast':
            vid_model = SLOWFAST_K_Model(model)
        if model_name == 'tsm':
            vid_model = TSM_K_Model(model)
        if model_name == 'tsn':
            vid_model = TSN_K_Model(model)


        def GetPairs_ori( idx):
            x0 = torch.from_numpy(np.load('numpy_video/{}.npy'.format(idx)))
            # x0 = image_to_vector(model_name, dataset_name, x0)  # 将视频归一化在[0,1]之间
            if x0.size(0) == 3:
                x0 = x0.transpose(1, 0)
            return x0.cuda(), x0



        method = str('AST')
        result_path ='UN/ASTii'
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        result_root = 'UN/ASTii/{}_{}_{}'.format(method, args.model_name, args.dataset_name)
        if not os.path.exists(result_root):
           os.mkdir(result_root)
        total_iternum_average = 0
        total_pertubation_average = 0
        NUM = 0
        success_num = 0
        metric_path = os.path.join(result_root, 'metric.txt')

        for idx in range(0, len(attacked_ids)):
            vid, x0 = GetPairs_ori(attacked_ids[idx])
            ori_vid_batch = vid
            top_val, label, logits = vid_model(ori_vid_batch[None, :])
            _, h = logits.sort()
            target_label = h[0, -3].view(-1, 1)
            '--------------------Attack-----------------------'
            print('THE {}th Attacking.....'.format(attacked_ids[idx]))
            res, iter_num, adv_vid = targeted_video_attack(vid_model, vid, x0, label, target_label, model_name,   args)
            NUM += 1
            '--------------------complete-----------------------'
            AP = pertubation(vid, adv_vid)
            print('The average pertubation of video is: {}'.format(AP.cpu()))
            total_pertubation_average += AP.cpu()
            if res:
                # 成功
                total_iternum_average += iter_num
                f = open(metric_path, 'a')
                f.write(str('----------------{}-------------------'.format(attacked_ids[idx])))
                f.write('\n')
                f.write(str(iter_num))
                f.write('\n')
                f.write(str(AP.cpu()*255))
                f.write('\n')
                f.close()
                print('untargeted attack succeed using {} quries'.format(iter_num))
                success_num += 1
            else:
                # 失败
                total_iternum_average += iter_num
                metric_path = os.path.join(result_root, 'metric.txt')
                f = open(metric_path, 'a')
                f.write(str('----------------{}-------------------'.format(attacked_ids[idx])))
                f.write('\n')
                f.write(str('Attack Fails'))
                f.write('\n')
                f.write(str(AP.cpu()*255))
                f.write('\n')
                f.close()
                print('--------------------Attack Fails-----------------------')


            print('total iternum average is {} '.format(total_iternum_average))
            print('total pertubation average is {} '.format(total_pertubation_average))
            print('fail number is {} '.format(NUM-success_num))

        total_iternum_averages = total_iternum_average / (NUM)
        total_pertubation_averages = total_pertubation_average / (NUM)
        print('total iternum average is {:.4} '.format(total_iternum_averages))
        print('total pertubation average is {:.4} '.format(total_pertubation_averages*255))
        print('total success rate  is {:.4} '.format(success_num*100 / NUM))
        f = open(metric_path, 'a')
        f.write('\n')
        f.write(str('********************** total results ***********************'))
        f.write('\n')
        f.write(str('total iternum average is {:.4} '.format(total_iternum_averages)))
        f.write('\n')
        f.write(str('total pertubation average is {:.4} '.format(total_pertubation_averages*255)))
        f.write('\n')
        f.write(str('success rate is {:.4} '.format(success_num*100 / NUM)))
        f.write('\n')



if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main(args)


class Obj:
    def __init__(self, info):
        self.info = info
