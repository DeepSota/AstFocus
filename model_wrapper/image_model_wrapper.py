import torch
import torch.nn as nn
import torch.nn.functional as F


# resnet特征提取
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model, extracted_layers):
        super(ResNetFeatureExtractor, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = model.fc
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        if 'conv1' in self.extracted_layers:
            outputs += [x]
            if len(outputs) == len(self.extracted_layers):
                return outputs
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if 'maxpool' in self.extracted_layers:
            outputs += [x]
            if len(outputs) == len(self.extracted_layers):
                return outputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if 'layer3' in self.extracted_layers:
            outputs += [x]
            if len(outputs) == len(self.extracted_layers):
                return outputs
        x = self.layer4(x)
        if 'layer4' in self.extracted_layers:
            outputs += [x]
            if len(outputs) == len(self.extracted_layers):
                return outputs

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if 'avgpool' in self.extracted_layers:
            outputs += [x]
        if 'fc' in self.extracted_layers:
            x = self.fc(x)
            outputs += [x]
        return outputs


# Inception特征提取
class InceptionFeatureExtractor(nn.Module):
    def __init__(self, model, extracted_layers, transform_input):
        super(InceptionFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.fc = model.fc
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        if 'mix7' in self.extracted_layers:
            outputs += [x]
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        if 'avgpool' in self.extracted_layers:
            outputs += [x]
        # 2048
        if 'fc' in self.extracted_layers:
            x = self.fc(x)
            outputs += [x]
        # 1000 (num_classes)
        return outputs


# densennet特征提取
class DensenetFeatureExtractor(nn.Module):
    def __init__(self, model, extracted_layers):
        super(DensenetFeatureExtractor, self).__init__()
        self.extracted_layers = extracted_layers
        self.features = model.features
        self.classifier = model.classifier

    def forward(self, x):
        outputs = []
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        if 'avgpool' in self.extracted_layers:
            outputs += [out]
        if 'fc' in self.extracted_layers:
            out = self.classifier(out)
            outputs += [out]
        return outputs


# 最初扰动生成器
class TentativePerturbationGenerator():

    def __init__(self, extractors, part_size=100, preprocess=True, device=0):
        self.r = None
        self.extractors = extractors    # 分类器
        self.part_size = part_size      # 划分块儿的大小
        self.preprocess = preprocess    # 是否需要对输入进行处理（默认为真）
        self.device = device            # 指定运行设备

    # 设置有目标攻击参数
    def set_targeted_params(self, target_vid, random_mask=1.):
        self.target = True                    # 有目标
        self.random_mask = random_mask        # 随机mask
        self.target_feature = []              # 模型预测的结果
        with torch.no_grad():
            target_vid = target_vid.clone().cuda(self.device)
            if self.preprocess:
                # IMAGENET数据预处理
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=target_vid.get_device())[None, :,
                       None, None]
                std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=target_vid.get_device())[None, :,
                      None, None]
                target_vid = target_vid.sub_(mean).div_(std)  # 标准化
            for extractor in self.extractors:
                outputs = extractor(target_vid)               # 模型输出结果
                self.target_feature.append(outputs[0].view((outputs[0].size(0), -1)))

    # 设置无目标攻击参数
    def set_untargeted_params(self, ori_video, random_mask=1., translate=0., scale=1.):
        self.target = False                  # 有目标
        self.translate = translate           # 数据类型的转换
        self.scale = scale                   # 缩放的尺度
        self.random_mask = random_mask       # 随机mask
        self.target_feature = []             # 结果
        with torch.no_grad():
            ori_video = ori_video.clone().cuda(self.device)    # 复制原始视频
            for extractor in self.extractors:
                outputs = extractor(ori_video)                 # 获得输出
                output_size = outputs[0].size()                # 获得输出大小
                del outputs
                r = torch.randn(output_size, device=self.device) * self.scale + self.translate  # 随机产生
                r = torch.where(r >= 0, r, -r)                 # 如果r大于，则选r否则选择-r，保证输出结果为正数
                self.target_feature.append(r.view((output_size[0], -1)))

    # 反向更新视频帧中的指定区域
    def backpropagate2frames(self, part_vid, start_idx, end_idx, random):
        part_vid.requires_grad = True      # 划分区域保留梯度
        processed_vid = part_vid.clone()   # 对划分区域进行复制
        # 数据的预处理
        if self.preprocess:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=part_vid.get_device())[None, :, None,
                   None]
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=part_vid.get_device())[None, :, None,
                  None]
            processed_vid = processed_vid.sub_(mean).div_(std)
        for idx, extractor in enumerate(self.extractors):
            perturb_loss = 0
            o = extractor(processed_vid)[0]   # 提取的分类结果
            o = o.view((o.size(0), -1))
            if self.target:
                if random:
                    # 采用部分值随机
                    mask = torch.rand_like(o) <= self.random_mask
                    perturb_loss += nn.MSELoss(reduction='mean')(torch.masked_select(o, mask),
                                                                             torch.masked_select(
                                                                                 self.target_feature[idx][
                                                                                 start_idx:end_idx], mask))
                else:
                    # 当前区域的损失
                    perturb_loss += nn.MSELoss(reduction='mean')(o, self.target_feature[idx][
                                                                                start_idx:end_idx])
            else:
                r = torch.randn_like(o) * self.scale + self.translate
                r = torch.where(r >= 0, r, -r)
                perturb_loss += nn.MSELoss(reduction='mean')(o, r)  # 随机后计算损失
            # print(perturb_loss.item())
        perturb_loss.backward()  # 反向传播
        extractor.zero_grad()    # 更新梯度
        sign_grad = torch.sign(part_vid.grad)  # 获取梯度符号
        return sign_grad

    # 创建对抗样本的方向
    def create_adv_directions(self, vid, random=True):
        # 视频格式: [num_frames, c, w, h]
        vid = vid.clone().cuda(self.device)
        assert hasattr(self, 'target'), 'Error, AdvDirectionCreator\' mode unset'
        start_idx = 0
        adv_directions = []           # 对抗样本的方向
        part_size = self.part_size    # 划分块的大小
        while start_idx < vid.size(0):
            adv_directions.append(self.backpropagate2frames(vid[start_idx:min(start_idx + part_size, vid.size(0))],
                                                            start_idx, start_idx + part_size, random))  # 返回一个方向，并更新响应的块
            start_idx += part_size    # 开始块移动
        adv_directions = torch.cat(adv_directions, 0)  # 拼合
        return adv_directions

    # 调用此对象，直接运行此函数
    def __call__(self, vid):
        return self.create_adv_directions(vid)
