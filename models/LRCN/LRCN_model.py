import torch
import torch.nn as nn
from torchvision import models


# LSTM
class LSTMModel(nn.Module):

    def __init__(self,original_model,num_classes,hidden_size, fc_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size   # 隐藏层数量
        self.num_classes = num_classes   # 类别数
        self.fc_size = fc_size           # 全连接层大小（连接CNN与LSTM）
        # 选择一个特征提取器
        self.features = nn.Sequential(*list(original_model.children())[:-1])   # 特征提取层
        for i, param in enumerate(self.features.parameters()):
            param.requires_grad = False
        self.rnn = nn.LSTM(input_size = fc_size,
                    hidden_size = hidden_size,
                    batch_first = True)                                       # LSTM
        self.fc = nn.Linear(hidden_size, num_classes)                         # logits

        #  CNN+LSTM
    def forward(self, inputs, hidden=None, steps=0):
        # 去掉for循环
        length = len(inputs)    # 帧数
        fs = torch.zeros(inputs[0].size(0), length, self.rnn.input_size).cuda()
        for i in range(length):
            f = self.features(inputs[i])
            f = f.view(f.size(0), -1)
            fs[:, i, :] = f
        outputs, hidden = self.rnn(fs, hidden)
        outputs = self.fc(outputs)
        return outputs


# LRCN
def get_model(checkpoint,num_class):
    original_model = models.__dict__['resnet50'](pretrained=False)         # CNN特征提取器
    model = LSTMModel(original_model,num_classes=num_class,hidden_size=512,fc_size=2048)
    model = model.cuda()
    model_info = torch.load(checkpoint)
    model.load_state_dict(model_info['state_dict'])
    return model