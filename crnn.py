import torch.nn as nn
import utils
import torch.nn.functional as F
import torch
import torch as tc
import numpy as np
from Yourmodels import resnet,rnn

torch.set_printoptions(profile="full")

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        
        self.resnet=resnet(pretrained=False)
        self.rnn = rnn() 

    def forward(self, input):
        conv=self.resnet(input) # [batch channel height=1 width] 
        conv = conv.squeeze(2) # [batch channel width]
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)

        return (output,conv)
    
def get_features(batch_num, char_pos_return, embedding, label_return):
    """
    根据字符的位置从相应时间步中获取 features
    Args:
        char_pos_return: [[24, 42], [24, 46], [24, 49], [24, 52], [24, 53], [24, 55], [24, 60], [24, 62], [24, 66], [24, 69], [24, 74], [24, 78], [24, 117], [33, 52], [33, 55], [33, 57],[33, 60], [33, 63], [33, 117], [35, 38], [35, 43], ] 其中24表示batch中第24张图，42 指的是其在第42张图的第42个分片
        embedding: 输入全连接层的 tensor ----->   batch_size, w, c
    Returns:
        features: 字符对应的 feature
    """
    features = torch.tensor([]).cuda()
    #embedding = embedding.cpu().detach().numpy() # (506, 119, 512)
    img_start_index = 0
    for key,value in batch_num.items():
        #print(img_start_index)
        feature_img = torch.tensor([]).cuda()
        
        # 这里key指的是符合条件（pred num == GT num）的batchsize中可用的图片id，因此下面去embedding的时候会有问题
        # print(key) # key 值应该是24,33
        img_num_of_batch = int(key) 
        for i in range(value):
            pos_index = char_pos_return[i+img_start_index][1].type(torch.long).cuda()
            feature_one_char = embedding[img_num_of_batch][pos_index,:].cuda()
            #print(len(feature_one_char)) # 512
            feature_img = torch.cat((feature_img, feature_one_char),0).cuda()
            #print(feature_img.size())
        features = torch.cat((features,feature_img),0).cuda()
        img_start_index = img_start_index + value
    features = features.view(-1, 512).cuda()
    #print(features.size())  # [6414, 512]
    new_label_return = np.array(label_return)
    label_return = torch.from_numpy(new_label_return).cuda()
    #print(label_return.size()) 
    #return None,None
    return features, label_return

def raw_pred_to_features(raw_pred1, char_num, labels, embedding1):
    """
    得到用于计算 centerloss 的 embedding features，和对应的标签
    Args:
        raw_pred: 原始的预测结果，形如 [[6941, 6941, 0, 6941, 6941, 5, 6941], …]
        labels: 字符标签，形如 [0,5,102,10,…]
        embedding: 全连接的输入张量
        char_num: 每个样本的字符数，用于校验是否可以对齐
        # poses: 初始化的字符位置
    Returns:
        self.embedding: embedding features
        # self.char_label: 和 embedding features 对应的标签
        # self.char_pos: 和 embedding features 对应的字符位置
    """
    # 判断是否为预测的字符
    embedding = embedding1.permute(1, 0, 2) # w,batch_size,c -> batch_size, w, c
    raw_pred = raw_pred1.permute(1, 0)#w, batchsize -> batchsize, w
    

    #print(len(labels)) # 6680 输入的label是一维的
    
    char_pos_return = torch.tensor([]).cuda()
    #label_return_tensor = torch.tensor([]).cuda()
    label_return = []
    batch_num_mark = torch.tensor(0)
    batch_num = {}
    i = 0
    for i in range(raw_pred.shape[0]):
        is_char = tc.le(raw_pred[i], 7193 - 1).cuda()
        x = raw_pred[i][:-1].cuda()
        b = raw_pred[i][1:].cuda()
        char_rep = tc.eq(x, b).cuda()
        tail = tc.gt(raw_pred[i][:1], 7193 - 1).cuda()
        char_rep = tc.cat([char_rep, tail]).cuda()
        # remove zero whose value is true        
        mask = tc.le(raw_pred[i], 1)
        mask_or = tc.logical_or(mask, char_rep)
        char_no_rep = tc.logical_and(is_char, tc.logical_not(mask_or).cuda()).cuda()
        #char_no_rep = tc.logical_and(is_char, tc.logical_not(char_rep).cuda()).cuda()
        char_pos = tc.nonzero(char_no_rep, as_tuple=False).cuda()
        label = labels[:char_num[i]]
        labels = labels[char_num[i]:]
        pre_len = char_pos.size()[0]
        if not torch.eq(torch.tensor(pre_len), char_num[i]):
            continue
        batch_i = torch.zeros_like(char_pos).cuda()
            
        batch_i = batch_i + i
        char_pos_index = tc.cat([batch_i, char_pos], 1).cuda()
        char_pos_return = tc.cat([char_pos_return, char_pos_index],0).cuda()
        #print("char pos return: ", char_pos_return.size())
        #label_return_tensor = tc.cat([label_return_tensor, label],1).cuda()
        #print("label_return_tensor: ", label_return_tensor)
        # 将可以预测出字符数和gt一致的label, pos保存起来
        label_return = label_return + label 
        batch_num_mark = batch_num_mark + torch.tensor(1)
        batch_num[i] = pre_len
    #print("-========================")
    # 根据字符位置得到字符的 embedding
    if batch_num_mark != 0:
        # char_pos_return: [[24, 42], [24, 46], [24, 49], [24, 52], [24, 53], [24, 55], [24, 60], [24, 62], [24, 66], [24, 69], [24, 74], [24, 78], [24, 117], [33, 52], [33, 55], [33, 57], [33, 60], [33, 63], [33, 117], [35, 38], [35, 43], [35, 47], [35, 52], [35, 57], [35, 62], [35, 67], [35, 76], [35, 81], [35, 117], [49, 38], [49, 43], [49, 47], [49, 52], [49, 57], [49, 67], [49, 71], [49, 76], [49, 81], [49, 117], [72, 43], [72, 51], [72, 53], [72, 57], [72, 61], [72, 64], [72, 69], [72, 72], [72, 77], [72, 117] #  其中24表示batch中第24张图，42表示slice42 
        # label_return: [tensor([7165, 7246, 7221, 7225, 7166, 7209, 7164, 7187, 7182, 7246, 7166, 7204,7182], dtype=torch.int32), tensor([7182, 7184, 7163, 7164, 7246, 7182], dtype=torch.int32), tensor([5788,  512, 5182, 4295,  443, 5054, 5180,  186, 3120, 5205],
        embedding, label_tensor = get_features(batch_num, char_pos_return, embedding, label_return)
        return embedding, label_tensor #label_return是一维的数据
    else:
        return None, None


