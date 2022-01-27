import torch
import torch.nn as nn

#class CenterLoss(nn.Module):
#    """Center loss.
#    
#    Reference:
#    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
#    
#    Args:
#        num_classes (int): number of classes.
#        feat_dim (int): feature dimension.
#    """
#    def __init__(self, num_classes=7257, feat_dim=512, use_gpu=True):
#        super(CenterLoss, self).__init__()
#        self.num_classes = num_classes
#        self.feat_dim = feat_dim
#        self.use_gpu = use_gpu
#        self.loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
# 
#        if self.use_gpu:
#            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
#        else:
#            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
# 
#    def forward(self, x_, labels):#x:   labels:1 dim,(batch_size)
#        """
#        Args:
#            x: feature matrix with shape (batch_size, feat_dim).
#            labels: ground truth labels with shape (batch_size).
#        """
#        x = x_.reshape(x_.size(0), x_.size(1)*x_.size(2))
#        batch_size = x.size(0) #注意x的维度:batch_size*feat_dim
# 
#        #计算了均方误差
#        #pow(x,2)没有改变维度（元素的2次幂）,sum(dim=1)按行求和,
#        #expand(batch_size, self.num_classes):将矩阵每一行扩展到self.num_classes列，注意：扩展之前的行数应该是batch_size行；
#        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#        distmat.addmm_(1, -2, x, self.centers.t())#x维度是batch_size*feat_dim;centers维度是num_classes*feat_dim
# 
# 
# 
#        # 将所有预测正确的位置信息保留下来
#        dist = distmat * labels.float()#矩阵dismat和矩阵mask维度要保持一致
# 
#        # 所有预测正确的/总的样本量
#        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
# 
#        return loss



class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=7257, feat_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            
        nn.init.normal_(self.centers, mean=0, std=1)
        # nn.init.kaiming_normal_(tensor)
        # nn.init.constant_(tensor,0.5)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        x = torch.tensor(x, dtype=torch.float32).cuda()
        labels = labels.cuda()

        batch_size = x.size(0)
        #print("batch size: ", batch_size)
        #print("x size: ", x.size())
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        #distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(x, self.centers.t(),beta=1,alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
