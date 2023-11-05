import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
criterion_md = nn.CrossEntropyLoss()


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

def cla_loss(view1_predict, view2_predict, labels_1, labels_2):   #分类损失即预测标签的损失
    cla_loss1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean()
    cla_loss2 = ((view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()
    return cla_loss1 + cla_loss2

def paw_loss(view1_feature, view1_gcn_feature,view2_feature,labels_1, labels_2):
    cos = lambda x, y: x.mm(y.t()) / (
        (x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    
    view1_feature=view1_feature.to(device)
    view2_feature=view2_feature.to(device)
    view1_gcn_feature=view1_gcn_feature.to(device)

    theta11 = cos(view1_gcn_feature, view1_gcn_feature)
    theta12 = cos(view1_gcn_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()

    lisc = ((1+torch.exp(theta11)).log() - Sim11 * theta11).mean() +((view1_feature - view1_gcn_feature) ** 2).sum(1).sqrt().mean()
    ltsc = ((1+torch.exp(theta12)).log() - Sim12 * theta12).mean() 
    lcsc = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()

    return lisc,ltsc,lcsc




def InfoNCE_loss(x_inp ,x_image_feature,label_adj,image_daj,labels,adj_file, num_classes=10,kappa=0.21,tau=0.3):  #软正对比损失，输入两个特征

    x_inp = F.normalize(x_inp, dim=1)
    x_image_feature = F.normalize(x_image_feature, dim=1)

    label_adj = F.normalize(label_adj, dim=1)
    image_daj = F.normalize(image_daj, dim=1)
    label_adj=label_adj.to(device)
    image_daj=image_daj.to(device)


    label_dim=x_inp.shape[0]  
    image_dim=x_image_feature.shape[0]
    Sim = calc_label_sim(labels, labels).float()
    adj = torch.where(Sim >= 1, 1, Sim)
    adj=adj.to(device)
    adj_statistical = torch.FloatTensor(gen_A(num_classes, tau, adj_file))
    adj_statistical=adj_statistical.to(device)
    #intra img_img
    den_sim_label_label=torch.exp(torch.matmul(x_inp,x_inp.T)/kappa)  #分母的余弦相似性 64.64
    w_ll=torch.matmul(label_adj,label_adj.T)  #64.64
    nume_sim_label_label=torch.matmul(x_inp,x_inp.T)  #64.64
    nume_label_label=torch.exp(torch.mul(w_ll,nume_sim_label_label)/kappa)    #w与相似性对位相乘 64.64得到求和之前的分母
    nume_label_label=nume_label_label.sum(dim=1,keepdim=False) #行求和，64维，对应每一个i的分母
    nume_label_label=nume_label_label.repeat(label_dim,1).T  #给分母按行进行扩充，扩充后需要转置  100.100

    adj_statistical=adj_statistical-torch.diag_embed(torch.diag(adj_statistical))  #把对角线元素置0
    n_label_label=adj_statistical.sum(dim=1,keepdim=False).clamp(min=1.0) #第i个样本所有的正对个数，如果为0则置为1  1个向量
    n_label_label1=adj_statistical.sum(dim=1,keepdim=False).clamp(max=1.0)
    n_label_label1=n_label_label1.sum(dim=0,keepdim=False).clamp(min=1.0)  #1个batch有几个样本存在正对  1个数
    label_loss=-torch.log(torch.div(den_sim_label_label,(nume_label_label-den_sim_label_label)))#减去之后对位相除
    label_loss=torch.mul(adj_statistical,label_loss)  #对求得结果进行约束，有正对的才符合
    label_loss=torch.div(label_loss.sum(dim=1,keepdim=False) ,n_label_label) #行求和，对于第i个样本的对比损失和   64维
    lstc=label_loss.sum(dim=0,keepdim=False)/n_label_label1

    den_sim_img_img=torch.exp(torch.matmul(x_image_feature,x_image_feature.T)/kappa)  #分母的余弦相似性 64.64
    w_vv=torch.matmul(image_daj,image_daj.T)  #64.64
    nume_sim_img_img=torch.matmul(x_image_feature,x_image_feature.T)  #64.64
    nume_img_img=torch.exp(torch.mul(w_vv,nume_sim_img_img)/kappa)    #w与相似性对位相乘 64.64得到求和之前的分母
    # nume_img_img=torch.exp(nume_sim_img_img/t)    #w与相似性对位相乘 64.64得到求和之前的分母
    nume_img_img=nume_img_img.sum(dim=1,keepdim=False) #行求和，64维，对应每一个i的分母

    nume_img_img=nume_img_img.repeat(image_dim,1).T  #给分母按行进行扩充，扩充后需要转置  100.100
    adj=adj-torch.diag_embed(torch.diag(adj))  #把对角线元素置0
    n_img_img=adj.sum(dim=1,keepdim=False).clamp(min=1.0) #第i个样本所有的正对个数，如果为0则置为1  1个向量
    n_img_img1=adj.sum(dim=1,keepdim=False).clamp(max=1.0)
    n_img_img1=n_img_img1.sum(dim=0,keepdim=False).clamp(min=1.0)  #1个batch有几个样本存在正对  1个数
    image_loss=-torch.log(torch.div(den_sim_img_img,(nume_img_img-den_sim_img_img)))#减去之后对位相除

    image_loss=torch.mul(adj,image_loss)  #对求得结果进行约束，有正对的才符合
    image_loss=torch.div(image_loss.sum(dim=1,keepdim=False) ,n_img_img) #行求和，对于第i个样本的对比损失和   64维
    listc=image_loss.sum(dim=0,keepdim=False)/n_img_img1



    return listc,lstc









