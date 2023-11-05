import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter
from torchvision import models

from util import *
batch_size = 100

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features).to(device))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input=input.to(device)
        # self.weight=self.weight.to(device)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features

class TxtMLP(nn.Module):
    def __init__(self, code_len=300, txt_bow_len=1386, num_class=24):
        super(TxtMLP, self).__init__()
        self.fc1 = nn.Linear(txt_bow_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.classifier = nn.Linear(code_len, num_class)

    def forward(self, x):
        feat = F.leaky_relu(self.fc1(x), 0.2)
        feat = F.leaky_relu(self.fc2(feat), 0.2)
        predict = self.classifier(feat)
        return feat, predict

class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        return out

class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        return out

class ReverseLayerF(Function):   #梯度反转，因为生成器跟辨别器梯度所求不一样
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ModalClassifier(nn.Module):   #分类层，也就是辨别器的参数   单表签辨别其
    """Network to discriminate modalities"""

    def __init__(self, input_dim=40):
        super(ModalClassifier, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, input_dim // 4)
        self.denseL2 = nn.Linear(input_dim // 4, input_dim // 16)
        self.denseL3 = nn.Linear(input_dim // 16, 2)

    def forward(self, x):
        x = ReverseLayerF.apply(x, 1.0)
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        out = F.relu(self.denseL2(out))
        out = self.denseL3(out)
        return out

class ImgDec(nn.Module):
    """Network to decode image representations"""
    def __init__(self, input_dim=1024, output_dim=4096, hidden_dim=2048):
        super(ImgDec, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, hidden_dim)
        self.denseL2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        out = F.relu(self.denseL2(out))
        return out





class TextDec(nn.Module):
    """Network to decode image representations""" #解码

    def __init__(self, input_dim=1024, output_dim=300, hidden_dim=512):
        super(TextDec, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, hidden_dim)
        self.denseL2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.leaky_relu(self.denseL1(x), 0.2)
        out = F.leaky_relu(self.denseL2(out), 0.2)
        return out







class TSSAGCN(nn.Module):   #用Pgnn
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=4096, num_classes=10, in_channel=300,
                tau=0.3,upsilon1=0.5,upsilon2=0.5,upsilon3=0.1,alph=0.5,adj_file=None, inp=None,adj_content=None,adj_image=None, n_layers=3,image_all_dim=0):
        super(TSSAGCN, self).__init__()
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)

        self.num_classes = num_classes
        self.upsilon1=upsilon1
        self.upsilon2=upsilon2
        self.upsilon3=upsilon3
        self.alph=alph
        self.adj_content=adj_content   #标签邻接矩阵
        self.adj_image=adj_image  #图像邻接矩阵
        self.gnn = GraphConvolution
        
        self.n_layers = n_layers
        self.label_dim=inp.shape[0]   #24


        self.relu = nn.LeakyReLU(0.2)
        self.lrn = [self.gnn(in_channel, 512)]  #第一层

        self.lrn.append(self.gnn(512, 1024))  #第二层
        self.lrn.append(self.gnn(1024, 300))  #第三层   输出300维

        self.lrn_image = [self.gnn(img_input_dim, 512)]  #第一层
        self.lrn_image.append(self.gnn(512, 1024))  #第二层
        self.lrn_image.append(self.gnn(1024, 4096))  #第三层   输出300维


        # for i in range(1, self.n_layers):
        #     self.lrn.append(self.gnn(minus_one_dim, minus_one_dim))
        for i, layer in enumerate(self.lrn):
            self.add_module('lrn_{}'.format(i), layer)

        self.hypo1 = nn.Linear(300, minus_one_dim)  #输出1024，全连接层
        self.hypo2 = nn.Linear(4096, minus_one_dim)  #输出1024，全连接层

        adj_statistical = torch.FloatTensor(gen_A(num_classes, tau, adj_file))  #标签节点邻接矩阵
        self.adj_statistical = Parameter(adj_statistical, requires_grad=False)   #标签节点邻接矩阵  24.24
        self.adj_label_dadptive = Parameter(0.02 * torch.rand(self.label_dim))  #自适应邻接矩阵24 24
        self.adj_content = Parameter(self.adj_content, requires_grad=False)
        self.adj_image_adptive = Parameter(0.02 * torch.rand(image_all_dim))  #自适应邻接矩阵image_all_dim，image_all_dim
        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature_img, feature_text,batch,img_train,evals):
        view1_feature = self.img_net(feature_img)  #100.1024
        view2_feature = self.text_net(feature_text)
        if evals =="0":
            img_train = torch.FloatTensor(img_train)
            self.image_dim=view1_feature.shape[0]   #batch
            self.adj_image=self.adj_image.to(device)
            self.adj_image_adptive=self.adj_image_adptive.to(device)
            img_train=img_train.to(device)
            view1_feature=view1_feature.to(device)
            view2_feature=view2_feature.to(device)

            label_adj = gen_adj(torch.relu(self.upsilon1 *self.adj_statistical + self.upsilon2 * self.adj_content+self.upsilon3 * self.adj_label_dadptive))  #最终自适应的邻接矩阵对邻接矩阵进行操作，归一化和×度矩阵   最终label的矩阵
            image_daj=gen_adj((torch.relu(self.adj_image+self.alph*self.adj_image_adptive)))

            img_train[batch*batch_size:(batch+1)*batch_size,:]=view1_feature  #替换一部分


            layers = []
            x_inp = self.inp  #24.300
            x_inp=x_inp.to(device)  
            label_adj=label_adj.to(device) 
            for i in range(3):
                x_inp = self.lrn[i](x_inp, label_adj)
                x_inp = self.relu(x_inp)
                layers.append(x_inp)   #输出label就是300维，经过全连接层hypo映射到1024

        
            image_daj=image_daj.to(device) 
            x_image = img_train  #24.300  
            x_image=x_image.to(device) 
            for i in range(3):
                x_image = self.lrn_image[i](x_image, image_daj)
                x_image = self.relu(x_image)
                layers.append(x_image)   #输出label就是300维，经过全连接层hypo映射到1024

            x1 = self.hypo1(x_inp)   #24.1024   经过全连接层出来300维
            x2 = self.hypo2(x_image)  #经过GCN以后出来的图像特征，

            view1_gcn_feature=x2[batch*batch_size:(batch+1)*batch_size,:]  #经过GCN以后出来的图像特征，分batch后
            # print(x1.shape)
            # print(x2.shape)
            # print(view1_feature.shape)
            # print(view1_gcn_feature.shape)
            # print(view2_feature.shape)



            #在这使用单表签节点的特征24.4096进行标签预测
            norm_img = torch.norm(view1_gcn_feature, dim=1)[:, None] * torch.norm(x1, dim=1)[None, :] + 1e-6  #100.24
            norm_txt = torch.norm(view2_feature, dim=1)[:, None] * torch.norm(x1, dim=1)[None, :] + 1e-6  #100.24

            x1 = x1.transpose(0, 1) #本来是24，4096  反转之后是4096，24
            y_img = torch.matmul(view1_gcn_feature, x1)  #生成预测100，24
            y_text = torch.matmul(view2_feature, x1) #100，24
            y_img = y_img / norm_img   #100.24
            y_text = y_text / norm_txt  #100.24
            x1=x1.T
            # print(x1.shape)
            # print(x2.shape)
            # print(norm_img.shape)
            # print(norm_txt.shape)
            # input()
            #输出的x1 24.4096  x2  18000, 4096   
            return view1_feature, view2_feature, y_img, y_text,view1_gcn_feature,x1 ,x2,label_adj,image_daj
        else: 
            return view1_feature, view2_feature
            
            

               



