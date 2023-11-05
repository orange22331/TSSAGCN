import torch
from scipy.io import loadmat, savemat
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gen_A(num_classes, tau, adj_file):
    result = loadmat(adj_file)
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, None]
    _adj = _adj / _nums
    _adj[_adj < tau] = 0
    _adj[_adj >= tau] = 1
    _adj = _adj.squeeze()
    _adj = _adj * 0.4
    _adj = _adj + np.identity(num_classes, np.float64)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(axis=1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj



def sim_ml_adj(ml_adj):  #生成多标签节点的邻接矩阵用标签之间相似性生成   input=ml_train=np.unique(label_train,axis=0)#删除重复行
    dim=ml_adj.shape[0]  #2441
    ml_train=torch.tensor(ml_adj)
    A=torch.zeros(dim,dim)
    for i in range(dim):
        for j in range(dim):
            if i !=j :
                sim=torch.dot(ml_train[i],ml_train[j])  #分母
                L1=ml_train[i].sum(dim=0)
                L2=ml_train[j].sum(dim=0)
                A[i][j]=sim/(L1+L2-sim)
    return A


def sim_l_ml_adj(ml_adj,num_classes):    #生成单表签节点与多标签的邻接矩阵  输入为多标签节点的去重后的标签即ml_train
    dim=ml_adj.shape[0]
    ml_label=torch.tensor(ml_adj)  #1010.24
    l_label=torch.eye(num_classes)  #24.24
    ml_label = ml_label.float()
    l_label = l_label.float()
    A=torch.zeros(num_classes,dim)
    for i in range(num_classes):
        for j in range(dim):
            if i !=j :
                sim=torch.dot(l_label[i],ml_label[j])  #分母
                L1=l_label[i].sum(dim=0)
                L2=ml_label[j].sum(dim=0)
                A[i][j]=sim/(L1+L2-sim)
    return A

def l_ml_adj(A,ml_adj,lml_adj):  #l-ml的邻接矩阵 输入分别为AL AML  AL_ML  矩阵拼接

    A = A.to(device)
    lml_adj = lml_adj.to(device)
    ml_adj = ml_adj.to(device)

    R0 = torch.cat([A, lml_adj.T], dim=0)  #拼成行,先拼左边#2645.24    
    R1 = torch.cat([lml_adj, ml_adj], dim=0)  #拼成行，先拼右边
    R=torch.cat([R0, R1], dim=1) #按列拼整个
 
    return R
              

def l_mlfeature(inp ,mlfeature):  #合并m_ml特征矩阵
    dim1=inp.shape[0]  #24
    dim2=mlfeature.shape[0]  
    d=inp.shape[1]   #特征维度
    A=torch.zeros(dim1+dim2,d)
    for i in range(dim1):
        A[i]=inp[i]
    for j in range(dim2):
        A[j+dim1]=mlfeature[j]
    return A





def finding(labels,Al_ml_adj,ml_train,x_ml):
    batch=labels.shape[0]  #100
    ml_dim=x_ml.shape[0]  #2441
    f_dim=x_ml.shape[1]
    adj=Al_ml_adj[24:24+ml_dim,:]
    A=torch.zeros(batch,ml_dim+24)
    feature_ml=torch.zeros(batch,f_dim)

    labels=torch.tensor(labels)
    Al_ml_adj=torch.tensor(Al_ml_adj)
    ml_train=torch.tensor(ml_train)
    
    labels = labels.to(device)
    ml_train=ml_train.to(device)

    a=x_ml.shape[0]
    if a ==1024:
        x_ml=x_ml.T

    for i in range(batch):
        for j in range(ml_dim):
            if labels[i].equal(ml_train[j])==True:
                A[i]=adj[j]
                # print(x_ml.shape)
                feature_ml[i]=x_ml[j]
    return A,feature_ml 







"""
#based on http://stackoverflow.com/questions/7323664/python-generator-pre-fetch
"""

import threading
import sys

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """

        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.

        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


# decorator
class background:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(gen(*args, **kwargs), max_prefetch=self.max_prefetch)

        return bg_generator
