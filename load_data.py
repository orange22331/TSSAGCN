import torch
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from util import BackgroundGenerator
import numpy as np


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels


    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]

        return img, text, label

    def __len__(self):
        count = len(self.images)
        return count


class SingleModalDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        count = len(self.data)
        return count


def get_loader(path, batch_size):
    img_train = loadmat(path + "train_img.mat")['train_img']  #18013.4096
    img_test = loadmat(path + "test_img.mat")['test_img']  #2002.4096
    text_train = loadmat(path + "train_txt.mat")['train_txt']
    text_test = loadmat(path + "test_txt.mat")['test_txt']
    label_train = loadmat(path + "train_lab.mat")['train_lab']  #18013.24
    label_test = loadmat(path + "test_lab.mat")['test_lab']




    # 取五千3副作为训练集
    img_train=img_train[0:1000,:] 
    text_train=text_train[0:1000,:]
    label_train=label_train[0:1000,:]
    img_test = img_test[0:1000,:]   #2002.4096
    text_test = text_test[0:1000,:] 
    label_test = label_test[0:1000,:] 
    image_all_dim=img_train.shape[0]


    split = img_train.shape[0] // 5
   
    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}


   
    shuffle = {'train': True, 'test': False}
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x] )
                   for x in ['train', 'test']}
    dataloader = {x: DataLoaderX(dataset[x], batch_size=batch_size,
                                     shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    img_dim = img_train.shape[1]  #4096
    text_dim = text_train.shape[1]  #300
    num_class = label_train.shape[1]  #24
    test_num=img_test[0]
    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_train'] = img_train  #18013 4096
    input_data_par['text_train'] = text_train  #18013 300
    input_data_par['label_train'] = label_train  #18013 24
    input_data_par['img_dim'] = img_dim   #2002.4096
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    input_data_par['image_all_dim'] = image_all_dim
    input_data_par['test_num'] = test_num


    return dataloader, input_data_par







# dataset = 'mirflickr'  # 'mirflickr' or 'NUS-WIDE-TC21' or 'MS-COCO'


#     # data parameters
# DATA_DIR = 'data/' + dataset + '/'

# print('...Data loading is beginning...')
# batch_size=64
# INCOMPLETE=False
# data_loader, input_data_par = get_loader(DATA_DIR, batch_size, INCOMPLETE, False)
# print(input_data_par['img_train'].shape)
# print(input_data_par['text_train'].shape)
# print(input_data_par['label_train'].shape)
# print(input_data_par['img_test'].shape)
# print(input_data_par['text_test'].shape)





