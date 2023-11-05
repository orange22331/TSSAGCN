from __future__ import print_function
from __future__ import division
import time
import copy
from util import *
import numpy as np
import torch
import torch.nn as nn
import torchvision
import datetime
from evaluate import fx_calc_map_label
from loss import cla_loss, InfoNCE_loss,paw_loss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


000

def train_model(model, data_loaders,input_data_par, optimizer,train_phase, lamuda1,lamuda2,lamuda3,kappa, num_epochs,adj_file):
    since = time.time()

    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history = []
    num_classes=24
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    batch=-1
    for epoch in range(num_epochs):
        # alpha=alpha+0.05
        # print('alpha {}'.format(alpha ))
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode
                model.train()
                batch=batch+1
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for imgs, txts, labels in data_loaders[phase]:
                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")
                # print(imgs.shape)
                # print(txts.shape)
                # print(labels.shape)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.float().cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    img_train=  input_data_par['img_train']  #整个图像数据集

                    label_train=input_data_par['label_train'] 
                    label_train = torch.FloatTensor(label_train)
  
                    # Forward
                    view1_feature, view2_feature, view1_predict, view2_predict,view1_gcn_feature,x_inp ,x_image_feature,label_adj,image_daj= model(imgs, txts,batch,img_train,evals="0")     
                    lcls = cla_loss(view1_predict, view2_predict, labels, labels)   #分类损失 V1为图像，v2为文本
                    listc,lstc = InfoNCE_loss(x_inp ,x_image_feature,label_adj,image_daj,label_train,adj_file, num_classes,kappa,tau=0.3)  #软正对比损失
                    lisc,ltsc,lcsc = paw_loss(view1_feature, view1_gcn_feature,view2_feature,labels, labels)  #dcmh损失
 
                    l_t=lamuda1*ltsc
                    l_r=lcls+lamuda2*(lisc+lcsc)+lamuda3*(listc+lstc)
                    # print(lcls)
                    # print(lisc)
                    # print(lcsc)
                    # print(listc)
                    # print(lstc)

                    # print(l_t)
                    # print(l_r)
                    # input()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #select train text or image
                            if train_phase=="image":
                                l_r.backward()
                                optimizer.step()
                            if train_phase=="text":
                                l_t.backward()
                                optimizer.step()

            time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # print(running_loss)
            # print(len(data_loaders[phase].dataset))

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            if phase == 'train':
                print(time2)
                print('Train Loss: {:.7f}'.format(epoch_loss))
            if phase == 'test':
                t_imgs, t_txts, t_labels = [], [], []
                with torch.no_grad():
                    for imgs, txts, labels in data_loaders['test']:
                        if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                            labels = labels.float().cuda()

                        t_view1_feature, t_view2_feature= model(imgs, txts,batch,img_train,evals="1")
                        t_imgs.append(t_view1_feature.cpu().numpy())
                        t_txts.append(t_view2_feature.cpu().numpy())
                        t_labels.append(labels.cpu().numpy())

                t_imgs = np.concatenate(t_imgs)
                t_txts = np.concatenate(t_txts)
                t_labels = np.concatenate(t_labels)

                img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
                txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

                print('{} Loss: {:.7f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))
 
            # deep copy the model
            if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
                best_acc = (img2text + txt2img) / 2.
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                test_img_acc_history.append(img2text)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history

