import os
import torch
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from model import TSSAGCN
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label
from scipy.io import loadmat
from scipy.io import savemat

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = 'mirflickr'  
    embedding = 'glove' 
    # data parameters
    DATA_DIR = 'data/' + dataset + '/'
    EVAL = False   
    adj_file='data/' + dataset + '/adj.mat'
    max_epoch =100
    batch_size = 100
    lr = 1e-5
    betas = (0.5, 0.999)
    alph = 0.2
    tau=0.4
    n_layers = 5   
    kappa=0.21
    lamuda1=0.5
    lamuda2=0.5
    lamuda3=0.1
    upsilon1=0.4
    upsilon2=0.3
    upsilon3=0.3

    adj_image = loadmat('data/' + dataset + 'adj_image.mat')['adj_image']
    adj_image = torch.FloatTensor(adj_image)
    adj_content = loadmat('data/' + dataset + 'adj_content.mat')['adj_content']
    adj_content = torch.FloatTensor(adj_content)

    seed = 103
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if embedding == 'glove':
        inp = loadmat('embedding/' + dataset + '-inp-glove6B.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'googlenews':
        inp = loadmat('embedding/' + dataset + '-inp-googlenews.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'fasttext':
        inp = loadmat('embedding/' + dataset + '-inp-fasttext.mat')['inp']
        inp = torch.FloatTensor(inp)
    else:
        inp = None

    print('...Data loading is beginning...')

    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)

    print('...Data loading is completed...')

    model_ft = TSSAGCN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],num_classes=input_data_par['num_class'], 
                       tau=tau,upsilon1=upsilon1,upsilon2=upsilon2,upsilon3=upsilon3,alph=alph,adj_file=adj_file, inp=inp,adj_content=adj_content,adj_image=adj_image,
                        n_layers=n_layers,image_all_dim=input_data_par['image_all_dim']).cuda()
        
    params_to_update = list(model_ft.parameters())
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    if EVAL:
        model_ft.load_state_dict(torch.load('model/TSSAGCN' + dataset + '.pth')) 
    else:
        print('...Training is beginning...')
        train_phase="image" #select "image" or "text"
        model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader,input_data_par, optimizer,train_phase, lamuda1,lamuda2,lamuda3,kappa, max_epoch,adj_file)

        print('...Training is completed...')
        torch.save(model_ft.state_dict(), 'model/TSSAGCN' + dataset + '.pth')
    print('...Evaluation on testing data...')
    model_ft.eval()          
    view1_feature, view2_feature = model_ft(
        torch.tensor(input_data_par['img_test']).cuda(), torch.tensor(input_data_par['text_test']).cuda(),batch=input_data_par['test_num'],img_train=torch.FloatTensor(input_data_par['img_train']),evals="1")
    label = input_data_par['label_test']
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()

    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))







              