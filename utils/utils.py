import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from utils.AP_loss import *

def generate_database_code(model, data_loader, num_data, batch_size,bit,save_path):
    print('------------Calculating database code------------')
    B = torch.zeros(num_data, bit)
    for it, data in tqdm(enumerate(data_loader)):
        data_input = data[0]
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        batch_size_ = output.size(0)
        u_ind = np.linspace(it * batch_size, np.min((num_data, (it + 1) * batch_size)) - 1, batch_size_, dtype=int)
        B[u_ind, :] = torch.sign(output.cpu().data)
    np.savetxt(save_path,B)  




def target_adv_no(query,epsilon=8/255):
    noise = torch.rand_like(query).cuda()
    noise.data = noise.data.clamp(-epsilon, epsilon)
    noise.data = (query.data + noise.data).clamp(0, 1) - noise.data

    return query + noise.detach()

def target_adv(query,model,pos_hash,neg_hash,pos_num=50,epsilon=8/255, iteration=20):#inmatrix:(1+n-1,3,224)
    # print(torch.sign(model((query).unsqueeze(0),alpha)))
    delta = torch.zeros_like(query).cuda()
    noise = torch.rand_like(query).cuda()
    noise.data = noise.data.clamp(-epsilon, epsilon)
    noise.data = (query.data + noise.data).clamp(0, 1) - noise.data
    # plot(delta,'delta_0')
    database_hash = torch.cat((pos_hash,neg_hash),0)
    s = epsilon/iteration
    decay_factor=1.0
    g=0
    delta.requires_grad = True
    lossObj = SmoothAP(0.1).requires_grad_()
    for i in (range(iteration)):
        # print(i)
        
    # alpha = get_alpha(i)
        # random_start = random.randint(0,database.shape[0]-33)
        # batabase_hash = model(database[random_start:random_start+32])

        # noisy_query_hash = model((query+delta),alpha)
        noisy_query_hash = model((query+delta))

        
        loss = lossObj(noisy_query_hash,database_hash,pos_num)
        loss.backward()

        
        # if(i==0):
        #     gradma = delta.grad.detach()
        # else:
        #     sim=0
        #     for inde in range(3):
        #         sim += torch.sum(torch.mul(gradma[0][inde],delta.grad.detach()[0][inde]))
        #     print(sim)

        # g = decay_factor*g + data_grad/torch.norm(data_grad,p=1)

        delta.data = delta - s * delta.grad.detach().sign()
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    # if plot_jug:
    #     plot((query + delta).reshape([3,224,224]),'plot')
    delta.grad.zero_()
    return query + delta.detach(),query + noise.detach()
from torch.autograd import Variable as V

def target_adv4(query,model,pos_hash,neg_hash,pos_num=50,epsilon=8/255, step=100, iteration=20,alpha=5):#inmatrix:(1+n-1,3,224)

    eps = epsilon
    num_iter = iteration
    alpha = eps / num_iter
    momentum = 1
    grad = 0
    number = 10
    beta= 3
    vt_alpha = 1
    up_time = 5
    noise = 0

    x = query.clone().detach().cuda()
    # delta = torch.zeros_like(query).cuda()
    database_hash = torch.cat((pos_hash,neg_hash),0)

    for i in range(num_iter):
        print(i)
        x = V(x, requires_grad = True)

        noisy_query_hash = model((x),0.2)
        lossObj = SmoothAP(0.1).requires_grad_()
        loss = lossObj(noisy_query_hash,database_hash,pos_num)
        loss.backward()
        new_grad = x.grad.data
        global_grad = 0

        for ii in range(number):
            neighbor = (((torch.rand_like(x) - 0.5) * 2) * (beta * eps))
            if i<up_time:
                x_neighbor = x + neighbor
            else:
                new_point = x - new_grad.sign() * eps * vt_alpha
                x_neighbor = new_point + neighbor
            x_neighbor = V(x_neighbor, requires_grad = True)
            noisy_query_hash = model((x_neighbor),0.2)
            loss = lossObj(noisy_query_hash,database_hash,pos_num)
            loss.backward()
            global_grad += x_neighbor.grad.data
        
        # current_grad = variance if i > 0 else new_grad
        current_grad = global_grad / number
        noise = momentum * grad + current_grad / torch.abs(current_grad).mean([1,2,3], keepdim=True)         
        grad = noise
        # variance = global_grad / number

        x = x - alpha * torch.sign(noise)
        x = x.clamp(0, 1)

    return x.detach(),x.detach()

def mifgsm_attack(input,epsilon,data_grad):
  iter=10
  decay_factor=1.0
  pert_out = input
  alpha = epsilon/iter
  g=0
  for i in range(iter-1):
    print(i)
    g = decay_factor*g + data_grad/torch.norm(data_grad,p=1)
    pert_out = pert_out - alpha*torch.sign(g)
    pert_out = torch.clamp(pert_out, 0, 1)
    # if torch.norm((pert_out-input),p=float('inf')) > epsilon:
    #   break
  return pert_out

def ifgsm_attack(input,epsilon,data_grad):
  iter = 10
  alpha = epsilon/iter
  pert_out = input
  for i in range(iter-1):
    print(i)
    pert_out = pert_out - alpha*data_grad.sign()
    pert_out = torch.clamp(pert_out, 0, 1)
    # if torch.norm((pert_out-input),p=float('inf')) > epsilon:
    #   break
  return pert_out


def load_model(path):
    model = torch.load(path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model

# def load_model(path):
#     model = ResNet(64,'ResNet50')
#     model.load_state_dict(torch.load(path,map_location='cuda:2'))
#     model.eval()
#     if torch.cuda.is_available():
#         model = model.cuda()
#     model.eval()
#     return model

def compute(x,y):
    assert x.shape==y.shape
    k = x.shape[0]
    return k-x.t().mm(y)

def get_prediction(index, model, alpha=1):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    img_path = '/dataset/NUS-WIDE'
    for ind in index:
        img = Image.open(os.path.join(img_path, X[ind]))
        img = img.convert('RGB')
        img = transform(img)
        img = Variable(img.cuda())
        img = torch.unsqueeze(img, dim=0)
        if index.index(ind) == 0:
            imgs = img
        else:
            imgs = torch.cat((imgs, img), 0)
    predict = model(imgs, alpha)
    return predict


def plot(tensor, img_name):
    tensor = tensor.clamp(0, 1)
    channel_last_tensor = tensor.permute(1, 2, 0).cpu()
    np_img = channel_last_tensor.detach().numpy()
    plt.imsave(f'./save_fig/{img_name}.png', np_img)


# def get_target_label(ind):
#     label = np.array(Y[ind])
#     zero_index = np.where(label == 0)
#     zero_index = np.array(zero_index).reshape(len(zero_index[0]))
#     target_index = random.choice(zero_index)
#     return target_index

def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall
    
def generate_multilabel_sample(database_img,database_label,index,file='/data/NUS-WIDE/'):
    target_label= np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    target_label[index] = 1

    sample_index = np.where(np.dot(target_label,database_label.transpose())==1)[0]
    test_label = np.ones([21])
    ret = []

    for i in sample_index:
        if np.dot(database_label[i],test_label)!=1:
            ret.append(i)
    sample_100 = np.random.choice(ret,100,replace=False)

    with open(file+'test_label.txt','a') as f:
        for ii in sample_100:
            f.write(' '.join(str(i) for i in database_label[ii]))
            f.write('\n')
    
    with open(file+'test_img.txt','a') as f:
        for ii in sample_100:
            f.write(database_img[ii])
            f.write('\n')
    f.close()

def get_img(index):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    img_path = '/dataset/NUS-WIDE'
    for ind in index:
        img = Image.open(os.path.join(img_path, X[ind]))
        img = img.convert('RGB')
        img = transform(img)
        img = Variable(img.cuda())
        img = torch.unsqueeze(img, dim=0)
        if index.index(ind) == 0:
            imgs = img
        else:
            imgs = torch.cat((imgs, img), 0)
    return imgs


def generate_hash_code(model, data_loader, num_data, batch_size,bit,cls):
    print('------------Calculating binary code------------')
    B = torch.zeros(num_data, bit)
    labels = np.zeros([num_data, cls])
    for it, data in tqdm(enumerate(data_loader)):
        data_input = data[0]
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        batch_size_ = output.size(0)
        u_ind = np.linspace(it * batch_size, np.min((num_data, (it + 1) * batch_size)) - 1, batch_size_, dtype=int)
        B[u_ind, :] = torch.sign(output.cpu().data)
        labels[u_ind, :] = data[1].cpu().detach().numpy()
    return torch.Tensor(B).cuda(),labels

def generate_hash_feature(model, alpha,data_loader, num_data, batch_size,bit,cls):
    print('------------Calculating feature------------')
    B = torch.zeros(num_data, bit)
    labels = np.zeros([num_data, cls])
    for it, data in tqdm(enumerate(data_loader)):
        data_input = data[0]
        data_input = Variable(data_input.cuda())
        output = model(data_input,alpha=alpha)
        batch_size_ = output.size(0)
        u_ind = np.linspace(it * batch_size, np.min((num_data, (it + 1) * batch_size)) - 1, batch_size_, dtype=int)
        B[u_ind, :] = output.cpu().data
        labels[u_ind, :] = data[1].cpu().detach().numpy()
    return torch.Tensor(B).cuda(),labels

def get_iter_data(begin,index,size):
    data = [begin]
    for it, i in enumerate(index):
        if it==0:
            i_ind = np.random.randint((i-1)*500,i*500,size=400)
        else:
            i_ind = np.random.randint((i-1)*500,i*500,size=30)
        for j in i_ind : data.append(j)
    return data

def generate_database_code(model, data_loader, num_data, batch_size,bit,path):
    print('------------Calculating database code------------')
    B = torch.zeros(num_data, bit)
    for it, data in (enumerate(tqdm(data_loader))):
        data_input = data[0]
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        batch_size_ = output.size(0)
        u_ind = np.linspace(it * batch_size, np.min((num_data, (it + 1) * batch_size)) - 1, batch_size_, dtype=int)
        B[u_ind, :] = torch.sign(output.cpu().data)
    np.savetxt(path,B)        
