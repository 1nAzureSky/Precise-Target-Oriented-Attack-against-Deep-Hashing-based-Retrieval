import os
import torch
from PIL import Image
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def CalcHammingDist(B1, B2):
    B1 = B1.cpu().detach().numpy()
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    retrievalL = retrievalL.numpy()

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

def CalcTopMap(qB, rB, queryL, retrievalL, topk):
    #qB queryBinary, query数据集，都转成了哈希码
    #rB retrievalBinary，gallery数据集，都转成了哈希码
    #queryL queryLabel，query数据集的标签
    #retrievalL retrievalLabel，gallery数据集的标签
    retrievalL = retrievalL.numpy()
    num_query = qB.shape[0]
    print(num_query)
            #共有多少个查询
    topkmap = 0
    # print('------------Calculating top-k MAP------------')

    for iter in tqdm(range(num_query)): 
        if num_query==1 :
            gnd = (np.dot(queryL, retrievalL.transpose()) > 0).astype(np.float32)
        else:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        if num_query == 1:
            hamm = CalcHammingDist(qB, rB)
        else:
            hamm = CalcHammingDist(qB[iter, :], rB)  
        ind = np.argsort(hamm)                   
        gnd = gnd[ind]                          
 
        tgnd = gnd[0:topk]                       
        tsum = np.sum(tgnd).astype(int)          
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
 
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        if num_query == 1:
            topkmap_ = np.mean(count / (tindex[1]))
        else:
            topkmap_ = np.mean(count / (tindex[0]))
        topkmap += topkmap_
    topkmap = topkmap / num_query
    return topkmap

def CalcMap(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    retrievalL = retrievalL.numpy()
    # print(qB.shape,rB.shape,queryL.shape,retrievalL.shape)
    num_query = qB.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # print('------------Calculating MAP------------')
    for iter in tqdm(range(num_query)):
        if num_query==1 :
            gnd = (np.dot(queryL, retrievalL.transpose()) > 0).astype(np.float32)
        else:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        if num_query == 1:
            hamm = CalcHammingDist(qB, rB)
        else:
            hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, int(tsum))

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        if num_query == 1:
            map_ = np.mean(count[:10] / (tindex[1]))
        else:
            map_ = np.mean(count[:10] / (tindex[0][:10]))

        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map
def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())

def compute_dis(x):
    """computes the affinity matrix between an input vector and itself"""
    ou = nn.PairwiseDistance(p=2)
    return ou(x.t(),x)

def compute_hamming(x):
    """computes the affinity matrix between an input vector and itself"""
    k = torch.ones((x.shape[0],x.shape[0])).cuda()
    k *= x.shape[1]
    # return (k-torch.mm(x, x.t()))/2
    return ((k-torch.mm(x, x.t()))/2)/5

def hmm(x,y,k):
    x = x.unsqueeze(0)
    return ((k-torch.mm(x, y.permute(1,0)))/2)