from utils.Dataset import *
from utils.utils import *
from utils.Hanmming import *
from utils.AP_loss import *
import os
import json
from torch.utils.data.sampler import SequentialSampler
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def evaluate_no(args):
    test_set = HashingDataset(target=False, txt_path=args.txt_path, data_path=args.data_path,
                              img_filename='test_img.txt', label_filename='test_label.txt')
    sampler = SequentialSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False, sampler=sampler)

    if not os.path.exists(args.database_hash_path):
        database_set = HashingDataset(target=False, txt_path=args.txt_path, data_path=args.data_path,
                                      img_filename='database_img.txt', label_filename='database_label.txt')
        database_loader = DataLoader(database_set, batch_size=32, num_workers=4)
        generate_database_code(model=args.model, data_loader=database_loader, num_data=len(database_set), batch_size=32,
                               bit=args.bit, path=args.database_hash_path)
    database_hash = np.loadtxt(args.database_hash_path, dtype=float)
    database_label = load_label(f'{args.txt_path}/database_label.txt')

    target_labels = np.loadtxt(args.target_label_path)
    qN, queryL = [], []
    for it, data in tqdm(enumerate(test_loader)):
        query = data[0].cuda()
        label = data[1]
        target_label = target_labels[it]

        a = target_adv_no(query=query, epsilon=8 / 255)

        qN.append(torch.sign(args.model(a)).cpu().detach())
        queryL.append((torch.Tensor(target_label).reshape(1, len(target_label))).cpu().detach())

    qN = torch.cat(qN, 0)
    queryL = torch.cat(queryL, 0).numpy()

    tmap = CalcTopMap(qN, database_hash, queryL, database_label, args.topk)

    return tmap


def get_target_label(label, target=0):
    zero_index = np.where(label == target)
    # zero_index = np.array(zero_index).reshape(len(zero_index[0]))
    zero_index = np.array(zero_index)
    # print(zero_index.shape)
    random.seed(3)
    target_index = random.choice(zero_index[1])
    # print(target_index)
    # queryL = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # queryL[target_index] = 1
    # re_queryL =label-queryL
    return target_index


def evaluate(args):
    test_set = HashingDataset(target=False, txt_path=args.txt_path, data_path=args.data_path,
                              img_filename='test_img.txt', label_filename='test_label.txt')
    sampler = SequentialSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False, sampler=sampler)

    if not os.path.exists(args.database_hash_path):
        database_set = HashingDataset(target=False, txt_path=args.txt_path, data_path=args.data_path,
                                      img_filename='database_img.txt', label_filename='database_label.txt')
        database_loader = DataLoader(database_set, batch_size=32, num_workers=4)
        generate_database_code(model=args.model, data_loader=database_loader, num_data=len(database_set), batch_size=32,
                               bit=args.bit, path=args.database_hash_path)
    database_hash = np.loadtxt(args.database_hash_path, dtype=float)
    database_label = load_label(f'{args.txt_path}/database_label.txt')
    database_label_ = np.loadtxt(f'{args.txt_path}/database_label.txt', dtype=np.int64)

    if os.path.exists(args.target_label_path):
        target_labels = np.loadtxt(args.target_label_path)
    qB, qB_clean, clean_labelL, queryL = [], [], [], []
    for it, data in (enumerate(tqdm(test_loader))):
        query = data[0].cuda()
        label = data[1]

        if not os.path.exists(args.target_label_path):
            target_index = get_target_label(label, args.target)
            target_label = np.zeros([args.num_cls])
            target_label[target_index] = 1
        else:
            target_label = target_labels[it]
            target_index = np.where(target_label == 1)
        res_true_label = label.reshape(args.num_cls) - target_label
        res_true_label[target_index] = -(args.num_cls - 1)
        coe = len(np.where(res_true_label == 1)[0])

        pos_index = np.where(np.dot(target_label, database_label_.transpose()) == 1)[0]

        neg_index = np.where((np.dot(res_true_label, database_label_.transpose()) > 0))[0]

        # np.random.seed(3)
        pos_index_ = np.random.choice(pos_index, args.pos_size, replace=False)
        # np.random.seed(3)
        try:
            # neg_index_ = np.random.choice(neg_index,neg_size,replace=False)
            neg_index_ = np.random.choice(neg_index, int(args.pos_size * coe * 4), replace=False)
        except:
            neg_index = np.where(np.dot(target_label, database_label_.transpose()) == 0)[0]
            # neg_index_ = np.random.choice(neg_index,neg_size,replace=False)
            neg_index_ = np.random.choice(neg_index, int(args.pos_size * coe * 4), replace=False)
            # qB = torch.cat((qB,torch.sign(model(query))))
            # continue;
        pos_train_hash = torch.Tensor(database_hash[pos_index_]).cuda()
        neg_train_hash = torch.Tensor(database_hash[neg_index_]).cuda()

        a, b = target_adv(query=query, model=args.model, pos_hash=pos_train_hash, neg_hash=neg_train_hash,
                          pos_num=pos_train_hash.shape[0], epsilon=args.epsilon, iteration=args.iteration)
        # if it ==0 :
        #     q = torch.sign(args.model(a))
        # else:
        #     q = torch.cat((qB,torch.sign(args.model(a))))

        # a = target_adv3(query=query,model=model,pos_hash=pos_train_hash,neg_hash=neg_train_hash,
        # pos_num=pos_train_hash.shape[0],epsilon=8/255, step=10, iteration=10,alpha=alpha)

        #     if it ==0 :
        #         qB = torch.sign(model(a))
        #     else :
        #         qB = torch.cat((qB,torch.sign(model(a))))
        #     if it ==150 : break
        #     # tmap = CalcMap(qB, database_hash, query_target_label, database_label)
        # queryL = queryL.cpu().detach().numpy()
        # tmap = CalcTopMap(qB, database_hash, queryL, database_label,50)
        # return tmap
        clean_labelL.append(label.cpu().detach())
        qB.append(torch.sign(args.model(a)).cpu().detach())
        qB_clean.append(torch.sign(args.model(query)).cpu().detach())
        queryL.append((torch.Tensor(target_label).reshape(1, len(target_label))).cpu().detach())

    qB = torch.cat(qB, 0)
    qB_clean = torch.cat(qB_clean, 0)
    queryL = torch.cat(queryL, 0).numpy()
    clean_labelL = torch.cat(clean_labelL, 0)
    # if not os.path.exists(args.save_pr):
    #     mAP, cum_prec, cum_recall = CalcTopMapWithPR(qB, queryL, database_hash,database_label,-1)
    #     index_range = 23300 // 100
    #     index = [i * 100 - 1 for i in range(1, index_range + 1)]
    #     max_index = max(index)
    #     overflow = 23300 - index_range * 100
    #     index = index + [max_index + i for i in range(1, overflow + 1)]
    #     c_prec = cum_prec[index]
    #     c_recall = cum_recall[index]

    #     pr_data = {
    #         "index": index,
    #         "P": c_prec.tolist(),
    #         "R": c_recall.tolist()
    #     }
    #     with open(args.save_pr, 'w') as g:
    #         g.write(json.dumps(pr_data))
    #     print(f"pr curve saved at {args.save_pr}")

    # tmap = CalcTopMap(qB, database_hash, queryL, database_label,50)
    if not os.path.exists(args.target_label_path):
        np.savetxt(args.target_label_path, queryL, fmt='%d')

        # np.savetxt(args.target_label_path,queryL.cpu().detach().numpy(),fmt='%d')
    # np.savetxt('./logs/adv_qB.txt',qB.cpu().detach().numpy(),fmt='%d')
    # np.savetxt('./logs/clean_qB.txt',qB_clean.cpu().detach().numpy(),fmt='%d')
    tmap = CalcTopMap(qB, database_hash, queryL, database_label, args.topk)
    tmap_clean = CalcTopMap(qB_clean, database_hash, queryL, database_label, args.topk)
    tmap_clean_ori = CalcTopMap(qB, database_hash, clean_labelL, database_label, args.topk)
    tmap_clean_ori_multi = CalcTopMap(qB_clean, database_hash, clean_labelL, database_label, args.topk)
    return tmap, tmap_clean_ori, tmap_clean, tmap_clean_ori_multi


def evaluate_multi(args):
    test_set = HashingDataset(target=False, txt_path=args.txt_path, data_path=args.data_path,
                              img_filename='test_img.txt', label_filename='test_label.txt')
    sampler = SequentialSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False, sampler=sampler)
    # test_hash,test_label = generate_hash_code(model=model,data_loader=test_loader,num_data=num_data,batch_size=batch,bit=bit,cls=cls_num)
    # torch.Size([2100, 32]) (2100, 21)

    # torch.Size([10500, 32]) (10500, 21)
    if not os.path.exists(args.database_hash_path):
        database_set = HashingDataset(target=False, txt_path=args.txt_path, data_path=args.data_path,
                                      img_filename='database_img.txt', label_filename='database_label.txt')
        database_loader = DataLoader(database_set, batch_size=32, num_workers=4)
        generate_database_code(model=args.model, data_loader=database_loader, num_data=len(database_set), batch_size=32,
                               bit=args.bit, path=args.database_hash_path)
    database_hash = np.loadtxt(args.database_hash_path, dtype=float)
    database_label = load_label(f'{args.txt_path}/database_label.txt')
    database_label_ = np.loadtxt(f'{args.txt_path}/database_label.txt', dtype=np.int64)

    tmap_mean = 0
    tmap_mean_clean = 0
    # f = open('./log.txt', 'w')
    if os.path.exists(args.target_label_path):
        target_labels = np.loadtxt(args.target_label_path)
    labels_candidate = load_label(args.txt_path + '/database_label.txt')
    candidate_labels = labels_candidate.unique(dim=0)

    for it, data in (enumerate(test_loader)):
        # plot_jug = False
        # plot_index = [449]
        # if it in plot_index : plot_jug = True
        query = data[0].cuda()
        label = data[1]
        if it == 0:
            clean_labelL = label
        else:
            clean_labelL = torch.cat((clean_labelL, label))

        if not os.path.exists(args.target_label_path):
            candi = []
            for iii in range(candidate_labels.shape[0]):
                if np.dot(candidate_labels[iii], label.numpy().reshape(args.num_cls)) == 0 and torch.sum(
                        candidate_labels[iii]) != 0:
                    candi.append(iii)
            # target_label_index = np.random.choice(range(candidate_labels.size(0)), size=1)
            target_label_index = np.random.choice(range(len(candi)), size=1)

            # target_label = candidate_labels[target_label_index]
            target_label = candidate_labels.index_select(0,
                                                         torch.from_numpy(np.array(candi)[target_label_index])).squeeze(
                0)
            target_index = np.where(target_label == 1)[0]

            if it == 0:
                L = torch.Tensor(target_label).cuda().reshape(1, len(target_label))
            else:
                L = torch.cat((L, torch.Tensor(target_label).cuda().reshape(1, len(target_label))))
        else:
            target_label = target_labels[it]
            target_index = np.where(target_label == 1)[0]
        res_true_label = label.reshape(args.num_cls) - target_label
        res_true_label[target_index] = -(args.num_cls - 1)
        pos_len = len(np.where(target_label == 1)[0])
        coe = len(np.where(res_true_label == 1)[0])
        target_label_ = (-args.num_cls - 1) * (torch.ones([args.num_cls]) - target_label) + 1

        if it % 50 == 0:
            queryL = torch.Tensor(target_label).cuda().reshape(1, len(target_label))
        else:
            queryL = torch.cat((queryL, torch.Tensor(target_label).cuda().reshape(1, len(target_label))))
            # print(queryL.shape)
        # neg_label_index = list(set(label)-{target_index})
        pos_index = np.where(np.dot(target_label_, database_label_.transpose()) > 0)[0]

        neg_index = np.where((np.dot(res_true_label, database_label_.transpose()) > 0))[0]

        # np.random.seed(3)
        pos_index_ = np.random.choice(pos_index, args.pos_size, replace=args.replace)
        # np.random.seed(3)
        try:
            # neg_index_ = np.random.choice(neg_index,neg_size,replace=False)
            neg_index_ = np.random.choice(neg_index, int(args.pos_size * coe * 6), replace=args.replace)
        except:
            neg_index = np.where(np.dot(target_label, database_label_.transpose()) == 0)[0]
            # neg_index_ = np.random.choice(neg_index,neg_size,replace=False)
            neg_index_ = np.random.choice(neg_index, int(args.pos_size * coe * 6), replace=args.replace)
            # qB = torch.cat((qB,torch.sign(model(query))))
            # continue;
        pos_train_hash = torch.Tensor(database_hash[pos_index_]).cuda()
        neg_train_hash = torch.Tensor(database_hash[neg_index_]).cuda()

        a, b = target_adv(query=query, model=args.model, pos_hash=pos_train_hash, neg_hash=neg_train_hash,
                          pos_num=pos_train_hash.shape[0], epsilon=args.epsilon, iteration=args.iteration)
        if it == 0:
            q = torch.sign(args.model(a))

        else:
            q = torch.cat((qB, torch.sign(args.model(a))))

        # a = target_adv3(query=query,model=model,pos_hash=pos_train_hash,neg_hash=neg_train_hash,
        # pos_num=pos_train_hash.shape[0],epsilon=8/255, step=10, iteration=10,alpha=alpha)

        #     if it ==0 :
        #         qB = torch.sign(model(a))
        #     else :
        #         qB = torch.cat((qB,torch.sign(model(a))))
        #     if it ==150 : break
        #     # tmap = CalcMap(qB, database_hash, query_target_label, database_label)
        # queryL = queryL.cpu().detach().numpy()
        # tmap = CalcTopMap(qB, database_hash, queryL, database_label,50)
        # return tmap

        if it % 50 == 0:
            qB = torch.sign(args.model(a))
            qB_clean = torch.sign(args.model(data[0].cuda()))
        else:
            qB = torch.cat((qB, torch.sign(args.model(a))))
            qB_clean = torch.cat((qB_clean, torch.sign(args.model(data[0].cuda()))))
        if it % 50 == 49:
            # tmap = CalcMap(qB, database_hash, query_target_label, database_label)
            queryL = queryL.cpu().detach().numpy()
            # tmap = CalcTopMap(qB, database_hash, queryL, database_label,50)
            tmap = CalcTopMap(qB, database_hash, queryL, database_label, args.topk)
            # print(tmap, file=f)
            tmap_clean = CalcTopMap(qB_clean, database_hash, queryL, database_label, args.topk)
            tmap_mean += tmap
            tmap_mean_clean += tmap_clean

    # if not os.path.exists(args.save_pr):
    #     mAP, cum_prec, cum_recall = CalcTopMapWithPR(q, queryL, database_hash,database_label,-1)
    #     index_range = 23300 // 100
    #     index = [i * 100 - 1 for i in range(1, index_range + 1)]
    #     max_index = max(index)
    #     overflow = 23300 - index_range * 100
    #     index = index + [max_index + i for i in range(1, overflow + 1)]
    #     c_prec = cum_prec[index]
    #     c_recall = cum_recall[index]

    #     pr_data = {
    #         "index": index,
    #         "P": c_prec.tolist(),
    #         "R": c_recall.tolist()
    #     }
    #     with open(args.save_pr, 'w') as g:
    #         g.write(json.dumps(pr_data))
    #     print(f"pr curve saved at {args.save_pr}")
    if not os.path.exists(args.target_label_path):
        np.savetxt(args.target_label_path, L.cpu().detach().numpy(), fmt='%d')

    tmap_clean_ori = CalcMap(qB, database_hash, clean_labelL, database_label)
    tmap_clean_ori_multi = CalcMap(qB_clean, database_hash, clean_labelL, database_label)
    return tmap_mean / (len(test_set) / 50), tmap_clean_ori, tmap_mean_clean / (
                len(test_set) / 50), tmap_clean_ori_multi