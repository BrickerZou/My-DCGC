import os
import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import my_model
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--sigma', type=float, default=0.01, help='Sigma of gaussian distribution')
parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--device', type=str, default='mps:0', help='device')

args = parser.parse_args()


# for args.dataset in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
for args.dataset in ["cora"]:
    print("Using {} dataset".format(args.dataset))

    if args.dataset == 'cora':
        args.cluster_num = 7
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'citeseer':
        args.cluster_num = 6
        args.gnnlayers = 2
        args.lr = 5e-5
        args.dims = [500]
    elif args.dataset == 'amap':
        args.cluster_num = 8
        args.gnnlayers = 5
        args.lr = 1e-5
        args.dims = [500]
    elif args.dataset == 'bat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'eat':
        args.cluster_num = 4
        args.gnnlayers = 5
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'uat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'corafull':
        args.cluster_num = 70
        args.gnnlayers = 2
        args.lr = 1e-3
        args.dims = [500]

    # load data
    X, y, A = load_graph_data(args.dataset, show_details=False)
    features = X
    true_labels = y
    '''将邻接矩阵转为稀疏矩阵，对角线元素为0'''
    adj = sp.csr_matrix(A)
    adj.setdiag(0)
    adj.eliminate_zeros()

    # 归一化邻接矩阵
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    print("adj_norm_s shape: ", adj_norm_s[0].shape)
    # 特征矩阵转为稠密矩阵
    sm_fea_s = sp.csr_matrix(features).toarray()
    print("sm_fea_s shape: ", sm_fea_s.shape)


    """保存归一化的特征矩阵, 减少重复计算"""
    path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    if os.path.exists(path):
        sm_fea_s = sp.csr_matrix(np.load(path, allow_pickle=True)).toarray()
    else:
        for a in adj_norm_s:
            sm_fea_s = a @ sm_fea_s
        np.save(path, sm_fea_s, allow_pickle=True)

    # 转为Tensor
    sm_fea_s = torch.FloatTensor(sm_fea_s)
    # 添加自环,转为Tensor
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()
    adj_tensor = torch.FloatTensor(adj_1st)

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    for seed in range(3):
        setup_seed(seed)
        best_acc, best_nmi, best_ari, best_f1, pred_labels = clustering(sm_fea_s, true_labels, args.cluster_num)
        model = my_model([features.shape[1]] + args.dims)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(args.device)
        fea = sm_fea_s.to(args.device)
        target_adj = adj_tensor.to(args.device)

        print('Start Training...')
        for epoch in tqdm(range(args.epochs)):
            model.train()
            z1, z2 = model(fea, is_train=True, sigma=args.sigma)
            S = z1 @ z2.T
            # 对比相似度矩阵和邻接矩阵的差异
            loss = F.mse_loss(S, target_adj)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0:
                model.eval()
                # 生成两个不同视图的嵌入
                z1, z2 = model(fea, is_train=False, sigma=args.sigma)
                # 合并两个视图的嵌入
                hidden_emb = (z1 + z2) / 2
                acc, nmi, ari, f1, pred_labels = clustering(hidden_emb, true_labels, args.cluster_num)
                tqdm.write('loss: {:.4f}, acc: {}, nmi: {}, ari: {}, f1: {}'.format(loss,acc, nmi, ari, f1))
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1

        tqdm.write('Best acc: {}, nmi: {}, ari: {}, f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
        acc_list.append(best_acc)
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)
        f1_list.append(best_f1)

    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    nmi_mean, nmi_std = np.mean(nmi_list), np.std(nmi_list)
    ari_mean, ari_std = np.mean(ari_list), np.std(ari_list)
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
    print("Accuracy: {:.2f} ± {:.2f}".format(acc_mean, acc_std))
    print("NMI: {:.2f} ± {:.2f}".format(nmi_mean, nmi_std))
    print("ARI: {:.2f} ± {:.2f}".format(ari_mean, ari_std))
    print("F1: {:.2f} ± {:.2f}".format(f1_mean, f1_std))
