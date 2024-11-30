import torch
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans

from kmeans_gpu import kmeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score



def setup_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cluster_acc(y_true, y_pred):
    # 计算聚类准确率和 F1 分数
    y_true = y_true - np.min(y_true)
    l1, l2 = list(set(y_true)), list(set(y_pred))
    num_class1, num_class2 = len(l1), len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i not in l2:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    if num_class1 != len(l2):
        print('error')
        return
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            cost[i][j] = len([i1 for i1 in mps if y_pred[i1] == c2])
    m = Munkres()
    indexes = m.compute(cost.__neg__().tolist())
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        new_predict[[ind for ind, elm in enumerate(y_pred) if elm == c2]] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro



def load_graph_data(dataset_name, show_details=False):
    # 加载图数据
    load_path = f"dataset/{dataset_name}/{dataset_name}"
    feat = np.load(f"{load_path}_feat.npy", allow_pickle=True)
    label = np.load(f"{load_path}_label.npy", allow_pickle=True)
    adj = np.load(f"{load_path}_adj.npy", allow_pickle=True)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print(f"dataset name:   {dataset_name}")
        print(f"feature shape:  {feat.shape}")
        print(f"label shape:    {label.shape}")
        print(f"adj shape:      {adj.shape}")
        print(f"undirected edge num:   {int(np.nonzero(adj)[0].shape[0]/2)}")
        print(f"category num:          {max(label)-min(label)+1}")
        print("category distribution: ")
        for i in range(max(label)+1):
            print(f"label {i}: {len(label[np.where(label == i)])}")
        print("++++++++++++++++++++++++++++++")
    return feat, label, adj



def preprocess_graph(adj, layer, norm='sym', renorm=True):
    # 预处理图
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    adj_ = adj + ident if renorm else adj
    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    return [ident - (1 * laplacian) for _ in range(layer)]

def eva(y_true, y_pred, show_details=True):
    # 评估聚类性能
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(f':acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}')
    return acc, nmi, ari, f1

def clustering(feature, true_labels, cluster_num):
    # 聚类
    predict_labels, _ = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="mps")

    # kmeans = KMeans(n_clusters=cluster_num)
    # X = feature.cpu().detach().numpy().astype(np.float64)
    # kmeans.fit(X)
    # predict_labels = kmeans.predict(X)
    # acc, nmi, ari, f1 = eva(true_labels, predict_labels, show_details=False)

    acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)

    # 通过对比，自实现的k-measn结果更优，但稳定性较差Accuracy: 73.91 ± 2.29。sklearn的k-means结果Accuracy: 72.81 ± 0.72
    return round(100 * acc, 2), round(100 * nmi, 2), round(100 * ari, 2), round(100 * f1, 2), predict_labels.numpy()


# def load_data(dataset):
#     # 加载数据
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#
#     for name in names:
#         with open(f"data/ind.{dataset}.{name}", 'rb') as rf:
#             u = pkl._Unpickler(rf)
#             u.encoding = 'latin1'
#             objects.append(u.load())
#
#     x, y, tx, ty, allx, ally, graph = objects
#     test_idx_reorder = parse_index_file(f"data/ind.{dataset}.test.index")
#     test_idx_range = np.sort(test_idx_reorder)
#
#     if dataset == 'citeseer':
#         # 修复 citeseer 数据集中的孤立节点
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range - min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range - min(test_idx_range), :] = ty
#         ty = ty_extended
#
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     features = torch.FloatTensor(features.todense())
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#
#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]
#
#     return adj, features, np.argmax(labels, 1)
#
# def parse_index_file(filename):
#     # 解析索引文件
#     return [int(line.strip()) for line in open(filename)]



# def laplacian(adj):
#     # 计算拉普拉斯矩阵
#     rowsum = np.array(adj.sum(1))
#     degree_mat = sp.diags(rowsum.flatten())
#     lap = degree_mat - adj
#     return torch.FloatTensor(lap.toarray())

