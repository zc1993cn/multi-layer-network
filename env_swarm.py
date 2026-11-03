import networkx as nx
import numpy as np
import random
import torch

def sigmoid(x):
    return torch.from_numpy(np.array(1./(1 + np.exp(2 * (-x)))))

def nx_graph(adj):
    G = nx.Graph()
    tmp = adj

    for i in range(0, len(tmp)):
        G.add_node(i)

    for i in range(0, len(tmp)):
        idx = np.where(tmp[i, :] == 1)
        sq_idx = idx[0]
        for j in range(0, len(sq_idx)):
            G.add_edge(i, sq_idx[j])

    tag = nx.is_connected(G)
    if tag:
        d = nx.diameter(G)
    else:
        d = 2 * len(tmp)
    # ps = nx.shell_layout(G)
    # nx.draw(G, ps, with_labels=False, node_size=6.png)
    return d


def get_uav_input(dis, r):

    dis_i = dis.copy()
    dis = dis + np.eye(dis.shape[0]) * 100
    dis_array = dis - r

    for i in range(dis.shape[0]):
        idx_neigh_out = np.where(dis_array[i, :] > 0)[0]
        if idx_neigh_out.shape[0]:
            dis_i[i, idx_neigh_out.astype(int)] = 0

    return torch.from_numpy(dis_i)


def step(a, r, dis):
    dis = dis + np.eye(dis.shape[0]) * 100
    dis_array = dis - r
    adj = np.zeros([dis.shape[0], dis.shape[0]])

    for i in range(dis.shape[0]):
        idx_neigh = np.where(dis_array[i, :] < 0)[0]
        freq = np.zeros(dis.shape[0])
        prob = np.zeros(dis.shape[0])
        uav_linked = np.zeros(dis.shape[0])

        if a[i] >= idx_neigh.shape[0]:
            adj[i, idx_neigh] = 1
            adj[idx_neigh, i] = 1

        if a[i] < idx_neigh.shape[0]:

            for j in range(idx_neigh.shape[0]):
                idx_neigh_neigh = np.where(dis_array[idx_neigh[j], :] < 0)[0]

                inter = set(idx_neigh).intersection(set(idx_neigh_neigh))
                array_inter = np.array(list(inter))

                if array_inter.shape[0] > 0:
                    freq[array_inter] = freq[array_inter] + 1

            prob[idx_neigh] = 1 - freq[idx_neigh] / idx_neigh.shape[0]

            tmp_count = np.zeros(dis.shape[0])
            flag = 1

            while flag:
                r_seed = np.random.uniform(0, 1, dis.shape[0])
                samp_idx = prob - r_seed
                tmp = np.where(samp_idx > 0)[0]

                tmp_count[tmp] = tmp_count[tmp] + 1
                idx_sample = np.where(tmp_count > 0)[0]

                if idx_sample.shape[0] >= a[i]:
                    flag = 0

            if idx_sample.shape[0] == a[i]:
                uav_linked[idx_sample] = uav_linked[idx_sample] + 1

            if idx_sample.shape[0] > a[i]:
                tmp = tmp_count[idx_sample]
                s_array = np.sort(tmp)
                diff = idx_sample.shape[0] - a[i]
                gip = s_array[diff - 1]

                idx0 = np.where(tmp < gip)[0]
                idx1_tmp = np.where(tmp == gip)[0]
                idx1_n = diff - idx0.shape[0]
                idx1 = random.sample(list(idx_sample[idx1_tmp]), idx1_n)

                diff_set0 = set(idx_sample).difference(set(idx_sample[idx0]))
                diff_set1 = set(diff_set0).difference(set(idx1))
                array_diff = np.array(list(diff_set1), dtype=np.int)
                uav_linked[array_diff] = uav_linked[array_diff] + 1

            adj[i, :] = uav_linked
            adj[:, i] = uav_linked

    return adj


def reward(adj):

    d = nx_graph(adj)
    node_n = adj.shape[0]
    p1 = 0

    if d > node_n:
        p1 = -3

    #　s_v = np.sum(v_adj)
    s_v = node_n * node_n

    # 网络链路数量
    count = np.sum(adj)
    # 网络均匀分布指标
    degree = np.sum(adj, axis=1)
    diff1 = np.max(degree) - np.min(degree)
    diff = np.std(degree)

    # loss = count / s_v - diff / (node_n - 2)  + p1
    # loss = 1 / diff + p1
    loss = 1 / (1 + diff) + p1 + count / s_v + (1 - diff1 / (node_n - 2))
    

    #　loss = 1 - (count / s_v) * (count / s_v) + p1 + p2

    return loss, diff, count








