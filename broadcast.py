import torch
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

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
        d = 0

    return d

def get_adj(dis, r, g):
    adj_i = np.zeros([dis.shape[0], dis.shape[0]])

    dis = dis + np.eye(dis.shape[0]) * 500
    dis_array = dis - r

    for i in range(dis.shape[0]):
        idx_neigh_in = np.where(dis_array[i, :] <= 0)[0]

        if idx_neigh_in.shape[0] > 0:
            for j in range(idx_neigh_in.shape[0]):
                idx = idx_neigh_in[j]
                if np.abs(g[i] - g[idx]) < 2:
                    adj_i[i, idx] = 1

    return adj_i


def eval_adj(adj):
    loss = np.zeros(2)
    d = nx_graph(adj)

    # 网络跳数
    if d > 0:
        loss[0] = d

    # 网络链路数量
    count = np.sum(adj)

    loss[1] = count

    return loss


def eval_network(adj):
    G = nx.Graph()
    tmp = adj
    
    for i in range(0, len(tmp)):
        G.add_node(i)
    
    for i in range(0, len(tmp)):
        idx = np.where(tmp[i, :] == 1)
        sq_idx = idx[0]
        for j in range(0, len(sq_idx)):
            G.add_edge(i, sq_idx[j])
            
    conn = 0
    s1 = set(np.arange(0, 11, 1))
    s2 = set(np.arange(11, 22, 1))
    s3 = set(np.arange(22, 33, 1))
    
    for g in nx.connected_components(G):
        if len(g) > 0:
            f1 = g.intersection(s1)
            f2 = g.intersection(s2)
            f3 = g.intersection(s3)
            
            if len(f1) > 0:
                if len(f2) > 0:
                    if len(f3) > 0:
                        conn = 1
    
    return conn

def read_data(file_name, n):
    trace_data = torch.zeros([5, 200, n, 3])
    str1 = './trace_data/'
    str2 = str1 + file_name

    for it in range(5):
        str3 = str(it)
        file_path = str2 + str3
        trace_data[it, :, :, :] = torch.load(file_path)

    return trace_data


if __name__ == '__main__':
    node_num = 33
    #pos_uav_att = torch.load('uav_positions/uav_test_att_111')
    #pos_uav_att = torch.load('uav_positions/uav_att_111_1')
    pos_uav = torch.load('uav_positions/uav_111')
    pos_uav_att = torch.load('uav_positions/uav_att_33_1')
    #pos_uav = read_data('uav_train_att_33', node_num)
    R = 20
    step = pos_uav.size()[0]

    gip = int(node_num / 3)
    group = np.zeros(node_num)
    group[0:gip] = 3
    group[gip:(2 * gip)] = 2
    group[(2 * gip):node_num] = 1
    
    eval_v1 = np.zeros([step, 2])
    eval_v2 = np.zeros([step, 2])
    # uav_network = torch.zeros([5, step, node_num, node_num])
    uav_network = torch.zeros([step, node_num, node_num])
    uav_network_att = torch.zeros([step, node_num, node_num])

    for i in range(1):
        for j in range(step):
            dis_uav = squareform(pdist(pos_uav_att[j, :, :].numpy()))
            # dis_uav = squareform(pdist(pos_uav[i, j, :, :].numpy()))
            adj_uav1 = get_adj(dis_uav, R, group)
            eval_v1[j,:] = eval_adj(adj_uav1)
            # eval_v1[j,:] = eval_network(adj_uav1)
            # uav_network[i, j, :, :] = torch.from_numpy(adj_uav)
            uav_network_att[j, :, :] = torch.from_numpy(adj_uav1)
            
            # dis_uav = squareform(pdist(pos_uav[j, :, :].numpy()))
            # # dis_uav = squareform(pdist(pos_uav[i, j, :, :].numpy()))
            # adj_uav2 = get_adj(dis_uav, R, group)
            # uav_network[j, :, :] = torch.from_numpy(adj_uav2)
            # eval_v2[j, :] = eval_adj(adj_uav2)
            # eval_v2[j, :] = eval_network(adj_uav2)

    # torch.save(uav_network_att, 'optimized_network/uav_network_att_test_111')
    # torch.save(uav_network, 'optimized_network/uav_network_111')
    
    # plt.plot(eval_v1[:, 1],'r', label='method_att')
    # plt.plot(eval_v2[:, 1], 'g', label='method')
    # # plt.legend()
    # plt.show()
    # plt.close()

    #plt.figure(figsize=(9, 7))
    fig = plt.figure()
    left, bottom, width, height = 0.18, 0.15, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    
    ax.plot(eval_v1[:, 1], color='orange', linewidth=3)
    #print(eval_v1[:, 1])
    # plt.plot(eval_v2[:, 1], 'g', label='method')
    #plt.legend()
    # plt.xlabel("Times", fontsize=20)
    # plt.ylabel("Average number of neighbors", fontsize=20)
    plt.xlabel("时间", fontsize=20)
    plt.ylabel("平均邻居数量", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.text(10, 350, '(d)', fontsize=25)
    plt.text(30, 650, '(e)', fontsize=25)
    plt.text(95, 760, '(f)', fontsize=25)

    # plt.text(10, 4200, '(d)', fontsize=25)
    # plt.text(30, 6200, '(e)', fontsize=25)
    # plt.text(95, 8700, '(f)', fontsize=25)
    #plt.savefig('pic/neb_33_c.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    