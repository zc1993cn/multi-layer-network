from scipy.spatial.distance import pdist, squareform
from swarm_dqn import DQN
from swarm_dqn import Net
from swarm_dqn import GraphConvolution
import networkx as nx
import numpy as np
import env_swarm
import torch
import matplotlib.pyplot as plt

# device = torch.device("cuda:1")
device = torch.device("cpu")

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
    # ps = nx.shell_layout(G)
    # nx.draw(G, ps, with_labels=False, node_size=6.png)
    return tag, d


def evaluate_network(adj):
    eval_vec = np.zeros(2)
    # 连通性指标，传输效率指标
    # conn, d = nx_graph(adj)

    # 网络连边
    count = np.sum(adj)

    # eval_vec[0] = conn
    eval_vec[1] = count

    return torch.from_numpy(eval_vec)


def draw_evaluation(e_v):
    data1 = e_v[0, :, 0].detach().numpy()
    data2 = e_v[0, :, 1].detach().numpy()

    plt.plot(data1)
    plt.show()
    plt.close()

    plt.plot(data2)
    plt.show()
    plt.close()


def process_dis(dis_all, group):
    for i in range(len(group)):
        diff_abs = np.abs(group - group[i])
        idx = np.where(diff_abs > 1)[0]
        dis_all[i, idx] = 100
    
    return dis_all

if __name__ == '__main__':
    R = 20
    TIMES = 5
    NODE_NUM = 33
    DIS_MAX = 40
    gip = int(NODE_NUM / 3)

    uav_network = torch.zeros([TIMES, 100, NODE_NUM, NODE_NUM])
    uav_group = np.zeros(NODE_NUM)
    uav_group[0:gip] = 3
    uav_group[gip:(2 * gip)] = 2
    uav_group[(2 * gip):NODE_NUM] = 1

    file_rl = 'dqn_model/dqn_33_200_2'

    model_rl = torch.load(file_rl)
    model_rl.eval_net.to(device)
    model_rl.target_net.to(device)

    #file_path = 'uav_positions/uav_test_att_111'
    file_path = 'uav_positions/uav_test_att_33'
    sum_edges = 0

    for i in range(TIMES):
        file_name = file_path
        # str_num = str(i)
        # file_name = file_path + str_num
        pos_uav = torch.load(file_name)

        A = np.zeros([NODE_NUM, NODE_NUM])

        for j in range(100):
            t_A = torch.from_numpy(A)

            sum_edges = sum_edges + np.sum(A)
            
            uav_dis = squareform(pdist(pos_uav[j, :, :].numpy()))
            dis = process_dis(uav_dis, uav_group)
            dis_i = env_swarm.get_uav_input(dis, R)

            nor_dis_i = dis_i / DIS_MAX

            a = model_rl.choose_action(t_A.to(device), nor_dis_i.to(device))

            A_next = env_swarm.step(a, R, dis)
            uav_network[i, j, :, :] = torch.from_numpy(A_next)
            
            # evaluate_v[i, j, :] = evaluate_network(A_next)

            A = A_next.copy()
            
        avg_edges = sum_edges / 100

        # print(torch.mean(evaluate_v[i, :, 1]))

    #  draw_evaluation(evaluate_v)



    #torch.save(uav_network, "optimized_network/uav_network_111_dqn_4")





