import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import random
import torch

plt.rc('font', family='Times New Roman')

def create_graph(adj):
    G = nx.Graph()
    tmp = adj

    for i in range(0, len(tmp)):
        G.add_node(i)

    for i in range(0, len(tmp)):
        idx = np.where(tmp[i, :] == 1)
        sq_idx = idx[0]
        for j in range(0, len(sq_idx)):
            G.add_edge(i, sq_idx[j])

    return G


def max_degree_attack(net, rat):
    adj = nx.adjacency_matrix(net).todense()
    deg = np.sum(adj, axis=1)

    n = adj.shape[0]
    r_n = int(n * rat)
    idx = np.argsort(-np.squeeze(deg.A, axis=1))

    # print(s_deg)
    # print(np.squeeze(deg.A))

    for i in range(r_n):
        net.remove_node(idx[i])

    return net


def random_attack(net, rat):
    n_net = len(net)
    # print(n)
    seq_nodes = np.arange(0, n_net, 1)
    r_n = int(n_net * rat)
    idx = np.array(random.sample(set(seq_nodes), r_n))

    for i in range(r_n):
        net.remove_node(idx[i])

    return net


def betweenness_attack(net, rat):
    n = len(net)
    r_n = int(n * rat)

    score = nx.betweenness_centrality(net)
    s_score = sorted(score.items(), key=lambda x: x[1], reverse=False)

    for i in range(r_n):
        ele = s_score.pop()
        idx = ele[0]
        net.remove_node(idx)

    return net

def get_component(net):
    largest_size = 0
    for g in nx.connected_components(net):
        if len(g) > largest_size:
            largest_size = len(g)

    return largest_size

def draw_conn_cmp(cpt_data):

    plt.figure(figsize=(9, 7))
    # plt.figure()
    x_str = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    #x_str = ["0.0", "0.1", "0.2", "0.3", "0.4"]

    plt.plot(cpt_data[0, :], 'g', linewidth=4, label='Entropy')
    plt.plot(cpt_data[1, :], 'b', linewidth=4, label='Allocation')
    plt.plot(cpt_data[2, :], 'r', linewidth=4, label='Mitigation')
    # plt.plot(cpt_data[2, :], 'g', linewidth=4, label='our')

    #plt.xticks(np.arange(0, 5, 1), x_str, fontsize=20)
    plt.xticks(np.arange(0, 11, 1), x_str, fontsize=50)
    plt.yticks(fontsize=50)

    plt.legend(fontsize=20)
    # plt.xlabel("The ratio", fontsize=25)
    # plt.ylabel("The average of the max components", fontsize=25)

    #plt.savefig('attack_random_33.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def get_avg(cpt, cpt_att, cpt_dqn):
    avg_cpt = np.zeros([3, cpt.shape[0]])
    avg_cpt[0, :] = np.mean(cpt, axis=1)
    avg_cpt[1, :] = np.mean(cpt_att, axis=1)
    avg_cpt[2, :] = np.mean(cpt_dqn, axis=1)

    return avg_cpt / 100

def eval_network(net, node_n):
    node_total = 0
    gip = int(node_n / 3)

    s1 = set(np.arange(0, gip, 1))
    s2 = set(np.arange(gip, 2*gip, 1))
    s3 = set(np.arange(2*gip, 3*gip, 1))
    
    for g in nx.connected_components(net):
        if len(g) > 0:
            f1 = g.intersection(s1)
            f2 = g.intersection(s2)
            f3 = g.intersection(s3)
            
            if len(f1) > 0:
                if len(f2) > 0:
                    if len(f3) > 0:
                        node_total = node_total + len(g)
                        
    return node_total
    

if __name__ == '__main__':
    
    adj_entropy = np.load('cmp_method/entropy_network_111.npy')
    adj_allocation = np.load('cmp_method/allocation_network_111.npy')
    adj_mitigation = np.load('cmp_method/mitigation_network_111.npy')
    #uav_adj_dqn = torch.load('optimized_network/uav_network_111_dqn_0')
    
    step = 100
    node_num= 111
    r_s = 1
    attack_pro = 11
    ratio = 0

    conn_cpt_entropy = np.zeros(step)
    conn_cpt_allocation = np.zeros(step)
    conn_cpt_mitigation = np.zeros(step)

    sum_conn_cpt_entropy = np.zeros([attack_pro, r_s])
    sum_conn_cpt_allocation = np.zeros([attack_pro, r_s])
    sum_conn_cpt_mitigation = np.zeros([attack_pro, r_s])


    for n in range(attack_pro):
        for m in range(r_s):
            for it in range(step):
                graph_entropy = create_graph(adj_entropy[it, :, :])
                graph_allocation = create_graph(adj_allocation[it, :, :])
                graph_mitigation = create_graph(adj_mitigation[it, :, :])
                #graph_dqn = create_graph(adj_mitigation[m, it, :, :])
                
                # attack_entropy = random_attack(graph_entropy, ratio)
                # attack_allocation = random_attack(graph_allocation, ratio)
                # attack_mitigation = random_attack(graph_mitigation, ratio)

                # attack_entropy = max_degree_attack(graph_entropy, ratio)
                # attack_allocation = max_degree_attack(graph_allocation, ratio)
                # attack_mitigation = max_degree_attack(graph_mitigation, ratio)

                attack_entropy = betweenness_attack(graph_entropy, ratio)
                attack_allocation = betweenness_attack(graph_allocation, ratio)
                attack_mitigation = betweenness_attack(graph_mitigation, ratio)
            #
                conn_cpt_entropy[it] = eval_network(attack_entropy, node_num)
                conn_cpt_allocation[it] = eval_network(attack_allocation, node_num)
                conn_cpt_mitigation[it] = eval_network(attack_mitigation, node_num)

            sum_conn_cpt_entropy[n, m] = np.sum(conn_cpt_entropy)
            sum_conn_cpt_allocation[n, m] = np.sum(conn_cpt_allocation)
            sum_conn_cpt_mitigation[n, m] = np.sum(conn_cpt_mitigation)
            
        ratio = ratio + 0.1

    avg_conn = get_avg(sum_conn_cpt_entropy, sum_conn_cpt_allocation, sum_conn_cpt_mitigation)
    draw_conn_cmp(avg_conn)
    np.savetxt('attack_cmp/bet_111_1', avg_conn)



