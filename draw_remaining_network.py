# --*-- coding: utf-8 --*--
# @Author: dgon
# @Email: macaibei20@nudt.edu.cn
# @Time: 2023/3/13 上午11:06
# @File: draw_remaining_network.py
# @Software: PyCharm
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


def draw_graph(net, g, str_file):
	color_map_node = []
	color_map_edges = []
	idx = list(net)
	edges = list(net.edges())
	for it_n in range(len(idx)):
		# if 0 <= idx[it_n] < 37:
		# 	color_map_node.append('b')
		#
		# if 37 <= idx[it_n] < 74:
		# 	color_map_node.append('g')
		#
		# if 74 <= idx[it_n] < 111:
		# 	color_map_node.append('r')
		
		if 0 <= idx[it_n] < 11:
			color_map_node.append('b')

		if 11 <= idx[it_n] < 22:
			color_map_node.append('g')

		if 22 <= idx[it_n] < 33:
			color_map_node.append('r')
		
	for it_e in range(len(edges)):
		if np.abs(g[edges[it_e][0]] - g[edges[it_e][1]]) == 0:
			color_map_edges.append('orange')
		else:
			color_map_edges.append('purple')
		
	
	#kara_pos = nx.spring_layout(net, k=0.5)
	#kara_pos = nx.circular_layout(net)
	#kara_pos = nx.random_layout(net)
	#kara_pos = nx.spectral_layout(net) edge_color
	file_name = 'pic/bet_0.4_33_' + str_file + '.pdf'
	kara_pos = nx.shell_layout(net)
	labels = {node: str(node) for node in net.nodes}
	plt.figure()
	#nx.draw_networkx(net, kara_pos, node_color=color_map)
	#nx.draw(net, kara_pos, with_labels=True, labels=labels, node_color=color_map_node, edge_color=color_map_edges)
	nx.draw(net, kara_pos, node_color=color_map_node, edge_color=color_map_edges)
	plt.savefig(file_name, bbox_inches='tight')
	plt.show()
	plt.close()


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
	n = len(net)
	# print(n)
	seq_nodes = np.arange(0, n, 1)
	r_n = int(n * rat)
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


if __name__ == '__main__':#
	
	# uav_adj = torch.load('optimized_network/uav_network_111')
	# uav_adj_att = torch.load('optimized_network/uav_network_att_test_111')
	# uav_adj_dqn = torch.load('optimized_network/uav_network_111_dqn_0')
	# adj_entropy = np.load('cmp_method/entropy_network_111.npy')
	# adj_allocation = np.load('cmp_method/allocation_network_111.npy')
	# adj_mitigation = np.load('cmp_method/mitigation_network_111.npy')
	# adj_dqn = torch.load('optimized_network/uav_network_111_dqn_2')
	
	adj_entropy = np.load('cmp_method/entropy_network_33.npy')
	adj_allocation = np.load('cmp_method/allocation_network_33.npy')
	adj_mitigation = np.load('cmp_method/mitigation_network_33.npy')
	adj_dqn = torch.load('optimized_network/uav_network_33_dqn_2')
	
	ratio = 0.4
	time = 40 #19
	node_num = 33
	gip = int(node_num / 3)
	uav_group = np.zeros(node_num)
	uav_group[0:gip] = 3
	uav_group[gip:(2 * gip)] = 2
	uav_group[(2 * gip):node_num] = 1
	
	graph_entropy = create_graph(adj_entropy[time, :, :])
	graph_allocation = create_graph(adj_allocation[time, :, :])
	graph_mitigation = create_graph(adj_mitigation[time, :, :])
	graph_dqn = create_graph(adj_dqn[0, time, :, :])
	
	# attack_entropy = random_attack(graph_entropy, ratio)
	# attack_allocation = random_attack(graph_allocation, ratio)
	# attack_mitigation = random_attack(graph_mitigation, ratio)
	# attack_dqn = random_attack(graph_dqn, ratio)
	
	# attack_entropy = max_degree_attack(graph_entropy, ratio)
	# attack_allocation = max_degree_attack(graph_allocation, ratio)
	# attack_mitigation = max_degree_attack(graph_mitigation, ratio)
	# attack_dqn = max_degree_attack(graph_dqn, ratio)
	
	attack_entropy = betweenness_attack(graph_entropy, ratio)
	attack_allocation = betweenness_attack(graph_allocation, ratio)
	attack_mitigation = betweenness_attack(graph_mitigation, ratio)
	attack_dqn = betweenness_attack(graph_dqn, ratio)
	
	draw_graph(attack_entropy, uav_group,'entropy')
	draw_graph(attack_allocation, uav_group, 'allocation')
	draw_graph(attack_mitigation, uav_group, 'mitigation')
	draw_graph(attack_dqn, uav_group,'dqn')




