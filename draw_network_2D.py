# --*-- coding: utf-8 --*--
# @Author: dgon
# @Email: macaibei20@nudt.edu.cn
# @Time: 2023/2/15 下午4:52
# @File: draw_network2D.py
# @Software: PyCharm
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

def init_pos(num, l_num):
	pos = np.zeros([num, 2])
	x = 5
	for it in range(l_num):
		pos[it, 0] = x
		pos[it, 1] = 120
		x = x + 10
	
	x = 5
	for it in range(l_num, 2 * l_num):
		pos[it, 0] = x
		pos[it, 1] = 80
		x = x + 10
	
	x = 5
	for it in range(2 * l_num, 3 * l_num):
		pos[it, 0] = x
		pos[it, 1] = 40
		x = x + 10
	
	return pos
	
def draw_circle(pos, r, col):
	circ = Circle(xy=(pos[0], pos[1]), radius=r, alpha=1, color=col)
	ax.add_patch(circ)
	ax.plot()
	
def draw_edges(id_uav, adj, pos):
	idx = np.where(adj[id_uav, :] == 1)[0]
	for it in range(idx.shape[0]):
		
		ax.plot([pos[id_uav, 0], pos[idx[it], 0]], [pos[id_uav, 1], pos[idx[it], 1]])
		
	return 0


if __name__ == '__main__':
	node_num = 111
	layer_num = int(node_num / 3)
	rad = 4
	adj_uav = torch.load('optimized_network/uav_network_111_dqn_0')
	adj_times = adj_uav[0, 10, :, :].numpy()
	fig = plt.figure()
	ax = fig.add_subplot()
	
	uav_pos = init_pos(node_num, layer_num)
	
	for i in range(layer_num):
		draw_circle(uav_pos[i, :], rad, 'b')
		draw_edges(i, adj_times, uav_pos)

	for i in range(layer_num, 2*layer_num):
		draw_circle(uav_pos[i, :], rad, 'g')
		draw_edges(i, adj_times, uav_pos)

	for i in range(2*layer_num, 3*layer_num):
		draw_circle(uav_pos[i, :], rad, 'r')
		draw_edges(i, adj_times, uav_pos)


	plt.show()
	plt.close()
	
	
	
	
	