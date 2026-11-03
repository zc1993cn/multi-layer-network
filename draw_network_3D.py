# --*-- coding: utf-8 --*--
# @Author: dgon
# @Email: macaibei20@nudt.edu.cn
# @Time: 2023/2/11 下午1:21
# @File: draw_graph_3D.py
# @Software: PyCharm
import torch
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def get_UAVS_coordinate(center, radius):
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
	y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
	z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
	
	return x, y, z

def draw_edges(ax_line, pos, pos_uav, adj_i):
	idx = np.where(adj_i == 1)[0]
	for i in range(len(idx)):
		x = [pos[0], pos_uav[idx[i], 0]]
		y = [pos[1], pos_uav[idx[i], 1]]
		z = [pos[2], pos_uav[idx[i], 2]]
		
		ax_line.plot(x, y, z)
		
def draw_graph(UAV_pos, adj, gip):
	font = {
		'color': 'k',
		'style': 'oblique',
		'size': 20,
		'weight': 'bold'
	}
	
	fig = plt.figure()
	ax = Axes3D(fig)
	
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.view_init(elev=20, azim=100)
	ax.set_title("Entity Traces", fontdict=font)
	ax.set_xlim(0, 40)
	ax.set_ylim(0, 40)
	ax.set_zlim(0, 40)
	
	for i in range(gip):
		obs_x, obs_y, obs_z = get_UAVS_coordinate(UAV_pos[i, :], 1)
		# surface plot rstride 值越大，图像越粗糙
		ax.plot_surface(obs_x, obs_y, obs_z, rstride=4, cstride=4, color='b')
		draw_edges(ax, UAV_pos[i, :], UAV_pos[:, :], adj[i, :])

	for i in range(gip):
		obs_x, obs_y, obs_z = get_UAVS_coordinate(UAV_pos[gip + i, :], 1)
			# surface plot rstride 值越大，图像越粗糙
		ax.plot_surface(obs_x, obs_y, obs_z, rstride=4, cstride=4, color='g')
		draw_edges(ax, UAV_pos[11 + i, :], UAV_pos[:, :], adj[gip + i, :])

	for i in range(gip):
		obs_x, obs_y, obs_z = get_UAVS_coordinate(UAV_pos[2 * gip + i, :],1)
		# surface plot rstride 值越大，图像越粗糙
		ax.plot_surface(obs_x, obs_y, obs_z, rstride=4, cstride=4, color='r')
		draw_edges(ax, UAV_pos[22 + i, :], UAV_pos[:, :], adj[2 * gip + i, :])
	
	# ax.legend()
	plt.show()
	plt.close()
	
if __name__ == '__main__':
	node_num = 33
	times = 10
	uav_gip = 11
	# pos_uav = torch.load('uav_train_att_33')
	# adj_uav = torch.load('uav_network_att_33')
	
	# pos_uav = torch.load('uav_train_33')
	# adj_uav = torch.load('uav_network_33')
	
	pos_uav = torch.load('uav_positions/uav_test_att_33')
	adj_uav = torch.load('optimized_network/uav_network_33_dqn_2')
	
	# adj_uav = torch.load('uav_network_test')
	
	pos_uav_time = pos_uav[times, :, :].detach().numpy()
	adj_uav_time = adj_uav[0, times, :, :].detach().numpy()
	draw_graph(pos_uav_time, adj_uav_time, uav_gip)
	
