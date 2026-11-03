# --*-- coding: utf-8 --*--
# @Author: dgon
# @Email: macaibei20@nudt.edu.cn
# @Time: 2023/2/17 上午10:13
# @File: max_dis_test.py
# @Software: PyCharm
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

def process_dis(dis_all, group):
	for i in range(len(group)):
		diff_abs = np.abs(group - group[i])
		idx = np.where(diff_abs > 1)[0]
		dis_all[i, idx] = 0
	
	return dis_all


if __name__ == '__main__':
	pos_uav = torch.load('uav_positions/uav_test_att_111')
	uav_n = pos_uav.size()[1]
	gip = int(uav_n / 3)
	
	uav_group = np.zeros(uav_n)
	uav_group[0:gip] = 3
	uav_group[gip:(2 * gip)] = 2
	uav_group[(2 * gip):uav_n] = 1
	
	cur_max = 0
	
	for j in range(100):
		uav_dis = squareform(pdist(pos_uav[j, :, :].numpy()))
		# dis = process_dis(uav_dis, uav_group)
		# d_max = dis.max()
		d_max = uav_dis.max()
		if d_max > cur_max:
			cur_max = d_max
			
	print(cur_max)
		
	
