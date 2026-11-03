import torch
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
plt.rc('font', family='Times New Roman')

def draw_static_graph(UAV_pos, l_num, t):
    fig = plt.figure(figsize=(7, 6))
    ax = Axes3D(fig)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=10, azim=150)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    ax.set_zlim(0, 60)
    
    for i in range(l_num):
        obs_x, obs_y, obs_z = get_UAVS_coordinate(UAV_pos[t, i, :], 0.7)
        ax.plot_surface(obs_x, obs_y, obs_z, rstride=4, cstride=4, color='b')

    for i in range(l_num, 2 * l_num):
        obs_x, obs_y, obs_z = get_UAVS_coordinate(UAV_pos[t, i, :], 0.7)
        ax.plot_surface(obs_x, obs_y, obs_z, rstride=4, cstride=4, color='g')
        
    for i in range(2 * l_num, 3 * l_num):
        obs_x, obs_y, obs_z = get_UAVS_coordinate(UAV_pos[t, i, :], 0.7)
        ax.plot_surface(obs_x, obs_y, obs_z, rstride=4, cstride=4, color='r')

    plt.savefig('pos_111_0.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
def draw_trace(trace_matrix):
    # entity_vec = ['entity1', 'entity2', 'entity3']
    # font = {
    #     'color': 'k',
    #     'style': 'oblique',
    #     'size': 20,
    #     'weight': 'bold'
    # }

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=10, azim=150)
    # ax.set_xlim(0, 1000)
    # ax.set_ylim(0, 800)
    # ax.set_zlim(0, 600)
    
    for i in range(0, 3):
        x_trace = trace_matrix[:, i, 0].detach().numpy()
        y_trace = trace_matrix[:, i, 1].detach().numpy()
        z_trace = trace_matrix[:, i, 2].detach().numpy()

        # ax.scatter3D(x_trace, y_trace, z_trace)  # 绘制散点图 cmap='Blues'
        ax.plot3D(x_trace, y_trace, z_trace,color='b')

    for i in range(37, 40):
        x_trace = trace_matrix[:, i, 0].detach().numpy()
        y_trace = trace_matrix[:, i, 1].detach().numpy()
        z_trace = trace_matrix[:, i, 2].detach().numpy()
    
            # ax.scatter3D(x_trace, y_trace, z_trace)  # 绘制散点图 cmap='Blues'
        ax.plot3D(x_trace, y_trace, z_trace,color='g')

        for i in range(74, 77):
            x_trace = trace_matrix[:, i, 0].detach().numpy()
            y_trace = trace_matrix[:, i, 1].detach().numpy()
            z_trace = trace_matrix[:, i, 2].detach().numpy()
    
            # ax.scatter3D(x_trace, y_trace, z_trace)  # 绘制散点图 cmap='Blues'
            ax.plot3D(x_trace, y_trace, z_trace,color='r')
        # ax.legend()

    plt.show()
    plt.close()

def get_UAVS_coordinate(center, radius):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0].detach().numpy()
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1].detach().numpy()
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2].detach().numpy()

    return x, y, z
    
def draw_dynamic_graph(UAV_pos, times, l_num):

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
    ax.view_init(elev=30, azim=120)
    ax.set_title("Entity Traces", fontdict=font)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    ax.set_zlim(0, 60)
    
    def update(t):
        ax.cla()

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=120)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)
        ax.set_zlim(0, 60)
        for i in range(l_num):
            obs_x, obs_y, obs_z = get_UAVS_coordinate(UAV_pos[t, i, :], 0.7)
            # surface plot rstride 值越大，图像越粗糙
            ax.plot_surface(obs_x, obs_y, obs_z, rstride=4, cstride=4, color='b')
            
        for i in range(l_num, 2 * l_num):
            obs_x, obs_y, obs_z = get_UAVS_coordinate(UAV_pos[t, i, :], 0.7)
            ax.plot_surface(obs_x, obs_y, obs_z, rstride=4, cstride=4, color='g')
            
        for i in range(2 * l_num, 3 * l_num):
            obs_x, obs_y, obs_z = get_UAVS_coordinate(UAV_pos[t, i, :], 0.7)
            ax.plot_surface(obs_x, obs_y, obs_z, rstride=4, cstride=4, color='r')
        # ax.legend()

    ant = FuncAnimation(fig, update, frames=times, interval=500, blit=False, repeat=False)  # 创建动画效果
    plt.show()
    # ani.save('line.gif', writer='pillow')
    plt.close()

def init_position_layer(uav_n, g_layer):
    gip = int(uav_n / 3)
    uav_pos = np.zeros([uav_n, 3])
    group = np.zeros(uav_n)
    # g_r = np.arange(0, 1, 0.02) # 33 UAVs
    g_r = np.arange(0, 3, 0.02)

    group[0:gip] = 3
    group[gip:(2 * gip)] = 2
    group[(2 * gip):uav_n] = 1

    uav_pos[:, 0] = np.array(random.sample(set(g_r), uav_n))
    uav_pos[:, 1] = np.array(random.sample(set(g_r), uav_n))
    uav_pos[:, 2] = np.array(random.sample(set(g_r), uav_n))
    
    for i in range(3):
        uav_pos[0:gip, i] = uav_pos[0:gip, i] + 3 * g_layer
        uav_pos[gip:(2 * gip), i] = uav_pos[gip:(2 * gip), i] + 2 * g_layer
        uav_pos[(2 * gip):uav_n, i] = uav_pos[(2 * gip):uav_n, i] + g_layer

    return uav_pos, group

def compute_unit_vector(vs, ve):
    v_direct = ve - vs
    modulus = np.sqrt(np.dot(v_direct, v_direct))

    if modulus == 0:
        normal_v = np.zeros(3)

    else:
        normal_v = v_direct / modulus

    return normal_v

def v_repulsion(pos, obj_pos, obj_dis, s_r):
    v_force = np.zeros([obj_dis.shape[0], 3])
    
    for i in range(obj_pos.shape[0]):
        v_direct = compute_unit_vector(obj_pos[i, :], pos)
        force_mag = (1 / obj_dis[i] - 1 / s_r) * ((1 / obj_dis[i]) ** 3)
            
        v_force[i] = v_direct * force_mag

    v_rep = np.sum(v_force, axis=0)

    return v_rep

def v_attraction(id_g, pos, obj_pos, obj_dis, neb_g, gip):
    v_force = np.zeros([obj_dis.shape[0], 3])
    
    for i in range(len(neb_g)):
        v_direct = compute_unit_vector(pos, obj_pos[i, :])
        tmp = np.zeros(3)
        
        if id_g == neb_g[i]:
            force_mag = obj_dis[i]
            v_force[i, :] = v_direct * force_mag
        
        if 0 < int(id_g - neb_g[i]) and np.abs(int(id_g - neb_g[i])) < 2:

            x_diff = np.abs(pos[0] - obj_pos[i, 0])
            y_diff = np.abs(pos[1] - obj_pos[i, 1])
            z_diff = np.abs(pos[2] - obj_pos[i, 2])

            x_force_mag = (x_diff - gip) * obj_dis[i]
            tmp[0] = x_force_mag

            y_force_mag = (y_diff - gip) * obj_dis[i]
            tmp[1] = y_force_mag

            z_force_mag = (z_diff - gip) * obj_dis[i]
            tmp[2] = z_force_mag
            v_force[i, :] = v_direct * tmp

    v_att = np.sum(v_force, axis=0)

    return v_att

def get_adj(pos, cr):
    num = pos.shape[0]
    adj = np.zeros([num, num])
    
    dis_uav_tmp = squareform(pdist(pos))
    dis_uav_uav = dis_uav_tmp + np.eye(num) * 10000.0
    
    for i in range(num):
        idx = np.where(dis_uav_uav[i, :] - cr < 0)
        adj[i, idx] = 1
    
    return adj
    
def attack_degree(uav_sys, adj, rat):
    num = adj.shape[0]
    uav_degree = np.sum(adj, axis=1)
    sort_degree = np.argsort(-uav_degree)
    delete_num = np.floor(uav_sys.uav_num * rat)
    delete_node = sort_degree[0:int(delete_num)]
    tmp = np.zeros([num, 3])
    tmp[delete_node, :] = tmp[delete_node, :] + 10000.0
    n_pos = uav_sys.position + tmp
    
    return n_pos

class UAVSystem:
    def __init__(self, num, pos_mat, safety_r, comm_r, v, group, gip):
        self.uav_num = num
        self.safety_range = safety_r
        self.position = pos_mat
        self.communication_range = comm_r
        self.velocity = v
        self.group = group
        self.gip_layer = gip
        self.alpha = 10
        self.beta = 1e10 #10
        
    def adjust_position(self, id_uav, pos, dis_uav):
        dv = np.zeros([1, 3])
        comm_idx = np.where(dis_uav < self.communication_range)[0]

        if comm_idx.size > 0:
            id_group = self.group[id_uav]
            neb_group = self.group[comm_idx]
            tmp = v_attraction(id_group, pos, self.position[comm_idx,:], dis_uav[comm_idx],
                               neb_group, self.gip_layer)
            dv[0, :] = self.alpha * tmp

        resultant_dv = np.sum(dv, axis=0)
        
        return resultant_dv
    
    def adjust_safety_distance(self, pos, dis_uav):
        dv = np.zeros([1, 3])
        sr_idx = np.where(dis_uav < self.safety_range)[0]
        
        if sr_idx.size > 0:
            tmp = v_repulsion(pos, self.position[sr_idx,:], dis_uav[sr_idx],
                              self.safety_range)
            dv[0, :] =  self.beta * tmp

        resultant_dv = np.sum(dv, axis=0)

        return resultant_dv
        
    def get_moved_position(self, m_limit):
        direct_v = np.arange(0, 4, 1)
        direct = np.array(random.sample(set(direct_v), 1))
        
        pos_min = np.zeros(2)
        pos_max = np.zeros(2)
        new_pos = np.zeros([self.uav_num, 3])
        
        pos_min[0] = self.position[:, 0].min()
        pos_min[1] = self.position[:, 1].min()

        pos_max[0] = self.position[:, 0].max()
        pos_max[1] = self.position[:, 1].max()
        
        flag = 0
        if direct == 0: # forward
            if pos_max[0] < m_limit:
                new_pos = self.position + np.array([self.velocity, 0, 0])
                flag = 1

        if direct == 1: # backward
            if pos_min[0] > 0:
                new_pos = self.position - np.array([self.velocity, 0, 0])
                flag = 1
        
        if direct == 2: # leftward
            if pos_min[1] > 0:
                new_pos = self.position - np.array([0, self.velocity, 0])
                flag = 1

        if direct == 3: # rightward
            if pos_max[1] < m_limit:
                new_pos = self.position + np.array([0, self.velocity, 0])
                flag = 1
                
        if flag == 0:
            new_pos = self.position
                
            
        return new_pos
        
    def get_next_position(self, m_bound):
        pos_adjust = np.zeros([self.uav_num, 3])
        pos_safety = np.zeros([self.uav_num, 3])

        moved_position = self.get_moved_position(m_bound)
        dis_uav_tmp = squareform(pdist(moved_position))
        dis_uav_uav = dis_uav_tmp + np.eye(self.uav_num) * 1000.0
        self.position = moved_position

        for i in range(self.uav_num):
            r_x = float(np.random.uniform(-0.1, 0.1, 1))
            r_y = float(np.random.uniform(-0.1, 0.1, 1))
            r_z = float(np.random.uniform(-0.1, 0.1, 1))
            r_b = 1 * np.array([r_x, r_y, r_z])
            
            pos_adjust[i, :] = self.adjust_position(i, moved_position[i, :], dis_uav_uav[i, :])
            pos_safety[i, :] = self.adjust_safety_distance(self.position[i, :], dis_uav_uav[i, :])
            d_post = pos_adjust[i, :] + pos_safety[i, :]

            v_length = np.sqrt(np.dot(d_post, d_post))
            if self.velocity < v_length:
                d_post = d_post / v_length

            self.position[i, :] = self.position[i, :] + d_post + r_b
        
        return self.position
    

if __name__ == '__main__':
    uav_num = 33
    step = 110
    gip_layer_UAVs = 9
    UAV_v = 2  # 2
    safety_range = 2.0
    communication_range = 80
    map_bound = 60
    layer_num = int(uav_num / 3)

    start_point = np.array([5, 5, 5])
    
    all_position = torch.zeros([step, uav_num, 3])

    uav_position, uav_group = init_position_layer(uav_num, 15)

    us_obj = UAVSystem(uav_num, uav_position, safety_range, communication_range, UAV_v, uav_group, gip_layer_UAVs)

    all_position[0, :, :] = torch.from_numpy(uav_position)

    for it in range(step - 1):
        next_position = us_obj.get_next_position(map_bound)
        all_position[it + 1, :, :] = torch.from_numpy(next_position)

    #draw_static_graph(all_position, layer_num, 99)
    #draw_trace(all_position)
    # draw_dynamic_graph(all_position, step, layer_num)
    #torch.save(all_position, 'uav_positions/uav_test_att_33')
    #torch.save(all_position[4:105, :, :], 'uav_positions/uav_test_att_33')
    #torch.save(all_position, 'uav_positions/uav_att_111_1')
    torch.save(all_position, 'uav_positions/uav_att_33_1')
    
    #torch.save(all_position[10:110, :, :], 'uav_positions/uav_111')
    #torch.save(all_position[10:110, :, :], 'uav_positions/uav_test_att_111')
    #torch.save(all_position[10:110, :, :], 'uav_positions/uav_test_att_111_draw')

    # print(all_position[:, 30, :])


