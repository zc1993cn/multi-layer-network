import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import networkx as nx
import env_swarm
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

device = torch.device("cuda:0")
# device = torch.device("cpu")
# Hyper Parameters
BATCH_SIZE = 64
LR = 0.01                   # learning rate
EPSILON = 0.8               # greedy policy
GAMMA = 0.9                # reward discount
TARGET_REPLACE_ITER = 80   # target update frequency
MEMORY_CAPACITY = 256
N_ACTIONS = 110
NODE_NUM = 111
INPUT_DIM = 111

R = 20
DIS_MAX = 40


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)
        # output = torch.sparse.mm(adjacency, support)
        output = torch.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# ## 模型定义

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        # self.gcn2 = GraphConvolution(input_dim, 16)

        self.fc1 = nn.Linear(16, 64)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(64, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, adjacency, feature):
        h1 = F.relu(self.gcn1(adjacency, feature))
        # h2 = F.relu(self.gcn2(adjacency, h1))

        x = self.fc1(h1)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, input_dim):
        # self.eval_net, self.target_net = Net(input_dim).to(device), Net(input_dim).to(device)
        self.eval_net, self.target_net = Net(input_dim), Net(input_dim)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory_s = torch.zeros(MEMORY_CAPACITY, NODE_NUM, NODE_NUM)
        self.memory_s_next = torch.zeros(MEMORY_CAPACITY, NODE_NUM, NODE_NUM)
        self.memory_r = torch.zeros(MEMORY_CAPACITY)
        self.memory_a = torch.zeros(MEMORY_CAPACITY, NODE_NUM)
        self.memory_f = torch.zeros(MEMORY_CAPACITY, NODE_NUM, input_dim)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, adj, feature):
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(adj.float(), feature.float())
            action = torch.max(actions_value.cpu(), 1)[1].data.numpy()
            action = action
        else:   # random
            tmp = [np.random.randint(0, N_ACTIONS) for i in range(NODE_NUM)]
            action = np.array(tmp)
        return action

    def store_transition(self, s, store_a, store_r, s_, f):
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory_s[index, :] = s
        self.memory_s_next[index, :] = s_
        self.memory_a[index, :] = torch.from_numpy(store_a)
        self.memory_r[index] = store_r
        self.memory_f[index, :] = f
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # b_s = torch.FloatTensor(self.memory_s[sample_index, :, :]).to(device)
        # b_a = torch.LongTensor(self.memory_a[sample_index].numpy().astype(int)).to(device)
        # b_r = torch.FloatTensor(self.memory_r[sample_index]).to(device)
        # b_s_ = torch.FloatTensor(self.memory_s_next[sample_index, :, :]).to(device)
        # b_f = torch.FloatTensor(self.memory_f[sample_index, :, :]).to(device)

        b_s = torch.FloatTensor(self.memory_s[sample_index, :, :])
        b_a = torch.LongTensor(self.memory_a[sample_index].numpy().astype(int))
        b_r = torch.FloatTensor(self.memory_r[sample_index])
        b_s_ = torch.FloatTensor(self.memory_s_next[sample_index, :, :])
        b_f = torch.FloatTensor(self.memory_f[sample_index, :, :])

        b_q_eval = torch.zeros(BATCH_SIZE * NODE_NUM, 1)
        b_q_target = torch.zeros(BATCH_SIZE * NODE_NUM, 1)
        # q_eval w.r.t the action in experience
        start = 0
        end = NODE_NUM
        for i in range(0, BATCH_SIZE):
            ba = b_a[i, :].reshape(-1, 1)
            q_eval = self.eval_net(b_s[i, :, :], b_f[i, :, :]).gather(1, ba)  # shape (batch, 1)
            q_next = self.target_net(b_s_[i, :, :], b_f[i, :, :]).detach()  # detach from graph, don't backpropagate
            q_target = b_r[i] + GAMMA * q_next.max(1)[0]  # shape (batch, 1)
            b_q_eval[start:end] = q_eval
            b_q_target[start:end] = q_target.reshape(-1, 1)

            start = end
            end = end + NODE_NUM

        loss = self.loss_func(b_q_eval, b_q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def create_graph(adj):
    G = nx.Graph()
    tmp = adj.numpy()

    for i in range(0, NODE_NUM):
        G.add_node(i)

    for i in range(0, NODE_NUM):
        idx = np.where(tmp[i, :] == 1)
        sq_idx = idx[0]
        for j in range(0, len(sq_idx)):
            G.add_edge(i, sq_idx[j])

    return G


def sigmoid(x):
    return torch.from_numpy(np.array(1./(1 + np.exp(2 * (-x)))))

def process_dis(dis_all, group):
    for i in range(len(group)):
        diff_abs = np.abs(group - group[i])
        idx = np.where(diff_abs > 1)[0]
        dis_all[i, idx] = 100
    
    return dis_all
    
if __name__ == '__main__':
    pos_uav = torch.load('uav_positions/uav_train_att_111')
    uav_n = pos_uav.size()[1]
    gip = int(uav_n / 3)
    
    uav_group = np.zeros(uav_n)
    uav_group[0:gip] = 3
    uav_group[gip:(2 * gip)] = 2
    uav_group[(2 * gip):uav_n] = 1

    dqn = DQN(INPUT_DIM)
    epochs = 100
    epr_v = np.zeros(epochs)
    diff_v = np.zeros(epochs)
    edges_v = np.zeros(epochs)
    
    # a = np.zeros(33).astype(int) + 5
   
    # print('\nCollecting experience...')
    for i_episode in range(epochs):
        
        print(i_episode)

        A = np.zeros([uav_n, uav_n])
        s_diff = 0
        s_edges = 0

        ep_r = 0
        for j in range(100):
            uav_dis = squareform(pdist(pos_uav[j, :, :].numpy()))
            dis = process_dis(uav_dis, uav_group)
            dis_i = env_swarm.get_uav_input(dis, R)

            nor_dis_i = dis_i / DIS_MAX
            t_A = torch.from_numpy(A)
            # a = dqn.choose_action(t_A.to(device), nor_dis_i.to(device))
            a = dqn.choose_action(t_A, nor_dis_i)

            # A_next = env_swarm.step(a + 1, R, dis)
            A_next = env_swarm.step(a, R, dis)
            reward, diff, edges= env_swarm.reward(A_next)
            s_diff = s_diff + diff
            s_edges = s_edges + edges

            # sig_reward = sigmoid(reward)
            sig_reward = reward

            dqn.store_transition(t_A, a, sig_reward, torch.from_numpy(A_next), nor_dis_i)

            ep_r += sig_reward
            A = A_next.copy()

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        epr_v[i_episode] = ep_r
        diff_v[i_episode] = s_diff
        edges_v[i_episode] = s_edges
        

    plt.plot(epr_v)
    plt.title('reward')
    plt.show()

    plt.plot(diff_v)
    plt.title('diff')
    plt.show()

    plt.plot(edges_v)
    plt.title('edges')
    plt.show()
    
    plt.close()

    torch.save(dqn, 'dqn_model/tmp/dqn_111_200_0')
    np.savetxt('dqn_model/epr_111_200_0', epr_v)
