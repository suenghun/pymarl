import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from torch.distributions import Categorical
import numpy as np

from model import GAT
from layers import device




class VDN(nn.Module):

    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_local):
        return torch.sum(q_local, dim = 1)

class Network(nn.Module):
    def __init__(self, obs_and_action_size, hidden_size_q):
        super(Network, self).__init__()
        self.obs_and_action_size = obs_and_action_size
        self.fcn_1 = nn.Linear(obs_and_action_size, hidden_size_q)
        self.fcn_2 = nn.Linear(hidden_size_q, int(hidden_size_q/2))
        self.fcn_3 = nn.Linear(int(hidden_size_q/2), int(hidden_size_q/4))
        self.fcn_4 = nn.Linear(int(hidden_size_q/4), int(hidden_size_q/8))
        self.fcn_5 = nn.Linear(int(hidden_size_q/8), 1)

        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)
        torch.nn.init.xavier_uniform_(self.fcn_4.weight)
        torch.nn.init.xavier_uniform_(self.fcn_5.weight)

    def forward(self, obs_and_action):
        x = F.relu(self.fcn_1(obs_and_action))
        x = F.relu(self.fcn_2(x))
        x = F.relu(self.fcn_3(x))
        x = F.relu(self.fcn_4(x))
        q = self.fcn_5(x)
        return q

class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, hidden_size, n_representation_obs):

        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.fcn_1 = nn.Linear(feature_size, hidden_size)
        self.fcn_2 = nn.Linear(hidden_size, hidden_size)
        self.fcn_3 = nn.Linear(hidden_size, n_representation_obs)

        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)

    def forward(self, node_feature):
        x = F.relu(self.fcn_1(node_feature))
        x = F.relu(self.fcn_2(x))
        node_representation = self.fcn_3(x)
        return node_representation

class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, num_agent, action_size):
        self.buffer = deque()


        self.step_count_list = list()
        for _ in range(8):
            self.buffer.append(deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.num_agent = num_agent
        self.agent_id = np.eye(self.num_agent).tolist()
        self.one_hot_actions = np.eye(action_size).tolist()
        self.batch_size = batch_size
        self.step_count = 0


    def pop(self):
        self.buffer.pop()

    def memory(self, node_feature, action, action_feature, edge_index_enemy, edge_index_ally, reward, done, avail_action):
        self.buffer[0].append(node_feature)
        self.buffer[1].append(action)
        self.buffer[2].append(action_feature)
        self.buffer[3].append(edge_index_enemy)
        self.buffer[4].append(edge_index_ally)
        self.buffer[5].append(reward)
        self.buffer[6].append(done)
        self.buffer[7].append(avail_action)

        if self.step_count < self.buffer_size - 1:
            self.step_count_list.append(self.step_count)
            self.step_count += 1





    def generating_mini_batch(self, datas, batch_idx, cat):

        for s in batch_idx:
            if cat == 'node_feature':
                yield datas[0][s]
            if cat == 'action':
                yield datas[1][s]
            if cat == 'action_feature':
                yield datas[2][s]
            if cat == 'edge_index_enemy':
                yield datas[3][s]
            if cat == 'edge_index_ally':
                yield datas[4][s]
            if cat == 'reward':
                yield datas[5][s]
            if cat == 'done':
                yield datas[6][s]
            if cat == 'node_feature_next':
                yield datas[0][s+1]
            if cat == 'action_feature_next':
                yield datas[2][s+1]
            if cat == 'edge_index_enemy_next':
                yield datas[3][s+1]
            if cat == 'edge_index_ally_next':
                yield datas[4][s+1]
            if cat == 'avail_action_next':
                yield datas[7][s+1]



    def sample(self):
        step_count_list = self.step_count_list[:]
        step_count_list.pop()

        sampled_batch_idx = random.sample(step_count_list, self.batch_size)


        node_feature = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature')
        node_features = list(node_feature)

        action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action')
        actions = list(action)


        action_feature = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action_feature')
        action_features = list(action_feature)

        edge_index_enemy = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_enemy')
        edge_indices_enemy = list(edge_index_enemy)

        edge_index_ally = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_ally')
        edge_indices_ally = list(edge_index_ally)

        reward = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='reward')
        rewards = list(reward)

        done = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='done')
        dones = list(done)

        node_feature_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_next')
        node_features_next = list(node_feature_next)

        action_feature_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action_feature_next')
        action_features_next = list(action_feature_next)

        edge_index_enemy_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_enemy_next')
        edge_indices_enemy_next = list(edge_index_enemy_next)

        edge_index_ally_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_ally_next')
        edge_indices_ally_next = list(edge_index_ally_next)



        avail_action_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action_next')
        avail_actions_next = list(avail_action_next)


        return node_features, actions, action_features, edge_indices_enemy, edge_indices_ally, rewards, dones, node_features_next, action_features_next, edge_indices_enemy_next, edge_indices_ally_next, avail_actions_next

class Agent:
    def __init__(self,
                 num_agent,

                 feature_size,

                 hidden_size_obs,
                 hidden_size_comm,
                 hidden_size_Q,

                 n_multi_head,

                 n_representation_obs,
                 n_representation_comm,

                 max_episode_len,
                 dropout,
                 action_size,
                 buffer_size,
                 batch_size,
                 learning_rate,
                 gamma):
        torch.manual_seed(40)
        np.random.seed(40)
        random.seed(40)
        self.num_agent = num_agent
        self.feature_size = feature_size

        self.hidden_size_obs = hidden_size_obs
        self.hidden_size_comm = hidden_size_comm
        self.n_multi_head = n_multi_head

        self.n_representation_obs = n_representation_obs
        self.n_representation_comm = n_representation_comm
        self.max_episode_len = max_episode_len

        self.action_size = action_size

        self.dropout = dropout
        self.gamma = gamma
        self.agent_id = np.eye(self.num_agent).tolist()

        self.max_norm = 10
        self.VDN = VDN().to(device)
        self.VDN_target = VDN().to(device)
        self.Q = Network(n_representation_comm+n_representation_obs, hidden_size_Q).to(device)
        self.Q_tar = Network(n_representation_comm+n_representation_obs, hidden_size_Q).to(device)

        self.Q_tar.load_state_dict(self.Q.state_dict())
        self.VDN_target.load_state_dict(self.VDN.state_dict())



        self.buffer = Replay_Buffer(buffer_size, batch_size, num_agent, action_size)
        self.batch_size = batch_size


        self.action_space = [i for i in range(self.action_size)]

        self.node_representation_enemy_obs = NodeEmbedding(feature_size=feature_size, hidden_size=hidden_size_obs, n_representation_obs=n_representation_obs).to(device)  # 수정사항



        self.node_representation = NodeEmbedding(feature_size = feature_size, hidden_size = hidden_size_obs, n_representation_obs = n_representation_obs).to(device) # 수정사항
        self.action_representation = NodeEmbedding(feature_size = feature_size + 6, hidden_size = hidden_size_obs, n_representation_obs = n_representation_obs).to(device) # 수정사항

        self.func_enemy_obs = GAT(nfeat = n_representation_obs,
                                  nhid = hidden_size_obs,
                                  nheads = n_multi_head,
                                  nclass = n_representation_obs,
                                  dropout = dropout,
                                  alpha = 0.2 ).to(device)
        self.func_ally_comm = GAT(nfeat = 2 * n_representation_obs,
                                  nhid = hidden_size_comm,
                                  nheads = n_multi_head,
                                  nclass = n_representation_comm,
                                  dropout = dropout,
                                  alpha = 0.2).to(device)   # 수정사항

        self.eval_params = list(self.VDN.parameters()) + \
                           list(self.Q.parameters()) + \
                           list(self.node_representation_enemy_obs.parameters()) + \
                           list(self.node_representation.parameters()) + \
                           list(self.action_representation.parameters())+\
                           list(self.func_enemy_obs.parameters()) +\
                           list(self.func_ally_comm.parameters())

        self.optimizer = optim.RMSprop(self.eval_params, lr=learning_rate)





    def get_node_representation(self, node_feature, edge_index_enemy, edge_index_ally, n_node_features, mini_batch = False):
        if mini_batch == False:
            with torch.no_grad():
                node_feature = torch.tensor(node_feature, dtype=torch.float,device=device)

                node_embedding_enemy_obs = self.node_representation_enemy_obs(node_feature)

                edge_index_enemy = torch.tensor(edge_index_enemy, dtype=torch.long, device=device)
                node_embedding = self.node_representation(node_feature)
                h_enemy_obs = self.func_enemy_obs(node_embedding_enemy_obs, edge_index_enemy, n_node_features, mini_batch = mini_batch)
                edge_index_ally = torch.tensor(edge_index_ally, dtype=torch.long, device=device)
                cat_feature = torch.concat([node_embedding, h_enemy_obs], dim=1)
                node_representation = self.func_ally_comm(cat_feature, edge_index_ally, n_node_features, mini_batch = mini_batch)
        else:
            node_feature = torch.tensor(node_feature, dtype=torch.float,device=device)
            node_embedding_enemy_obs = self.node_representation_enemy_obs(node_feature)
            h_enemy_obs = self.func_enemy_obs(node_embedding_enemy_obs, edge_index_enemy, n_node_features, mini_batch = mini_batch)
            # h_enemy_obs.shape : (batch_size, node_size, feature_size_2)
            # node_feature : (batch_size, node_size, feature_size_1)
            node_embedding = self.node_representation(node_feature)
            cat_feature = torch.concat([node_embedding, h_enemy_obs], dim=2)
            node_representation = self.func_ally_comm(cat_feature, edge_index_ally, n_node_features, mini_batch = mini_batch)
        return node_representation




    def cal_Q(self, obs, actions, action_features, avail_actions_next, agent_id, target = False):
        if target == False:
            obs_n = obs[:, agent_id].unsqueeze(1).expand([self.batch_size, self.action_size, self.n_representation_comm])
            action_features = torch.tensor(action_features, device = device)
            action_embedding = self.action_representation(action_features)
            obs_and_action = torch.concat([obs_n, action_embedding], dim=2)
            obs_and_action = obs_and_action.float()
            q = self.Q(obs_and_action)
            q = q.squeeze(2)                                  # q.shape :      (batch_size, action_size)
            actions = torch.tensor(actions, device = device).long()
            act_n = actions[:, agent_id].unsqueeze(1)  # action.shape : (batch_size, 1)
            #print(q.shape, act_n.shape)
            q = torch.gather(q, 1, act_n).squeeze(1)                     # q.shape :      (batch_size, 1)
            return q
        else:
            with torch.no_grad():
                obs_next = obs
                action_features_next = action_features
                obs_next = obs_next[:, agent_id].unsqueeze(1).expand([self.batch_size, self.action_size, self.n_representation_comm])
                action_features_next = torch.tensor(action_features_next, device = device)
                action_embedding_next = self.action_representation(action_features_next)
                obs_and_action_next = torch.concat([obs_next, action_embedding_next], dim=2)
                obs_and_action_next = obs_and_action_next.float()
                q_tar = self.Q_tar(obs_and_action_next)                        # q.shape :      (batch_size, action_size, 1)
                q_tar = q_tar.squeeze(2)                                       # q.shape :      (batch_size, action_size)
                avail_actions_next = torch.tensor(avail_actions_next, device = device).bool()
                mask = avail_actions_next[:, agent_id]
                q_tar = q_tar.masked_fill(mask == 0, float('-inf'))
                q_tar_max = torch.max(q_tar, dim = 1)[0]
                return q_tar_max



    @torch.no_grad()
    def sample_action(self, node_representation, action_feature, avail_action, epsilon):
        """
        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """

        mask = torch.tensor(avail_action, device=device).bool()
        action_feature = torch.tensor(action_feature, device=device, dtype = torch.float64).float()
        action_size = action_feature.shape[0]
        action = []
        action_embedding = self.action_representation(action_feature)
        for n in range(self.num_agent):
            obs = node_representation[n].expand(action_size, node_representation[n].shape[0])         # 차원 : action_size X n_representation_comm
            obs_cat_action = torch.concat([obs, action_embedding], dim = 1)    # 차원 : action_size X
            obs_cat_action = obs_cat_action.float()
            Q = self.Q(obs_cat_action).squeeze(1)                                                # 차원 : action_size X 1

            Q = Q.masked_fill(mask[n, :]==0, float('-inf'))
            greedy_u = torch.argmax(Q)

            mask_n = np.array(avail_action[n], dtype=np.float64)

            if np.random.uniform(0, 1) >= epsilon:
                u = greedy_u
                action.append(u)
            else:
                u = np.random.choice(self.action_space, p=mask_n / np.sum(mask_n))
                action.append(u)

        return action






    def learn(self, regularizer):
        node_features, actions, action_features, edge_indices_enemy, edge_indices_ally, rewards, dones, node_features_next, action_features_next, edge_indices_enemy_next, edge_indices_ally_next, avail_actions_next = self.buffer.sample()
        n_node_features = torch.tensor(node_features).shape[1]
        obs = self.get_node_representation(node_features, edge_indices_enemy, edge_indices_ally, n_node_features,
                                            mini_batch=True)
        obs_next = self.get_node_representation(node_features_next, edge_indices_enemy_next, edge_indices_ally_next,
                                                 n_node_features, mini_batch=True)

        dones = torch.tensor(dones, device = device, dtype = torch.float)
        rewards = torch.tensor(rewards, device = device, dtype = torch.float)

        q = [self.cal_Q(obs=obs,
                         actions=actions,
                         action_features=action_features,
                         avail_actions_next=None,
                         agent_id=agent_id,
                         target=False) for agent_id in range(self.num_agent)]

        q_tar = [self.cal_Q(obs=obs_next,
                             actions=None,
                             action_features=action_features_next,
                             avail_actions_next=avail_actions_next,
                             agent_id=agent_id,
                             target=True) for agent_id in range(self.num_agent)]

        q_tot = torch.stack(q, dim=1)
        loss2 = torch.var(q_tot, dim = 1).mean()
        q_tot_tar = torch.stack(q_tar, dim=1)
        q_tot = self.VDN(q_tot)
        q_tot_tar = self.VDN_target(q_tot_tar)

        td_target = rewards*self.num_agent + self.gamma* (1-dones)*q_tot_tar
        loss1 = F.mse_loss(q_tot, td_target.detach())

        loss = loss1 + regularizer * loss2

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.eval_params, 10)
        self.optimizer.step()
        # if episode % 20 == 0 and episode > 0:
        #     self.Q_tar.load_state_dict(self.Q.state_dict())
        #     self.VDN_target.load_state_dict(self.VDN.state_dict())
        tau = 1e-3
        for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        for target_param, local_param in zip(self.VDN_target.parameters(), self.VDN.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        return loss

