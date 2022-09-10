from starcraft2_rev3 import StarCraft2Env
from GDN_rev3 import Agent
import numpy as np
import torch
import pickle
import gc
import pandas as pd
from pysc2.lib.remote_controller import ConnectError, RequestError
from pysc2.lib.protocol import ProtocolError

from functools import partial
import sys
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from functools import partial
import sys
import os
import vessl

vessl.init()


def env_fn(env, **kwargs):
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))






regularizer = 0.0
map_name1 = '3s5z_vs_3s6z'


"""
Protoss
colossi : 200.0150.01.0
stalkers : 80.080.00.625
zealots : 100.050.00.5

Terran
medivacs  : 150.00.00.75
marauders : 125.00.00.5625
marines   : 45.00.00.375

Zerg
zergling : 35.00.00.375
hydralisk : 80.00.00.625
baneling : 30.00.00.375
spine crawler : 300.00.01.125
"""

def evaluation(env, agent, num_eval):
    max_episode_len = env.episode_limit
    t = 0
    win_rates = 0
    for e in range(num_eval):
        env.reset()
        done = False
        episode_reward = 0
        step = 0
        while (not done) and (step < max_episode_len):
            step += 1
            node_feature = env.get_graph_feature()
            edge_index_enemy = env.get_enemy_visibility_edge_index()
            edge_index_ally = env.get_ally_visibility_edge_index()
            n_node_features = torch.tensor(node_feature).shape[0]
            node_representation = agent.get_node_representation(node_feature, edge_index_enemy, edge_index_ally,
                                                                n_node_features,
                                                                mini_batch=False)  # 차원 : n_agents X n_representation_comm
            avail_action = env.get_avail_actions()
            action_feature = env.get_action_feature()  # 차원 : action_size X n_action_feature
            action = agent.sample_action(node_representation, action_feature, avail_action, epsilon=0)
            reward, done, info = env.step(action)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += reward
            t += 1

        print("map name {} : Evaluation episode {}, episode reward {}, win_tag {}".format(env.map_name, e, episode_reward, win_tag))
        if win_tag == True:
            win_rates += 1 / num_eval
    print("map name : ", env.map_name, "승률", win_rates)
    return win_rates

def network_sharing(agent_group):
    agent_ref = agent_group[0]
    for agent in agent_group[1:]:
        agent.VDN = agent_ref.VDN
        agent.VDN_target = agent_ref.VDN_target
        agent.Q = agent_ref.Q
        agent.Q_tar = agent_ref.Q_tar
        agent.node_representation_enemy_obs = agent_ref.node_representation_enemy_obs
        agent.node_representation = agent_ref.node_representation
        agent.action_representation = agent_ref.action_representation
        agent.optimizer = agent_ref.optimizer



def get_agent_type_of_envs(envs):
    agent_type_ids = list()
    type_alliance = list()
    for env in envs:
        for agent_id, _ in env.agents.items():
            agent = env.get_unit_by_id(agent_id)
            agent_type_ids.append(str(agent.health_max)+str(agent.shield_max)+str(agent.radius))
            type_alliance.append([str(agent.health_max)+str(agent.shield_max)+str(agent.radius), agent.alliance])
        for e_id, e_unit in env.enemies.items():
            enemy = list(env.enemies.items())[e_id][1]
            agent_type_ids.append(str(enemy.health_max)+str(enemy.shield_max)+str(enemy.radius))
            type_alliance.append([str(enemy.health_max)+str(enemy.shield_max)+str(enemy.radius), enemy.alliance])
    agent_types_list = list(set(agent_type_ids))
    type_alliance_set = list()
    for x in type_alliance:
        if x not in type_alliance_set:
            type_alliance_set.append(x)
    print(type_alliance_set)
    for id in agent_types_list:
        print("id : ", id, "count : " , agent_type_ids.count(id))

    return len(agent_types_list), agent_types_list



def train(agent, env, e, t, train_start, epsilon, min_epsilon, anneal_epsilon):
    max_episode_limit = env.episode_limit
    env.reset()
    done = False
    episode_reward = 0
    step = 0
    losses = []
    eval = False
    while (not done) and (step < max_episode_limit):
        node_feature = env.get_graph_feature()
        edge_index_enemy = env.get_enemy_visibility_edge_index()
        edge_index_ally = env.get_ally_visibility_edge_index()
        n_node_features = torch.tensor(node_feature).shape[0]
        node_representation = agent.get_node_representation(node_feature, edge_index_enemy, edge_index_ally,
                                                            n_node_features,
                                                            mini_batch=False)  # 차원 : n_agents X n_representation_comm
        avail_action = env.get_avail_actions()
        action_feature = env.get_action_feature()  # 차원 : action_size X n_action_feature

        action = agent.sample_action(node_representation, action_feature, avail_action, epsilon)
        reward, done, info = env.step(action)
        agent.buffer.memory(node_feature, action, action_feature, edge_index_enemy, edge_index_ally, reward,
                            done, avail_action)
        episode_reward += reward
        t += 1
        step += 1
        if (t % 5000 == 0) and (t >0):
            eval = True

        if e >= train_start:
            loss = agent.learn(regularizer)
            losses.append(loss.detach().item())
        if epsilon >= min_epsilon:
            epsilon = epsilon - anneal_epsilon
        else:
            epsilon = min_epsilon


    if e >= train_start:

        print("{} Total reward in episode {} = {}, loss : {}, epsilon : {}, time_step : {}".format(env.map_name,
                                                                                                e,
                                                                                                episode_reward,
                                                                                                loss,
                                                                                                epsilon,
                                                                                                t))


    return episode_reward, epsilon, t, eval

def main():
    
    env1 = REGISTRY["sc2"](map_name=map_name1, seed=123, step_mul = 8)
    
    env1.reset()
    num_unit_types, unit_type_ids = get_agent_type_of_envs([env1])
    env1.generate_num_unit_types(num_unit_types, unit_type_ids)


    hidden_size_obs = 32
    hidden_size_comm = 42
    hidden_size_Q = 64
    n_representation_obs = 42
    n_representation_comm = 64
    buffer_size = 150000
    batch_size = 32
    gamma = 0.99
    learning_rate = 2e-4
    n_multi_head = 1
    dropout = 0.6
    num_episode = 1000000

    train_start = 10
    epsilon = 1
    min_epsilon = 0.05
    anneal_steps = 50000
    anneal_epsilon = (epsilon - min_epsilon) / anneal_steps


    agent1 = Agent(num_agent=env1.get_env_info()["n_agents"],
                  feature_size=env1.get_env_info()["node_features"],
                  hidden_size_obs=hidden_size_obs,
                  hidden_size_comm=hidden_size_comm,
                  hidden_size_Q=hidden_size_Q,
                  n_multi_head=n_multi_head,
                  n_representation_obs=n_representation_obs,
                  n_representation_comm=n_representation_comm,
                  dropout=dropout,
                  action_size=env1.get_env_info()["n_actions"],
                  buffer_size=buffer_size,
                  batch_size=batch_size,
                  max_episode_len=env1.episode_limit,
                  learning_rate=learning_rate,
                  gamma=gamma)

    # agent2 = Agent(num_agent=env2.get_env_info()["n_agents"],
    #               feature_size=env2.get_env_info()["node_features"],
    #               hidden_size_obs=hidden_size_obs,
    #               hidden_size_comm=hidden_size_comm,
    #               hidden_size_Q=hidden_size_Q,
    #               n_multi_head=n_multi_head,
    #               n_representation_obs=n_representation_obs,
    #               n_representation_comm=n_representation_comm,
    #               dropout=dropout,
    #               action_size=env2.get_env_info()["n_actions"],
    #               buffer_size=buffer_size,
    #               batch_size=batch_size,
    #               max_episode_len=env2.episode_limit,
    #               learning_rate=learning_rate,
    #               gamma=gamma)






    #network_sharing([agent1])
    t = 0
    
    
    epi_r = []
    for e in range(num_episode):
        episode_reward, epsilon, t, eval = train(agent1, env1, e, t, train_start, epsilon, min_epsilon, anneal_epsilon)
        epi_r.append(episode_reward)
        if e % 100 == 1:
            vessl.log(step=e, payload={'reward': np.mean(epi_r)})
            epi_r = []
            
        if eval == True:
            win_rate = evaluation(env1, agent1, 32)
            vessl.log(step=t, payload={'win_rate': win_rate})








main()

