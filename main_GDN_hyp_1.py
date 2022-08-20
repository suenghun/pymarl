from starcraft2 import StarCraft2Env
from GDN import Agent
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
import vessl

vessl.init()

def env_fn(env, **kwargs):
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))




regularizer = 0.8
map_name = '3s_vs_5z'




def evaluation(env, agent, num_eval, win_rates_record):

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
            node_representation = agent.get_node_representation(node_feature, edge_index_enemy, edge_index_ally, n_node_features, mini_batch = False) # 차원 : n_agents X n_representation_comm

            avail_action = env.get_avail_actions()
            action_feature = env.get_action_feature()                                                            # 차원 : action_size X n_action_feature

            action = agent.sample_action(node_representation, action_feature, avail_action, epsilon=0)
            reward, done, info = env.step(action)

            episode_reward += reward
            t+=1
            if done == True:
                win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
                print("Evaluation episode {}, episode reward {}, win_tag {}".format(e, episode_reward, win_tag))
                if win_tag == True:
                    win_rates+= 1/num_eval
    print("승률", win_rates)
    win_rates_record.append(win_rates)
    return win_rates




def main():
    try:
        win_rates_record = []
        env = REGISTRY["sc2"](map_name = map_name, seed = 123, num_total_unit_types = 3)
        env_info = env.get_env_info()
        feature_size = env_info["node_features"]
        action_size = env_info["n_actions"]
        num_agent = env_info["n_agents"]
        action_feature_size = 6 + feature_size

        print(env_info["obs_shape"], action_size, num_agent)

        hidden_size_obs = 24
        hidden_size_comm = 36
        n_representation_obs = 36
        n_representation_comm = 48

        max_episode_len = env.episode_limit
        buffer_size = 150000
        batch_size = 32

        gamma = 0.99
        epsilon = 1
        min_epsilon = 0.05
        anneal_steps = 50000
        n_multi_head = 2

        dropout = 0.6

        anneal_epsilon = (epsilon - min_epsilon) / anneal_steps

        one_hot_actions = np.eye(action_size).tolist()
        init_last_actions = [0] * action_size



        agent = Agent(num_agent=num_agent,
                      feature_size=feature_size,
                      hidden_size_obs=hidden_size_obs,
                      hidden_size_comm=hidden_size_comm,
                      n_multi_head=n_multi_head,
                      n_representation_obs=n_representation_obs,
                      n_representation_comm=n_representation_comm,
                      dropout=dropout,
                      action_size=action_size,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      max_episode_len=max_episode_len,
                      gamma=gamma)



        t = 0
        n_episodes = 1000000
        win_rates = []
        epi_r = []
        for e in range(n_episodes):

            env.reset()
            done = False
            episode_reward = 0
            step = 0


            eval = False





            while (not done) and (step < max_episode_len):
                step += 1
                node_feature = env.get_graph_feature()
                edge_index_enemy = env.get_enemy_visibility_edge_index()
                edge_index_ally = env.get_ally_visibility_edge_index()
                n_node_features = torch.tensor(node_feature).shape[0]
                node_representation = agent.get_node_representation(node_feature, edge_index_enemy, edge_index_ally, n_node_features, mini_batch = False) # 차원 : n_agents X n_representation_comm

                avail_action = env.get_avail_actions()
                action_feature = env.get_action_feature()                                                            # 차원 : action_size X n_action_feature

                action = agent.sample_action(node_representation, action_feature, avail_action, epsilon)
                reward, done, info = env.step(action)
                agent.buffer.memory(node_feature, action, action_feature, edge_index_enemy, edge_index_ally, reward, done, avail_action)

                episode_reward += reward
                t+=1
                if t % 5000 == 0:
                    eval = True
                if epsilon >= min_epsilon:
                    epsilon = epsilon - anneal_epsilon
                else:
                    epsilon = min_epsilon

                if t % 5000 == 0 and t > 0:
                    eval = True
            if eval == True:
                win_rate = evaluation(env, agent, 32, win_rates_record)
                vessl.log(step = t, payload = {'win_rate' : win_rate})
                eval = False
            if e >= 10:
                losses = []
                for _ in range(step):
                    loss = agent.learn(e)
                    losses.append(loss.detach().item())

                print("Total reward in episode {} = {}, loss : {}, epsilon : {}, time_step : {}".format(e,
                                                                                                            episode_reward,
                                                                                                            loss,
                                                                                                            epsilon,
                                                                                                            t))

            # done = True
            # padding = 0
            # for _ in range(step, max_episode_len):
            #     agent.memory(obs, action, reward, avail_action, done, padding, last_action)

            # loss = agent.learn(e, variance = True, regularizer = regularizer)




    except RequestError or ProtocolError or ConnectError:
        env.close()
        agent.buffer.episode_indices.pop()
        agent.buffer.episode_idx -=1
        agent.buffer.buffer.pop()




main()

