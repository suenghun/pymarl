from smac.env import StarCraft2Env
from VDN_CORR import Agent
import numpy as np
import torch
import pickle
import pandas as pd
from pysc2.lib.remote_controller import ConnectError, RequestError
from pysc2.lib.protocol import ProtocolError
from envs import REGISTRY as env_REGISTRY





# lr 5e -4, max_grad 10, 층하나 더많듬
regularizer = 0.8
map_name = '27m_vs_30m'

reward_save_path = 'reward_{}_regularizer_{}_not_one_by_n.csv'.format(map_name, regularizer)
win_rate_save_path = 'win_rate_{}_regularizer_{}_not_one_by_n.csv'.format(map_name, regularizer)




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



def main():
    try:
        torch.manual_seed(123)
        #env = StarCraft2Env(map_name=map_name, step_mul=8)
        env = REGISTRY["sc2"](map_name = map_name, seed = 123)
        env_info = env.get_env_info()
        state_size = env_info["state_shape"]
        action_size = env_info["n_actions"]
        num_agent = env_info["n_agents"]
        obs_size = env_info["obs_shape"] + action_size + num_agent
        print(env_info["obs_shape"], action_size, num_agent)
        hidden_size = 64
        max_episode_len = env.episode_limit
        buffer_size = int(5e3)
        batch_size = 32
        gamma = 0.99
        epsilon = 1
        min_epsilon = 0.05
        anneal_steps = 50000

        anneal_epsilon = (epsilon - min_epsilon) / anneal_steps

        one_hot_actions = np.eye(action_size).tolist()
        init_last_actions = [0] * action_size

        agent = Agent(num_agent, obs_size, state_size, hidden_size, action_size, buffer_size, batch_size, max_episode_len,
                      gamma)
        t = 0
        n_episodes = 1000000
        for e in range(n_episodes):
            env.reset()
            done = False
            episode_reward = 0
            step = 0
            agent.initialize_episode()
            padding = 1
            hidden = [torch.zeros(1, hidden_size) for _ in range(num_agent)]
            obs = env.get_obs()
            last_action = [init_last_actions] * num_agent

            state = env.get_state()
            avail_action = env.get_avail_actions()
            eval = False

            while (not done) and (step < max_episode_len):

                temp_hidden = list()
                action = list()
                for n in range(num_agent):
                    u, h = agent.sample_action(obs[n], hidden[n], avail_action[n], epsilon, last_action[n], n)
                    action.append(u)
                    temp_hidden.append(h)

                hidden = temp_hidden

                reward, done, info = env.step(action)

                obs_next = env.get_obs()
                state_next = env.get_state()
                avail_action_next = env.get_avail_actions()

                agent.memory(obs, action, reward, avail_action, done, padding, last_action)

                last_action = [one_hot_actions[int(a)] for a in action]
                episode_reward += reward

                step += 1
                t += 1
                obs = obs_next
                state = state_next
                avail_action = avail_action_next

                if epsilon >= min_epsilon:
                    epsilon = epsilon - anneal_epsilon
                else:
                    epsilon = min_epsilon

                if t % 5000 == 0 and t > 0:
                    eval = True

            done = True
            padding = 0
            for _ in range(step, max_episode_len):
                agent.memory(obs, action, reward, avail_action, done, padding, last_action)

            loss = agent.learn(e, variance = True, regularizer = regularizer)
            print("Total reward in episode {} = {}, loss : {}, epsilon : {}, time_step : {}".format(e, episode_reward,
                                                                                                    loss.item(),
                                                                                                    epsilon, t))




#             if e % 200 == 0 and e > 0:

                # with open('agent_4d.pkl', 'wb') as f:
                #     pickle.dump(agent, f, protocol=pickle.HIGHEST_PROTOCOL)
            if t>= 10000000:
                eval_num = 1000
                win_rate = 0
                eval = False
                for k in range(eval_num):
                    env.reset()
                    d = False
                    eval_reward = 0
                    s = 0
                    h_ = [torch.zeros(1, hidden_size) for _ in range(num_agent)]
                    last_act = [init_last_actions] * num_agent
                    eps = 0
                    win_tag = False
                    while (not d) and (s < max_episode_len):
                        o = env.get_obs()
                        a_a = env.get_avail_actions()
                        temp_h = list()
                        act = list()
                        for n in range(num_agent):
                            u, h = agent.sample_action(o[n], h_[n], a_a[n], eps, last_act[n], n)
                            act.append(u)
                            temp_h.append(h)
                        h_ = temp_h
                        last_act = [one_hot_actions[int(a)] for a in act]
                        r, d, inf = env.step(act)
                        win_tag = True if d and 'battle_won' in inf and inf['battle_won'] else False
                        eval_reward += r
                        s += 1

                    print(eval_reward, win_tag)
                    if win_tag == True:
                        win_rate += 1 / 1000 * 100
                print("총 평가 승률 : {} 퍼센트".format(win_rate))
                vessl.log(step = k, payload = {'win_rate' : win_rate})
            if eval == True:
                eval_num = 32
                win_rate = 0
                eval = False
                for _ in range(eval_num):
                    env.reset()
                    d = False
                    eval_reward = 0
                    s = 0
                    h_ = [torch.zeros(1, hidden_size) for _ in range(num_agent)]

                    last_act = [init_last_actions] * num_agent

                    eps = 0
                    win_tag = False
                    while (not d) and (s < max_episode_len):
                        o = env.get_obs()
                        a_a = env.get_avail_actions()

                        temp_h = list()
                        act = list()

                        for n in range(num_agent):
                            u, h = agent.sample_action(o[n], h_[n], a_a[n], eps, last_act[n], n)
                            act.append(u)
                            temp_h.append(h)

                        h_ = temp_h

                        last_act = [one_hot_actions[int(a)] for a in act]
                        r, d, inf = env.step(act)
                        win_tag = True if d and 'battle_won' in inf and inf['battle_won'] else False

                        eval_reward += r
                        s += 1

                    print(eval_reward, win_tag)
                    if win_tag == True:
                        win_rate += 1 / 32 * 100
                print("평가 승률 : {} 퍼센트".format(win_rate))
                vessl.log(step = t, payload = {'win_rate' : win_rate})

                



    except RequestError or ProtocolError or ConnectError:
        env.close()
        agent.buffer.episode_indices.pop()
        agent.buffer.episode_idx -=1
        agent.buffer.buffer.pop()
        return agent, epsilon, t, e


def main2(agent, epsilon, t, ep):
    try:
        torch.manual_seed(123)
        env = REGISTRY["sc2"](map_name = map_name, seed = 123)
        env_info = env.get_env_info()
        action_size = env_info["n_actions"]

        num_agent = env_info["n_agents"]
        print("재시작 수행")
        hidden_size = 64
        max_episode_len = env.episode_limit
        min_epsilon = 0.05
        anneal_steps = 50000

        anneal_epsilon = (epsilon - min_epsilon) / anneal_steps

        one_hot_actions = np.eye(action_size).tolist()
        init_last_actions = [0] * action_size

        n_episodes = 1000000

        for e in range(ep, n_episodes):
            env.reset()
            done = False
            episode_reward = 0
            step = 0
            agent.initialize_episode()
            padding = 1
            hidden = [torch.zeros(1, hidden_size) for _ in range(num_agent)]
            obs = env.get_obs()
            last_action = [init_last_actions] * num_agent

            state = env.get_state()
            
            avail_action = env.get_avail_actions()
            eval = False

            while (not done) and (step < max_episode_len):

                temp_hidden = list()
                action = list()

                for n in range(num_agent):
                    u, h = agent.sample_action(obs[n], hidden[n], avail_action[n], epsilon, last_action[n], n)
                    action.append(u)
                    temp_hidden.append(h)

                hidden = temp_hidden

                reward, done, info = env.step(action)

                obs_next = env.get_obs()
                state_next = env.get_state()
                avail_action_next = env.get_avail_actions()

                agent.memory(obs, action, reward, avail_action, done, padding, last_action)

                last_action = [one_hot_actions[int(a)] for a in action]
                episode_reward += reward

                step += 1
                t += 1
                obs = obs_next
                state = state_next
                avail_action = avail_action_next

                if epsilon >= min_epsilon:
                    epsilon = epsilon - anneal_epsilon
                else:
                    epsilon = min_epsilon

                if t % 5000 == 0 and t > 0:
                    eval = True

            done = True
            padding = 0
            for _ in range(step, max_episode_len):
                agent.memory(obs, action, reward, avail_action, done, padding, last_action)

            loss = agent.learn(e, variance = True, regularizer = regularizer)
            print("Total reward in episode {} = {}, loss : {}, epsilon : {}, time_step : {}".format(e, episode_reward,
                                                                                                    loss.item(),
                                                                                                    epsilon, t))

            if t>= 10000000:
                eval_num = 1000
                win_rate = 0
                eval = False
                for k in range(eval_num):
                    env.reset()
                    d = False
                    eval_reward = 0
                    s = 0
                    h_ = [torch.zeros(1, hidden_size) for _ in range(num_agent)]
                    last_act = [init_last_actions] * num_agent
                    eps = 0
                    win_tag = False
                    while (not d) and (s < max_episode_len):
                        o = env.get_obs()
                        a_a = env.get_avail_actions()
                        temp_h = list()
                        act = list()
                        for n in range(num_agent):
                            u, h = agent.sample_action(o[n], h_[n], a_a[n], eps, last_act[n], n)
                            act.append(u)
                            temp_h.append(h)
                        h_ = temp_h
                        last_act = [one_hot_actions[int(a)] for a in act]
                        r, d, inf = env.step(act)
                        win_tag = True if d and 'battle_won' in inf and inf['battle_won'] else False
                        eval_reward += r
                        s += 1

                    print(eval_reward, win_tag)
                    if win_tag == True:
                        win_rate += 1 / 1000 * 100
                print("총 평가 승률 : {} 퍼센트".format(win_rate))
                vessl.log(step = k, payload = {'win_rate' : win_rate})
               

              
              
            if eval == True:
                eval_num = 32
                win_rate = 0
                eval = False
                for _ in range(eval_num):
                    env.reset()
                    d = False
                    eval_reward = 0
                    s = 0
                    h_ = [torch.zeros(1, hidden_size) for _ in range(num_agent)]
                    last_act = [init_last_actions] * num_agent
                    eps = 0
                    win_tag = False
                    while (not d) and (s < max_episode_len):
                        o = env.get_obs()
                        a_a = env.get_avail_actions()
                        temp_h = list()
                        act = list()
                        for n in range(num_agent):
                            u, h = agent.sample_action(o[n], h_[n], a_a[n], eps, last_act[n], n)
                            act.append(u)
                            temp_h.append(h)
                        h_ = temp_h
                        last_act = [one_hot_actions[int(a)] for a in act]
                        r, d, inf = env.step(act)
                        win_tag = True if d and 'battle_won' in inf and inf['battle_won'] else False
                        eval_reward += r
                        s += 1

                    print(eval_reward, win_tag)
                    if win_tag == True:
                        win_rate += 1 / 32 * 100
                print("평가 승률 : {} 퍼센트".format(win_rate))
                vessl.log(step = t, payload = {'win_rate' : win_rate})

                




    except RequestError or ProtocolError or ConnectError:
        env.close()
        agent.buffer.episode_indices.pop()
        agent.buffer.episode_idx -=1
        agent.buffer.buffer.pop()
        return agent, epsilon, t, e
    



agent, epsilon, t, e= main()
while True:
    agent, epsilon, t, e= main2(agent, epsilon, t, e)
