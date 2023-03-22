import random
import torch
import argparse
import gym
import pygame
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataloader import load_data
from agent import D3QNAgent, DuelDQNAgent
from functools import wraps
from tqdm import tqdm


def timer_wrapper(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)

        print("function {} running time :  {} s".format(func.__name__, time.time() - start_time))
        return result

    return measure_time


def play_montecarlo(env, agent, seed=None, train=True):
    state, _ = env.reset(seed=seed)
    agent.reset()

    episode_reward, episode_counter = 0., 0
    reward, terminated, truncated = 0., False, False
    while True:
        action = agent.decide(state)
        if train:
            agent.store(state, action, reward, terminated)
            if agent.replayer.counter >= agent.replayer.capacity * 0.95:
                agent.learn()

        if terminated or truncated:
            break
        state, reward, terminated, truncated, _ = env.step(action)

        episode_reward += reward
        episode_counter += 1

    return episode_reward, episode_counter


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='dynamic optimizer implementation')
    parser.add_argument('--filename', default='PCB8 - PCB.txt', type=str, help='load pcb data')
    parser.add_argument('--auto_register', default=1, type=int, help='register the component according the pcb data')

    parser.add_argument('--learning_rate', default=1e-5, type=float, help='')
    parser.add_argument('--gamma', default=1.0, type=float, help='')

    parser.add_argument('--net_update', default=8000, type=int, help='update frequency for target Q network')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--buffer_size', default=100000, type=int, help='')
    parser.add_argument('--episodes', default=1000, type=int, help='')

    params = parser.parse_args()

    # 结果输出显示所有行和列
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    # 加载PCB数据
    # pcb_data, component_data, _ = load_data(params.filename, default_feeder_limit=params.feeder_limit,
    #                                         cp_auto_register=params.auto_register)  # 加载PCB数据

    # 测试环境
    env = gym.make('MountainCar-v0')
    # env = gym.make('MountainCar-v0', render_mode="human")

    agent = DuelDQNAgent(env)

    episode_rewards = []
    with tqdm(total=params.episodes) as pbar:
        pbar.set_description('training process')

        for episode in range(params.episodes):
            episode_reward, episode_counter = play_montecarlo(env, agent, seed=episode, train=True)
            episode_rewards.append(episode_reward)

            if episode_counter != 200:
                print('episode_counter: ', episode + 1, ', reward: ', episode_reward)
                
            if np.mean(episode_rewards[-10:]) > -110:
                break
            pbar.update(1)

    # 测试
    agent.epsilon = 0.  # 取消探索
    test_rewards = [play_montecarlo(env, agent, train=False)[0] for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(test_rewards), len(test_rewards), np.mean(test_rewards)))

    return episode_rewards


if __name__ == '__main__':
    plt.plot(main())
    plt.show()
