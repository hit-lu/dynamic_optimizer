import os
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from replayer import Replayer


class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['state', 'action', 'reward', 'next_state', 'terminated'])
        self.i = 0
        self.counter = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = np.asarray(args, dtype=object)
        self.i = (self.i + 1) % self.capacity
        self.counter = min(self.counter + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.counter, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


class DuelNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.common_net = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU())
        self.advantage_net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_size))
        self.v_net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, s):
        h = self.common_net(s)
        adv = self.advantage_net(h)
        adv = adv - adv.mean(1).unsqueeze(1)
        v = self.v_net(h)
        q = v + adv
        return q


class DuelDQNAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.gamma = 0.99

        self.replayer = DQNReplayer(10000)
        self.memory_counter = 0

        self.eval_net = DuelNet(input_size=env.observation_space.shape[0], output_size=self.action_n)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.trajectory = []
        self.target_net = copy.deepcopy(self.eval_net)

    def reset(self):
        self.trajectory = []
        self.target_net = copy.deepcopy(self.eval_net)

    def decide(self, observation, train=True):
        if train and np.random.rand() < 0.001:
            # epsilon-greedy policy in train mode
            action = np.random.randint(self.action_n)
        else:
            state_tensor = torch.as_tensor(observation, dtype=torch.float).reshape(1, -1)
            q_tensor = self.eval_net(state_tensor)
            action_tensor = torch.argmax(q_tensor)
            action = action_tensor.item()
        return action

    def store(self, observation, action, reward, terminate):
        self.trajectory += [observation, reward, terminate, action]
        if len(self.trajectory) >= 8:
            state, _, _, act, next_state, reward, terminated, _ = self.trajectory[-8:]
            self.replayer.store(state, act, reward, next_state, terminated)

    def learn(self):
        # replay
        states, actions, rewards, next_states, terminate = self.replayer.sample(1024)
        state_tensor = torch.as_tensor(states, dtype=torch.float)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float)
        terminate_tensor = torch.as_tensor(terminate, dtype=torch.float)

        # update value net
        next_eval_q_tensor = self.eval_net(next_state_tensor)
        next_action_tensor = next_eval_q_tensor.argmax(axis=-1)
        next_q_tensor = self.target_net(next_state_tensor)

        next_max_q_tensor = torch.gather(next_q_tensor, 1, next_action_tensor.unsqueeze(1)).squeeze(1)
        target_tensor = reward_tensor + self.gamma * (1. - terminate_tensor) * next_max_q_tensor
        predict_tensor = self.eval_net(state_tensor)

        q_tensor = predict_tensor.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        loss_tensor = self.loss(target_tensor, q_tensor)
        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()

    def save_model(self, file):
        if not os.path.exists(file):
            os.makedirs(file)
        torch.save(self.eval_net.state_dict(), file + '/' + 'eval_net.pkl')
        torch.save(self.target_net.state_dict(), file + '/' + 'target_net.pkl')

    def load_model(self, file) -> bool:
        if not os.path.exists(file):
            return False
        self.eval_net.load_state_dict(torch.load(file + '/' + 'eval_net.pkl'))
        self.target_net.load_state_dict(torch.load(file + '/' + 'target_net.pkl'))
        return True
