import numpy as np


class SumTree(object):

    def __init__(self, capacity):
        self.write = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p

        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def global_update(self):
        for idx in range(2 * self.capacity - 2, 0, -2):
            self.tree[(idx - 1) // 2] = self.tree[idx] + self.tree[idx - 1]

    def get(self, s):
        idx = 0

        while True:
            left = 2 * idx + 1
            right = left + 1

            if left >= len(self.tree):
                break

            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
