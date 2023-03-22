import random
from sumtree import SumTree


class Replayer(object):
    def __init__(self, capacity=10000, pr_scale=1):
        self.capacity = capacity
        self.memory = SumTree(self.capacity)
        self.counter = 0
        self.pr_scale = pr_scale
        self.max_pr = 0
        self.e = 5e-6

    def get_priority(self, error):
        return (error + self.e) ** self.pr_scale

    def remember(self, sample, error):
        p = self.get_priority(error)

        self_max = max(self.max_pr, p)
        self.memory.add(self_max, sample)
        self.counter += 1

    def sample(self, n):
        sample_batch = []
        sample_batch_indices = []
        sample_batch_priorities = []

        num_segments = self.memory.total() / n
        for i in range(n):
            left = num_segments * i
            right = num_segments * (i + 1)

            s = random.uniform(left, right)
            idx, pr, data = self.memory.get(s)
            if data == 0:
                continue        # todo: 由于浮点误差无法采集到有效数据的问题如何解决？
            sample_batch.append(data)
            sample_batch_indices.append(idx)
            sample_batch_priorities.append(pr)

        return [sample_batch, sample_batch_indices, sample_batch_priorities]

    def update(self, batch_index, error):
        self.memory.update(batch_index, error)
