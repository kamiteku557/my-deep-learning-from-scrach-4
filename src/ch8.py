from typing import Deque, Iterable
import random
from dataclasses import dataclass
from collections import deque

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtyping import TensorType


@dataclass
class RecordOfBuffer:
    __slots__ = ("state", "action", "reward", "next_state", "done")
    state: npt.NDArray
    action: int
    reward: float
    next_state: npt.NDArray
    done: bool


@dataclass
class Batch:
    __slots__ = ("state", "action", "reward", "next_state", "done")
    state: TensorType["batch", "state_size"]
    action: TensorType["batch"]
    reward: TensorType["batch"]
    next_state: TensorType["batch"]
    done: TensorType["batch"]

    @classmethod
    def from_records(cls, records: Iterable[RecordOfBuffer]) -> "Batch":
        states = torch.tensor(np.stack([x.state for x in records]))
        actions = torch.tensor(np.stack([x.action for x in records]).astype(np.int64))
        rewards = torch.tensor(np.stack([x.reward for x in records]).astype(np.float32))
        next_states = torch.tensor(np.stack([x.next_state for x in records]))
        dones = torch.tensor(np.stack([x.done for x in records]).astype(np.int32))
        return cls(
            state=states,
            action=actions,
            reward=rewards,
            next_state=next_states,
            done=dones,
        )


class ReplayBuffer:
    """経験再生"""

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer: Deque[RecordOfBuffer] = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(
        self,
        state: npt.NDArray,
        action: int,
        reward: float,
        next_state: npt.NDArray,
        done: bool,
    ):
        data = RecordOfBuffer(state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self) -> int:
        return len(self.buffer)

    def get_batch(self) -> Batch:
        data = random.sample(self.buffer, self.batch_size)
        return Batch.from_records(data)


class QNet(nn.Module):
    def __init__(self, action_size: int):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x: TensorType["batch", 4]) -> TensorType["batch", "action_size"]:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self, seed: int = 0):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        # numpy seed
        self.rng = np.random.default_rng(seed=seed)

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def get_action(self, state: npt.NDArray) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.action_size)

        # add dimention of batch
        state = torch.tensor(state[np.newaxis, :])  # type:ignore
        qs = self.qnet(state)
        return qs.argmax().item()

    def update(
        self,
        state: npt.NDArray,
        action: int,
        reward: float,
        next_state: npt.NDArray,
        done: bool,
    ):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.get_batch()
        Qs = self.qnet(batch.state)
        Q = Qs[np.arange(self.batch_size), batch.action]

        next_Qs = self.qnet_target(batch.next_state)
        next_Q = next_Qs.max(1)[0]

        next_Q.detach()

        target = batch.reward + (1 - batch.done) * self.gamma * next_Q
        loss_fn = nn.MSELoss()

        loss = loss_fn(Q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
