
import torch
from dataclasses import dataclass 
from collections import deque
import random

@dataclass
class Transition:
    state: torch.tensor
    action: int
    reward: float
    new_state: torch.tensor
    done: bool 


class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, new_transition:Transition):
        self.buffer.append(new_transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([t.state for t in batch])
        device = states.device
        actions = torch.tensor([t.action for t in batch], device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        new_states = torch.stack([t.new_state for t in batch])
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, new_states, dones