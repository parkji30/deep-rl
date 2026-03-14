
import torch
import numpy as np
from dataclasses import dataclass 
from collections import deque
import random

@dataclass(slots=True)
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

    def sample(self, batch_size, device='cuda'):
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.stack([t.state for t in batch]).to(device=device, dtype=torch.float32) / 255.0
        new_states = torch.stack([t.new_state for t in batch]).to(device=device, dtype=torch.float32) / 255.0
        
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, new_states, dones