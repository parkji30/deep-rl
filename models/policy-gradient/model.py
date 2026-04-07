from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    "Policies are the brain of the RL Agent. How they make decisions"
    def __init__(self):
        super().__init__()
        self.blowup = nn.Linear(4, 128)
        self.output = nn.Linear(128, 2)

    def forward(self, x):
        x = self.blowup(x)
        x = F.gelu(x)
        action_scores = self.output(x) # left or right
        return F.softmax(action_scores, dim=-1)

    def act(self, state):
        " given a state do an action "
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.forward(state)
        m = Categorical(action_probs)
        action = m.sample()
        # We need log prob for differentiability
        return action.item(), m.log_prob(action)

