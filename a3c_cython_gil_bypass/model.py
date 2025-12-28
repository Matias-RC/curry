import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_size=2*10*10, hidden=128, action_space=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        # separate heads
        self.policy_head = nn.Linear(hidden, action_space)
        self.value_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        # small gain for near-uniform logits initially
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

        # make value head slightly larger gain
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return int(action.item()), value.item(), m.log_prob(action), m.entropy()
