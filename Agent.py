import random
import torch
import math
from torch import optim
from torch import nn

class Agent():
    def __init__(self, strategy, num_actions, device, target_net, policy_net, lr, gamma):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        self.target_net = target_net
        self.policy_net = policy_net
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_net.parameters(), self.lr)
        self.loss_function = nn.MSELoss()

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) #Explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim = 1).to(self.device) #Exploit 
    
    def train_memory(self, states, actions, rewards, next_states):
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_net(next_states).max(1)[0]
        discounted_q_values = rewards.squeeze(1) + next_q_values * self.gamma

        loss = self.loss_function(current_q_values, discounted_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def net_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class EpsilonGreedyStrat():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)
