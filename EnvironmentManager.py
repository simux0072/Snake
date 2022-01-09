import numpy as np
import torch
import torch
from torch._C import dtype
from torchvision import transforms
import snake

class EnvManager():
    def __init__(self, device):
        self.device = device
        self.done = False
        self.player = snake.player()
        self.food = snake.food(self.player)
        self.env = snake.environment(self.player, self.food)

    def reset(self):
        self.player = snake.player()
        self.food = snake.food(self.player)
        self.env = snake.environment(self.player, self.food)
        self.done = False
    
    def take_action(self, action):
        reward, points, self.done = self.player._play(self.food, action.item())
        return torch.tensor([reward], device=self.device), points

    def get_state(self):
        if self.done:
            return torch.zeros((1, 8, snake.DIMENSIONS[1], snake.DIMENSIONS[0]), dtype=torch.float32).to(self.device)
        else:
            state = torch.from_numpy(self.env._get_current_state()).unsqueeze(dim=0).to(self.device)
        return state

