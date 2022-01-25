import random

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory.append(experience)
            del self.memory[0]
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size * 3