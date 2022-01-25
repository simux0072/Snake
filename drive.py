import torch
from os.path import exists

class model_drive():
    def __init__(self, path, model_name):
        self.path = path
        self.model_name = model_name

    def upload(self, model, optimizer, update, current_step):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'update': update,
            'current_step': current_step
            }, self.path + self.model_name)

    def does_exist(self):
        return exists(self.path + self.model_name)

    def download(self):
        return torch.load(self.path + self.model_name)