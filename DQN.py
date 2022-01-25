import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_layer):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=3, padding=1)
        
        self.norm1_1 = nn.BatchNorm2d(num_features=num_layer)
        self.norm1_2 = nn.BatchNorm2d(num_features=num_layer)
        self.norm2_1 = nn.BatchNorm2d(num_features=num_layer)
        self.norm2_2 = nn.BatchNorm2d(num_features=num_layer)

        self.conv_res1 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=1)
        self.conv_res2 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=1)

        self.globpool = nn.AvgPool2d(kernel_size=(6, 10), stride=2, padding=1)

        self.fc1 = nn.Linear(in_features=8*5*5, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=25)
        self.fc4 = nn.Linear(in_features=25, out_features=15)
        self.out = nn.Linear(in_features=15, out_features=3)

    def forward(self, t):
        Y = t
        t = F.relu(self.norm1_1(self.conv1_1(t)))
        t = F.relu(self.norm1_2(self.conv1_2(t)) + self.conv_res1(Y))

        Y = t
        t = F.relu(self.norm2_1(self.conv2_1(t)))
        t = F.relu(self.norm2_2(self.conv2_2(t)) + self.conv_res2(Y))
        
        t = self.globpool(t)

        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = F.relu(self.fc4(t))
        t = F.relu(self.out(t))
        return t
