import torch
import torch.nn as nn


class HospitalFLModel(nn.Module):

    def __init__(self,input_size):

        super(HospitalFLModel,self).__init__()

        self.fc1 = nn.Linear(input_size,64)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(64,32)

        self.fc3 = nn.Linear(32,1)

    def forward(self,x):

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        x = self.fc3(x)

        return x

'''import torch
import torch.nn as nn

class FLModel(nn.Module):

    def __init__(self,input_size):

        super(FLModel,self).__init__()

        self.layer1 = nn.Linear(input_size,64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64,32)
        self.output = nn.Linear(32,1)

    def forward(self,x):

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.output(x)

        return x'''