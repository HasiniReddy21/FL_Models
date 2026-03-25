import flwr as fl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

from model import HospitalFLModel
from data_loader import load_data

import logging
logging.getLogger("flwr").setLevel(logging.ERROR)
# Load hospital dataset
X,y = load_data()

X = torch.tensor(X,dtype=torch.float32)
y = torch.tensor(y,dtype=torch.float32)

dataset = TensorDataset(X,y)

loader = DataLoader(dataset,batch_size=32,shuffle=True)

model = HospitalFLModel(X.shape[1])

criterion = torch.nn.MSELoss()

optimizer = optim.Adam(model.parameters(),lr=0.01)


def train():

    model.train()

    for epoch in range(3):

        for data,target in loader:

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output.squeeze(),target)

            loss.backward()

            optimizer.step()


class HospitalClient(fl.client.NumPyClient):

    def get_parameters(self,config):

        return [val.detach().numpy() for val in model.state_dict().values()]


    def set_parameters(self,parameters):

        params_dict = zip(model.state_dict().keys(),parameters)

        state_dict = {k:torch.tensor(v) for k,v in params_dict}

        model.load_state_dict(state_dict)


    def fit(self,parameters,config):

        self.set_parameters(parameters)

        train()

        return self.get_parameters({}),len(loader.dataset),{}


    def evaluate(self,parameters,config):

        self.set_parameters(parameters)

        loss = 0

        with torch.no_grad():

            for data,target in loader:

                output = model(data)

                loss += criterion(output.squeeze(),target).item()

        return float(loss),len(loader.dataset),{}


fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=HospitalClient()
)
# 🔹 Show Predictions
model.eval()

with torch.no_grad():

    sample_X = X[:10]
    sample_y = y[:10]

    predictions = model(sample_X).squeeze()

print("\nSample Predictions vs Actual:\n")

for i in range(10):
    print(f"Predicted: {predictions[i].item():.4f} | Actual: {sample_y[i].item():.4f}")

'''import flwr as fl
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

from model import FLModel
from data_loader import load_data

# Load dataset
X, y = load_data()

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X,y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = FLModel(X.shape[1])

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train():

    model.train()

    for epoch in range(2):

        for data,target in loader:

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output.squeeze(), target)

            loss.backward()

            optimizer.step()


class FLClient(fl.client.NumPyClient):

    def get_parameters(self):

        return [val.detach().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):

        params_dict = zip(model.state_dict().keys(), parameters)

        state_dict = {k: torch.tensor(v) for k, v in params_dict}

        model.load_state_dict(state_dict)

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        train()

        return self.get_parameters(), len(loader), {}

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        loss = 0

        with torch.no_grad():

            for data,target in loader:

                output = model(data)

                loss += criterion(output.squeeze(), target).item()

        return float(loss), len(loader), {}


fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FLClient(),
)'''