import torch
import torch.nn as nn
import torch.nn.functional as F

class iris_network(nn.Module):
    def __init__(self, input_size, num_classes):
        super(iris_network, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.4)

        self.fc4 = nn.Linear(128, num_classes)


    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.sigmoid(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.tanh(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x