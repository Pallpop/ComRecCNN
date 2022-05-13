import torch.nn as nn
from torch.nn.functional import relu, log_softmax

class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(784, 1000) # 784表示输入神经元数量，1000表示输出神经元数量
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)
        return log_softmax(x, dim=1)
