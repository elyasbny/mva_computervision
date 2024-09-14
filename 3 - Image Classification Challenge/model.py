import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet101, resnet152

nclasses = 250


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))

    
    
class resnet_p(nn.Module):
    def __init__(self):
        super(resnet_p, self).__init__()
        #self.model_name = model_name


        #if self.model_name == "resnet50":
        model = resnet50(pretrained=True)
        # replace the head
        base_model = list(model.children())[:-1]
        self.model = nn.Sequential(*base_model)

        # Head
        self.flatten = nn.Flatten()
        # layer 1
        print(model.fc)
        self.linear1 = nn.Linear(model.fc.in_features, model.fc.in_features * 2)
        self.dropout1 = nn.Dropout(p=0.6)

        # layer 2
        self.linear2 = nn.Linear(model.fc.in_features * 2, 1024)
        self.dropout2 = nn.Dropout(p=0.4)

        # layer 3
        self.linear3 = nn.Linear(1024, nclasses)
        self.dropout3 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)

        x = self.dropout1(x)
        x = F.relu(self.linear1(x))

        x = self.dropout2(x)
        x = F.relu(self.linear2(x))

        x = self.dropout3(x)
        x = F.relu(self.linear3(x))

        return x

    
 