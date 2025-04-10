
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, BatchNorm
import torch_geometric


class AlphabetGNN(torch.nn.Module):
    def __init__(self, num_features=2, num_classes=31):
        super(AlphabetGNN, self).__init__()

        self.conv1 = GCNConv(num_features, 128)
        self.bn1 = BatchNorm(128)
        self.conv2 = GCNConv(128, 256)
        self.bn2 = BatchNorm(256)
        self.conv3 = GATConv(256, 512) 
        self.bn3 = BatchNorm(512)

        self.fc1 = torch.nn.Linear(512, 256) 
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index


        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=0.5, training=self.training)


        x = F.elu(self.conv3(x, edge_index)) 
        x = self.bn3(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, data.batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
