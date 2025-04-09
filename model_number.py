import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, BatchNorm
import torch_geometric

class HandPoseGNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(HandPoseGNN, self).__init__()
        self.conv1 = GCNConv(2, 64)  
        self.conv2 = GCNConv(64, 128)
        self.fc = torch.nn.Linear(128, num_classes)  

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = torch_geometric.nn.global_mean_pool(x, data.batch)

        return F.log_softmax(self.fc(x), dim=1)
    


