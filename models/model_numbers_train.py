import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from model_number import NumbersGNN




def get_data(full_dataset_tensor, full_labelset_tensor, edge_list):
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    data_list = []
    
    for i in range(len(full_dataset_tensor)):
        x = full_dataset_tensor[i]
        y = full_labelset_tensor[i] 
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list



def train(model, batch_size, num_epochs, data_list, learning_rate, loader):
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in loader:
            optimizer.zero_grad()

            # Forward pass
            out = model(batch)
            
            # Compute the loss
            loss = loss_fn(out, batch.y)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")




def test(model, loader):
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")




full_numbers_dataset = np.load("number_dataset/full_datset.npy")
full_numbers_labelset = np.load("number_dataset/full_labelset.npy")

full_dataset_tensor = torch.tensor(full_numbers_dataset, dtype=torch.float)
full_labelset_tensor = torch.tensor(full_numbers_labelset, dtype=torch.long)

edge_list = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (0,18), (0,19), (0,20)] 

data_list = get_data(full_dataset_tensor, full_labelset_tensor, edge_list)

model = NumbersGNN(num_classes=10)  
batch_size = 16
num_epochs = 10
learning_rate = 0.001
loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)


train(model, batch_size, num_epochs, data_list, learning_rate, loader)
test(model, loader)


#torch.save(model.state_dict(), 'models/enter_file_name.pth')