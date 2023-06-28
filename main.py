import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data import SMDDataset


# parameters
data_dir = 'dataset/SMD/'
saved_data_dir = 'dataset/SMD/SMD.pkl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# save dataset
if os.path.exists(saved_data_dir):
    with open(saved_data_dir, 'rb') as f:
        dataset = pickle.load(f)
    print('Load saved dataset')
else:
    dataset = SMDDataset(data_dir=data_dir)
    with open(saved_data_dir, 'wb') as f:
        pickle.dump(dataset, f)
    print('Saved dataset')

# initialize dataset and dataloader

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


class MusicTransformer(nn.Module):
    def __init__(self, feature_dim, num_heads, num_encoder_layers, output_dim):
        super(MusicTransformer, self).__init__()
        
        # Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        # Decoder Layer
        self.decoder = nn.Linear(feature_dim, output_dim)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        
        output = self.transformer_encoder(x)
        
        output = output.mean(dim=1, keepdim=True)
        
        output = self.decoder(output)
        
        return output


# parameters
feature_dim = 1025  # 频率维度
num_heads = 5
num_encoder_layers = 6
output_dim = 64000  # MIDI标签的长度

# 初始化网络
net = MusicTransformer(feature_dim, num_heads, num_encoder_layers, output_dim).to(device)

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# train the network
num_epochs = 50
losses = []
print('Start Training')
torch.save(net.state_dict(), f'models/piano_model_0.pth')
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.float().squeeze()
        
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = outputs.view(-1) # add this line to reshape your outputs tensor

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(
                f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 10}')
            running_loss = 0.0
    losses.append((epoch, loss.item()))
    
    if epoch % 5 == 4:
        torch.save(net.state_dict(), f'models/piano_model_{epoch + 1}.pth')

with open('loss.txt', 'w') as f:
    for loss in losses:
        f.write(f'loss: {loss}\n')

print('Finished Training')

# save the model
torch.save(net.state_dict(), f'models/piano_model_final.pth')
