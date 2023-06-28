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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model))

        # Ensure that d_model is even
        assert d_model % 2 == 0, "d_model should be even"

        # Explicitly calculating sine and cosine terms with correct dimensions
        pe[:, 0::2] = torch.sin(position * div_term.unsqueeze(0))
        pe[:, 1::2] = torch.cos(position * div_term.unsqueeze(0))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MusicTransformer(nn.Module):
    def __init__(self, feature_dim, num_heads, num_encoder_layers, output_dim):
        super(MusicTransformer, self).__init__()

        # positional encoding
        self.positional_encoding = PositionalEncoding(feature_dim)

        # Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=2048)

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers)

        self.layer_norm = nn.LayerNorm(feature_dim)
        # Decoder Layer
        self.decoder = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = self.positional_encoding(x)
        output = self.transformer_encoder(x)
        output = self.layer_norm(output)
        output = output.mean(dim=1, keepdim=True)
        output = self.decoder(output)
        return output


# parameters
feature_dim = 1024  # frequency dim
num_heads = 8
num_encoder_layers = 6
output_dim = 64000  # MIDI label max length

# initialize network
net = MusicTransformer(feature_dim, num_heads,
                       num_encoder_layers, output_dim).to(device)

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=5, factor=0.5)

# train the network
num_epochs = 20
losses = []
print('Start Training')
if not os.path.exists('models'):
    os.makedirs('models')
# torch.save(net.state_dict(), f'models/piano_model_0.pth')
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.float().squeeze()
        print(inputs.shape, labels.shape)
        optimizer.zero_grad()
        outputs = net(inputs)
        # add this line to reshape your outputs tensor
        outputs = outputs.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(
                f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 10}')
            running_loss = 0.0
    scheduler.step(loss.item())
    losses.append((epoch, loss.item()))

    if epoch % 5 == 4:
        torch.save(net.state_dict(), f'models/piano_model_{epoch + 1}.pth')

with open('loss.txt', 'w') as f:
    for loss in losses:
        f.write(f'loss: {loss}\n')

print('Finished Training')

# save the model
torch.save(net.state_dict(), f'models/piano_model_final.pth')
