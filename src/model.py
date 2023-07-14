# File: src/model.py

import os
import torch
import torch.nn as nn
from dataset import VelocityDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNEncoder, self).__init__()

        # Define the CNN layers here
        self.conv1 = nn.Conv1d(input_dim, output_dim,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(output_dim, output_dim,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.float()
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x


class TransformerModel(nn.Module):
    def __init__(self, freq_dim, note_dim, hidden_dim, nhead, num_layers, dropout=0.5):
        super(TransformerModel, self).__init__()

        # Define the source and target embedding layers
        self.source_embedding = CNNEncoder(freq_dim, hidden_dim)
        self.target_embedding = CNNEncoder(note_dim, hidden_dim)

        # Define the transformer
        self.transformer = nn.Transformer(
            hidden_dim, nhead, num_layers, num_layers, hidden_dim, dropout)

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dim, 128)
        self.output_layer = nn.Linear(hidden_dim, 128)

    def forward(self, source, target):
        # Embed source and target
        # Source shape: (batch_size, hidden_dim, seq_length)
        source = self.source_embedding(source)
        # Source shape: (seq_length, batch_size, hidden_dim)
        source = source.permute(2, 0, 1)
        # Target shape: (note_num, batch_size, hidden_dim)
        target = target.permute(0, 2, 1)
        target = self.target_embedding(target).permute(2, 0, 1)

        # print(source.shape, target.shape)
        target = target.permute(0, 2, 1)
        target = self.target_embedding(target).permute(2, 0, 1)

        # print(source.shape, target.shape)

        # Run through transformer
        output = self.transformer(source, target)
        # output: (num_notes, batch_size, hidden_dim)
        # output: (num_notes, batch_size, hidden_dim)
        # Predict velocities
        output = self.output_layer(output)
        # Apply softmax to the last dimension
        output = F.softmax(output, dim=-1)
        # Apply softmax to the last dimension
        output = F.softmax(output, dim=-1)

        # Return shape: (batch_size, seq_length, 128)
        return output.transpose(0, 1)
        # Return shape: (batch_size, seq_length, 128)
        return output.transpose(0, 1)


if __name__ == "__main__":
    # Initialize a transformer model and some dummy data
    model = TransformerModel(freq_dim=1025,
                             note_dim=90, hidden_dim=512, nhead=8, num_layers=6)
                             note_dim=90, hidden_dim=512, nhead=8, num_layers=6)
    source = torch.rand(1, 1025, 345)  # example audio input
    target = torch.rand(1, 38, 90)  # example MIDI input
    target = torch.rand(1, 38, 90)  # example MIDI input

    # Forward pass
    velocities = model(source, target)
    # Should be (batch_size, seq_length, 128)
    # Should be (batch_size, seq_length, 128)
    print(f"Output shape: {velocities.shape}")
