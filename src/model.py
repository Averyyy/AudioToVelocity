# File: src/model.py

import os
import torch
import torch.nn as nn
from dataset import VelocityDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, hidden_dim, 2).float() * (-np.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x


class TransformerModel(nn.Module):
    def __init__(self, freq_dim, note_dim, hidden_dim, nhead, num_layers, dropout=0.5, device='cpu'):
        super(TransformerModel, self).__init__()
        self.device = device

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Define the source and target embedding layers
        self.source_embedding = CNNEncoder(freq_dim, hidden_dim)
        # self.target_embedding = CNNEncoder(note_dim, hidden_dim)
        self.start_end_embedding = CNNEncoder(2, hidden_dim//2)
        self.pitch_embedding = CNNEncoder(88, hidden_dim-hidden_dim//2)

        # Define the transformer
        self.transformer = nn.Transformer(
            hidden_dim, nhead, num_layers, num_layers, hidden_dim, dropout)

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dim, 129)

    def forward(self, source, target):
        # Embed source and target
        source = self.source_embedding(source)
        source = source.permute(2, 0, 1)
        source = self.pos_encoder(source)

        start_end = target[:, :, :2].permute(0, 2, 1)
        pitch = target[:, :, 2:].permute(0, 2, 1)
        start_end_embed = self.start_end_embedding(start_end)
        pitch_embed = self.pitch_embedding(pitch)
        target = torch.cat((start_end_embed, pitch_embed),
                           dim=1).permute(2, 0, 1)

        # Run through transformer
        src_mask = self.create_padding_mask(source[:, :, 0]).to(
            self.device)
        tgt_mask = self.create_padding_mask(target[:, :, 0]).to(
            self.device)
        output = self.transformer(
            source, target, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        output = self.output_layer(output)

        return output.transpose(0, 1)

    def create_padding_mask(self, seq):
        seq = torch.eq(seq, -1)  # get where the seq is 0
        return seq.T


if __name__ == "__main__":
    model = TransformerModel(freq_dim=1025, note_dim=90,
                             hidden_dim=512, nhead=8, num_layers=6)
    source = torch.rand(1, 1025, 345)  # example audio input
    target = torch.rand(1, 38, 90)  # example MIDI input

    # Forward pass
    velocities = model(source, target)
    print(f"Output shape: {velocities.shape}")
