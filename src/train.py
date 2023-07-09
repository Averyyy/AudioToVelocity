# src/train.py
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from dataset import VelocityDataset
from model import TransformerModel
import pickle
from dataset import collate_fn


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for i, (audio, midi, velocity) in enumerate(dataloader):
        # Move data to the right device
        audio = audio.to(device)
        midi = midi.to(device)
        velocity = velocity.to(device)

        # Forward pass
        output = model(audio, midi)
        # print(output.shape, velocity.shape)
        loss = criterion(output, velocity)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 100 == 0:
            print(f'Batch {i} Loss: {loss.item()}')

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i, (audio, midi, velocity) in enumerate(dataloader):
            # Move data to the right device
            audio = audio.to(device)
            midi = midi.to(device)
            velocity = velocity.to(device)

            # Forward pass and calculate loss
            output = model(audio, midi)
            loss = criterion(output, velocity)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    # Set device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define hyperparameters
    hidden_dim = 512
    num_layers = 32
    nhead = 16
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 4

    # Load and split dataset
    if not os.path.exists('data/processed/train_data.pkl'):
        data_dir = os.path.join('data', 'SMD-8s')
        dataset = VelocityDataset(data_dir)
        print(len(dataset))
        train_data, val_data = train_test_split(
            dataset, test_size=0.2, random_state=42)
        os.makedirs('data/processed')
        with open('data/processed/train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open('data/processed/val_data.pkl', 'wb') as f:
            pickle.dump(val_data, f)
    else:
        with open('data/processed/train_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('data/processed/val_data.pkl', 'rb') as f:
            val_data = pickle.load(f)

    # Define dataloaders
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Define the model, loss function, and optimizer
    model = TransformerModel(freq_dim=1025,
                             note_dim=3, hidden_dim=hidden_dim,
                             nhead=nhead, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader,
                           criterion, optimizer, device)
        val_loss = validate(model, val_dataloader, criterion, device)
        print(
            f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}')


if __name__ == '__main__':
    main()