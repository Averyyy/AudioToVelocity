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


def train(model, dataloader, criterion, optimizer, device, scheduler):
    model.train()
    total_loss = 0.0

    for i, (audio, midi, velocity) in enumerate(dataloader):
        audio = audio.to(device)
        midi = midi.to(device)
        velocity = velocity.to(device)

        output = model(audio, midi)
        loss = criterion(output, velocity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

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
            audio = audio.to(device)
            midi = midi.to(device)
            velocity = velocity.to(device)

            output = model(audio, midi)
            loss = criterion(output, velocity)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_dim = 512
    num_layers = 32
    nhead = 16
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 4

    print('----- Loading Dataset -----')
    data_dir = os.path.join('data', 'SMD-8s-normalize')

    if not os.path.exists('data/processed/train_data.pkl'):
        dataset = VelocityDataset(data_dir)
        train_data, val_data = train_test_split(
            dataset, test_size=0.2, random_state=42)

        if not os.path.exists('data/processed'):
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

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerModel(freq_dim=1025, note_dim=90, hidden_dim=hidden_dim,
                             nhead=nhead, num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if not os.path.exists('logs'):
        os.makedirs('logs')

    print('----- Start Training -----')
    with open('logs/train_loss.txt', 'w') as f:
        for epoch in range(num_epochs):
            train_loss = train(model, train_dataloader,
                               criterion, optimizer, device, scheduler)
            val_loss = validate(model, val_dataloader, criterion, device)

            print(
                f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
            f.write(
                f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}\n')

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    print('----- Saving Model -----')
    torch.save(model.state_dict(), 'checkpoints/model.pth')


if __name__ == '__main__':
    main()
