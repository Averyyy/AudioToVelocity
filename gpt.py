import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data import SMDDataset

# Parameters
data_dir = 'dataset/SMD/'
saved_data_dir = 'dataset/SMD/SMD.pkl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load dataset
if os.path.exists(saved_data_dir):
    with open(saved_data_dir, 'rb') as f:
        dataset = pickle.load(f)
else:
    dataset = SMDDataset(data_dir=data_dir)
    with open(saved_data_dir, 'wb') as f:
        pickle.dump(dataset, f)

# Initialize dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


class MusicGPT(nn.Module):
    def __init__(self, gpt2_model_name='gpt2'):
        super(MusicGPT, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
        self.input_linear1 = nn.Linear(1024, 768)
        self.output_linear = nn.Linear(768, 1)
        self.output_linear2 = nn.Linear(8000, 64000)

    def forward(self, x):
        # x shape: [1, 1024, 64000]
        x = F.max_pool1d(x, kernel_size=8, stride=8)  # shape: [1, 1024, 8000]
        # print(x.shape)

        x = x.permute(0, 2, 1)  # shape: [1, 8000, 1024]

        # Sliding window approach
        window_size = 512  # GPT-2 max: 1024
        outputs = []
        for i in range(0, x.size(1), window_size):

            window = x[:, i:i+window_size, :]  # shape: [1, window_size, 1024]
            # shape: [1, window_size, 768]
            window_embed = F.relu(self.input_linear1(window))
            gpt2_output = self.gpt2(inputs_embeds=window_embed)[
                0]  # shape: [1, window_size, 768]
            # print(gpt2_output.shape)
            output = self.output_linear(gpt2_output)
            outputs.append(output.squeeze(-1))

        # Concatenate the outputs
        outputs = torch.cat(outputs, dim=1)  # shape: [1, 16000]
        outputs = self.output_linear2(outputs)  # shape: [1, 64000]
        # print(outputs.shape)
        return outputs.squeeze(0)  # shape: [16000]


# Initialize network
net = MusicGPT().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=5, factor=0.5)

if not os.path.exists('modelsGPT'):
    os.makedirs('modelsGPT')

# train the network
num_epochs = 20
losses = []
print('Start Training')

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.float().squeeze()
        # print(inputs.shape, labels.shape)
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
        torch.save(net.state_dict(), f'modelsGPT/piano_model_{epoch + 1}.pth')

with open('lossGPT.txt', 'w') as f:
    for loss in losses:
        f.write(f'loss: {loss}\n')

print('Finished Training')

# save the model
torch.save(net.state_dict(), f'modelsGPT/piano_model_final.pth')
