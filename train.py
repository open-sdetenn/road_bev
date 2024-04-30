import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Model

batch_size = 32
learning_rate = 0.001
num_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_img_paths = ["0.npy"]
needed_classes = [0, 1, 6, 7, 8, 10, 11] 

train_dataset = Dataset(train_img_paths, needed_classes=needed_classes)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Model(needed_labels=needed_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
        running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'model.pth')
