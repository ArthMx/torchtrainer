import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchtrainer import Trainer

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64))
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x.view(-1, self.fc1.in_features))
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), 1)

def accuracy(y_pred, y):
    """Compute the accuracy for a batch of prediction and target."""
    _, y_pred = torch.max(y_pred, 1)
    return (y_pred == y).cpu().numpy().mean()

transform = transforms.ToTensor()
# Download MNIST dataset and make datasets
root_dir = "../example"
data_dir = root_dir + "/data"
train_dataset = datasets.MNIST(data_dir, train=True, transform=transform, 
                               download=True)
test_dataset = datasets.MNIST(data_dir, train=False, transform=transform, 
                              download=True)

# Make train and test Dataloader
BATCH_SIZE = 128
epochs = 5
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                         shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=4)

# Instanciate model and set optimizer and loss function
net = Net()
loss_fn = F.nll_loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

trainer = Trainer(net, loss_fn, optimizer, metrics={"acc": accuracy})
print("Number of trainable parameters: %d" % trainer.get_num_parameters())
print(trainer.model)

checkpoint_path = root_dir + "/models/chkpth.tar"
trainer.fit(train_loader, val_loader=test_loader, epochs=epochs, verbose=1,
          plot_loss=True, early_stopping=5, checkpoint_path=checkpoint_path)