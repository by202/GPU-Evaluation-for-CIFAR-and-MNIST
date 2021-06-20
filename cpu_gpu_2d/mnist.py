import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

# Device configuration

#Start time
time1 = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
n_layers = 2
num_epochs = 5
batch_size = 200
learning_rate = 0.001
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#Data load time
time2 = time.time()
data_load_time = time2 - time1
print('Data load time: ', data_load_time*1000, 'ms')

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, n_layers):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            # layers.append(nn.Dropout(0.5))
        self.inLayer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hiddenLayer = nn.Sequential(*layers)
        self.outLayer = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.inLayer(x)
        out = self.relu(out)
        out = self.hiddenLayer(out)
        out = self.outLayer(out)
        out = self.softmax(out)
        return out
model = NeuralNet(input_size, hidden_size, num_classes, n_layers).to(device)

#Model setup time
time3 = time.time()
model_load_time = time3 - time2
print('Model setup time: ', model_load_time*1000, 'ms')

#print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
model.train()
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:

            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100*correct/total

            print ('Epoch [{}/{}], Step [{}/{}], Accuracy: {}, Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, acc, loss.item()))
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

#Model training time
time4 = time.time()
model_train_time = time4 - time3
print('Model trainning time: ', model_train_time*1000, 'ms')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

#Evaluation time
time5 = time.time()
eval_time = time5 - time4
print('Evaluation time: ', eval_time*1000, 'ms')