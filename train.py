import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch import nn, optim

# Basic transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dummy dataset loader (replace with actual path)
train_data = torchvision.datasets.FakeData(transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Model
model = models.resnet50(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train loop (1 epoch sample)
model.train()
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
print("Training Complete")
