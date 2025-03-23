# Hey there! Let's start by importing the libraries we need.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# Since we don't have a GPU, we will work with the CPU.
device = torch.device("cpu")
print("Using device:", device)

# ---------------------------------------------------------
# Data Preprocessing and Augmentation
# ---------------------------------------------------------
# For CIFAR-100, our images are originally 32x32.
# However, AlexNet expects 224x224 images (like those from ImageNet).
# So we'll resize our images to 224x224.
# We'll also apply normalization with CIFAR-100's mean and std values.
transform_train = transforms.Compose([
    transforms.Resize(224),                      # Resize to 224x224 pixels.
    transforms.RandomHorizontalFlip(),           # Augment our training data by randomly flipping images.
    transforms.ToTensor(),                       # Convert images to PyTorch tensors.
    transforms.Normalize((0.5071, 0.4865, 0.4409), # Normalize using CIFAR-100's mean values.
                         (0.2673, 0.2564, 0.2761))  # Normalize using CIFAR-100's std values.
])

transform_test = transforms.Compose([
    transforms.Resize(224),                      # Resize test images to 224x224.
    transforms.ToTensor(),                       # Convert images to tensors.
    transforms.Normalize((0.5071, 0.4865, 0.4409), # Normalize using the same values as training.
                         (0.2673, 0.2564, 0.2761))
])

# Download and load the CIFAR-100 training and testing datasets.
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              transform=transform_train, download=True)
test_dataset  = torchvision.datasets.CIFAR100(root='./data', train=False,
                                              transform=transform_test, download=True)

# Create DataLoaders for easy batch processing.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# ---------------------------------------------------------
# Model Setup: AlexNet for CIFAR-100
# ---------------------------------------------------------
# Load the AlexNet model.
# We're not using pre-trained weights here since we're retraining from scratch.
model = models.alexnet(pretrained=False)

# The original AlexNet classifier is set up for 1000 classes (ImageNet).
# Since CIFAR-100 has 100 classes, we need to change the final linear layer.
# The classifier part of AlexNet is a Sequential container.
# We replace the last layer (index 6) with a new Linear layer that has 100 outputs.
model.classifier[6] = nn.Linear(4096, 100)

# Move the model to our CPU device.
model = model.to(device)

# ---------------------------------------------------------
# Define Loss Function and Optimizer
# ---------------------------------------------------------
# We'll use cross-entropy loss, which is standard for classification tasks.
criterion = nn.CrossEntropyLoss()
# Using Adam optimizer here with a learning rate of 0.001.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------
# Training and Testing Functions
# ---------------------------------------------------------
# This function trains the model for one epoch.
def train(epoch):
    model.train()  # Set the model to training mode.
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move the data to our CPU.
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero out the gradients from the previous iteration.
        optimizer.zero_grad()

        # Forward pass: compute predictions.
        outputs = model(inputs)

        # Calculate the loss between predictions and actual labels.
        loss = criterion(outputs, targets)

        # Backward pass: compute gradient of the loss with respect to parameters.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Accumulate loss to print the average loss every 100 batches.
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx+1}, Loss: {running_loss/100:.3f}")
            running_loss = 0.0

# This function evaluates the model's performance on the test set.
def test():
    model.eval()  # Set the model to evaluation mode.
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculations for efficiency.
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # The predicted class is the one with the highest score.
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}% ({correct}/{total})")

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
# Now we'll train our model for a given number of epochs.
num_epochs = 10  # Feel free to increase this number if you want more training.
for epoch in range(1, num_epochs + 1):
    train(epoch)  # Train the model for one epoch.
    test()        # Evaluate the model on the test set after each epoch.

# ---------------------------------------------------------
# Saving the Model
# ---------------------------------------------------------
# Once training is complete, we save the model's parameters to a file.
torch.save(model.state_dict(), 'alexnet_cifar100.pth')
print("Model saved as alexnet_cifar100.pth")
