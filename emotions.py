import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# Define the CNN model class
class SimpleCNN(nn.Module):
    def __init__(self, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust based on image size and network design
        self.fc2 = nn.Linear(128, output_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the image
    transforms.ToTensor(),        # Convert the image to Tensor
])

# Load data from the 5 folders
train_data = datasets.ImageFolder(root='datasets', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Create validation dataset
val_size = int(0.2 * len(train_data))  # 20% for validation

train_size = len(train_data) - val_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Model parameters
input_size = 3 * 64 * 64  # Input size for the fully connected layer (not used in CNN)
output_size = len(train_data.dataset.classes)  # Number of classes

# Instantiate the model and move it to GPU if available
model = SimpleCNN(output_size).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the model
epochs = 10
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        # Move images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Validation
    model.eval()  # Set the model to evaluation mode
    val_labels = []
    val_predictions = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_predictions.extend(predicted.cpu().numpy())
    
    val_accuracy = accuracy_score(val_labels, val_predictions)
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Save the model and optimizer state
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_names': train_data.dataset.classes  # Save class names for later use
}, 'model_checkpoint.pth')

print("Model training and evaluation completed. Model saved.")