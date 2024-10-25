import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ================================
# Dataset class
# ================================
class EmotionDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {"angry": 0, "happy": 1, "sad": 2, "surprised": 3, "neutral": 4}

        # Load images and labels
        for label in self.label_map.keys():
            label_folder = os.path.join(folder_path, label)
            for filename in os.listdir(label_folder):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(label_folder, filename))
                    self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('L')  # Convert to grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# ================================
# Define the CNN model
# ================================
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Set the input size for the first fully connected layer
        self.fc1 = nn.Linear(256 * 3 * 3, 512)  # Adjusted for output after convolution
        self.fc2 = nn.Linear(512, 5)  # 5 output classes for the emotions
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flattening
        x = x.view(x.size(0), -1)  # Flattening
        
        # Forward pass through fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ================================
# Data preprocessing and loading
# ================================
# Data transforms
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Parameters
batch_size = 32

# Dataset and DataLoader
train_dataset = EmotionDataset('datasets/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = EmotionDataset('datasets/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ================================
# Model training setup
# ================================
# Initialize model, loss function, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================================
# Model training
# ================================
best_val_accuracy = 0.0
early_stopping_counter = 0
patience = 5

for epoch in range(50):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = correct_train / total_train

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = correct_val / total_val

    print(f'Epoch [{epoch + 1}/50], Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Early Stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break

# ================================
# Model testing
# ================================
# การทดสอบโมเดล
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

def test_image(image_path):
    # เปิดภาพและแปลงเป็น grayscale
    image = Image.open(image_path).convert('L')  # แปลงเป็นสีเทา
    # ปรับขนาดภาพให้เป็น 48x48
    image = image.resize((48, 48))  # ปรับขนาดภาพที่เปิดเป็น 48x48
    # แปลงเป็น tensor
    image = transform(image).unsqueeze(0).to(device)  # เพิ่ม dimension สำหรับ batch size

    with torch.no_grad():
        outputs = model(image)  # ส่งภาพเข้าโมเดล
        probabilities = F.softmax(outputs, dim=1)  # คำนวณ softmax เพื่อให้เป็นค่าความน่าจะเป็น

    # แสดงผลลัพธ์
    emotion_labels = ["angry", "happy", "sad", "surprised", "neutral"]
    print("ผลลัพธ์การจำแนกอารมณ์:")
    for label, prob in zip(emotion_labels, probabilities[0]):
        print(f"{label}: {prob:.2f}")

# ทดสอบภาพที่ต้องการ
test_image('datasets/test/im8.png')