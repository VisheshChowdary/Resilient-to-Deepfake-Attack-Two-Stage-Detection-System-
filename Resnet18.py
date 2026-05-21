import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import traceback
from tqdm import tqdm
import json
from datetime import datetime
import os
import time
from PIL import Image
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['real', 'fake']
        self.samples = []
        
        print(f"Loading dataset from {root_dir}")
        start_time = time.time()
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"Directory {class_dir} not found")
            
            print(f"Loading {class_name} images...")
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if not os.path.exists(img_path):
                    print(f"Warning: Image {img_path} not found")
                    continue
                self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images in {time.time() - start_time:.2f} seconds")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None, None
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader


class ResNet18FC(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18FC, self).__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Extract all convolutional layers (everything except the FC layer)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  
        # This keeps layers up to the last conv block (excludes avgpool & fc)

        # Freeze first few layers (conv1 + layer1, i.e. shallow features)
        for name, param in list(self.features.named_parameters())[:20]:  
            param.requires_grad = False

        print("\nLayer Freezing Summary:")
        print("Frozen: Conv1 + Layer1 (shallow features)")
        print("Trainable: Deeper residual blocks + FC head")

        # The final ResNet18 feature map has 512 channels
        self.feature_size = 512

        # Fully connected classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Forward through ResNet18 backbone
        x = self.features(x)

        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        # Flatten
        x = torch.flatten(x, 1)

        # Forward through custom FC layers
        x = self.classifier(x)
        return x
def create_model(num_classes=2):
    model = ResNet18FC(num_classes=2)
    return model
import torch
import torch.nn as nn

def print_model_summary(model, input_size=(3, 224, 224)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    dummy_input = torch.randn(1, *input_size).to(device)

    print("\nModel Summary:")
    print("=" * 50)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            
            print("\nLayer Information:")
            total_params = 0
            trainable_params = 0
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    params = sum(p.numel() for p in module.parameters())
                    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    total_params += params
                    trainable_params += trainable
                    print(f"{name}: {module.__class__.__name__}, Parameters: {params:,} (Trainable: {trainable:,})")
            
            print(f"\nTotal Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            print("=" * 50)
    except Exception as e:
        print(f"Error in forward pass: {str(e)}")
        print(f"Input shape: {dummy_input.shape}")
def create_data_loaders(batch_size=32):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DeepfakeDataset('dataset_kaggle_unzipped/dataset/data/train', transform=transform_train)
    val_dataset = DeepfakeDataset('dataset_kaggle_unzipped/dataset/data/val', transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader
def train_model_mlp(model, train_loader, val_loader, num_epochs=7, patience=5, learning_rate=0.001):
    # Create directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"model/densenet_kan_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': [],
        'epochs': []
    }
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = model.to(device)
    
    # Initialize loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    # ✅ Add weight decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # ✅ Track LR changes
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} → {new_lr:.6f}")
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(new_lr)
        history['epochs'].append(epoch + 1)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {new_lr:.6f}')
        
        # Save model and history
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
            
            np.save(os.path.join(save_dir, 'train_confusion.npy'), {
                'predictions': np.array(all_preds),
                'labels': np.array(all_labels)
            })
            np.save(os.path.join(save_dir, 'val_confusion.npy'), {
                'predictions': np.array(val_preds),
                'labels': np.array(val_labels)
            })
            
            with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
    }, os.path.join(save_dir, 'final_model.pth'))
    
    print(f"Training completed. Results saved in {save_dir}")
    return history
# Main Function
def main():
    model = create_model()
    print_model_summary(model)
    train_loader, val_loader = create_data_loaders()
    start_time = time.time()
    history = train_model_mlp(model, train_loader, val_loader)
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTotal Training Time: {hours}h {minutes}m {seconds}s")

if __name__ == "__main__":
    main()
