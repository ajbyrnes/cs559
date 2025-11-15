import argparse
import os
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class ShapeDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (filepath, class_index)
        transform: optional torchvision transform
        """
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Load image with matplotlib
        image = plt.imread(path)
        
        # Convert numpy array to torch tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Change to (C, H, W)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def split_filenames(data_dir, class_label_map, class_length=10000, train_ratio=0.8):
    # Load and sort filenames
    filenames = sorted(os.listdir(data_dir))
    
    # Get labels for filenames
    labels = [class_label_map[filename.split('_')[0]] for filename in filenames]

    # Turn filenames into full paths
    filenames = [os.path.join(data_dir, fname) for fname in filenames]

    # Combine filenames and labels
    combined = list(zip(filenames, labels))

    # Split into train and test sets
    training_files, test_files = [], []
    for i in range(len(class_label_map)):
        start_index = i * class_length
        end_index = (i + 1) * class_length

        class_filenames = combined[start_index:end_index]
        training_files.extend(class_filenames[:int(class_length * train_ratio)])
        test_files.extend(class_filenames[int(class_length * train_ratio):])
        
    return training_files, test_files

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 47 * 47, 128)
        self.fc2 = nn.Linear(128, 9)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
def train(args, model, device, train_loader, optimizer, epoch):
    # Put model in training mode
    model.train()
    
    # Training loop
    total_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # for data, target in zip(batch_images, batch_labels):
        data, target = data.to(device), target.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)  # Add batch dimension
        
        # Compute loss
        loss = torch.nn.CrossEntropyLoss()(output, target)
        
        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Accumulate loss and accuracy
        total_loss += loss.item()
            
        # Report progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                total_loss / (batch_idx + 1),
                100. * correct / ((batch_idx + 1) * args.batch_size)
            ))
            
    # Report end of epoch stats
    print('End of Epoch: {}'.format(epoch))
    print('Training Loss: {:.6f}, Training Accuracy: {:.2f}%'.format(
        total_loss / len(train_loader),
        100. * correct / len(train_loader.dataset)
    ))
        

def test(args, model, device, test_loader):
    # Put model in evaluation
    model.eval()
    
    # Testing loop
    total_loss, correct = 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # Get next sample, label
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = torch.nn.CrossEntropyLoss()(output, target)
            
            # Accumulate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Accumulate loss
            total_loss += loss.item()
            
        # Report test stats
        print('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(
            total_loss / len(test_loader),
            100. * correct / len(test_loader.dataset)
        ))

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Homework 5')
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Split files in data into train and test sets
    # These are just tuples of (filepath, label) for now
    data_dir = 'geometry_dataset'
    
    classes = [
        "Circle", "Square", "Octagon", "Heptagon", "Nonagon",
        "Star", "Hexagon", "Pentagon", "Triangle"
    ]
    class_label_map = { cls_name: idx for idx, cls_name in enumerate(classes) }

    training_files, test_files = split_filenames(data_dir, class_label_map)

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create datasets
    train_dataset = ShapeDataset(training_files, transform=transform)
    test_dataset = ShapeDataset(test_files, transform=transform)
    
    # Load datasets into dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # Define a simple CNN model
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # Train-test loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # test(args, model, device, test_loader)
        scheduler.step()
        
    if args.save_model:
        torch.save(model.state_dict(), "shape_model.pt")
