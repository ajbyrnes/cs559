import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
def train(args, model, device, train_loader, optimizer, epoch):
    # Put model in training mode
    model.train()
    
    tot_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get next sample, label
        data, target = data.to(device), target.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Compute loss and backpropagate
        loss = torch.nn.CrossEntropyLoss()(output, target)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Accumulate loss
        tot_loss = tot_loss + loss.item()
        
        # Report progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                tot_loss / (batch_idx + 1),
                100. * correct / ((batch_idx + 1) * args.batch_size)
            ))
            
    # Report end of epoch stats
    print('End of Epoch: {}'.format(epoch))
    print('Training Loss: {:.6f}, Training Accuracy: {:.2f}%'.format(
        tot_loss / len(train_loader),
        100. * correct / len(train_loader.dataset)
    ))
    
def test(args, model, device, test_loader):
    # Put model in evaluation
    model.eval()
    
    tot_loss, correct = 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # Get next sample, label
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Accumulatel loss
            tot_loss += torch.nn.CrossEntropyLoss()(output, target).item()
            
            # Accumulate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    # Report test stats
    print('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(
        tot_loss / len(test_loader),
        100. * correct / len(test_loader.dataset)
    ))
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size)
    
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        
        scheduler.step()
        
        if epoch == args.epochs:
            import matplotlib.pyplot as plt
            import os
            
            weights = model.fc1.weight.data.cpu()
            
            os.makedirs('filters', exist_ok=True)
            
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for i, ax in enumerate(axes.flat):
                weight_img = weights[i].view(28, 28)
                ax.imshow(weight_img, cmap='gray')
                ax.set_title(f'Class {i}')
                ax.axis('off')
                
            plt.suptitle('FC Layer Weights After 5 Epochs')
            plt.tight_layout()
            plt.savefig('filters/fc1_weights_epoch5.png')
            plt.show()
            
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        
if __name__ == '__main__':
    main()