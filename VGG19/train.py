import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import model as VGGModel
from data_processing import DataProcessor
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """
    Train the VGG model
    
    Args:
        model: The VGG model to train
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        criterion: Loss function
        optimizer: Optimizer function
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on (cuda/cpu)
    
    Returns:
        model: Trained model with best weights
        history: Training history
    """
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                # Track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Record history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save best model
                torch.save(model.state_dict(), 'best_vgg_model.pth')
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def plot_training_history(history):
    """
    Plot the training and validation loss and accuracy
    
    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.savefig('training_history.png')
    #plt.show()

def main():
    # Set device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    data_dir = './data'  # Change to your data directory
    batch_size = 32
    data_proc = DataProcessor(data_dir, batch_size)
    dataloaders, dataset_sizes, class_names = data_proc.get_data_loaders()
    print(f"dataset_sizes: {dataset_sizes}")   
    print(f"class_names: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Number of training samples: {dataset_sizes['train']}")
    print(f"Number of validation samples: {dataset_sizes['val']}")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {data_proc.num_workers}") 
    # Initialize the model
    num_classes = len(class_names)
    model = VGGModel(num_classes=num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    model, history = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        exp_lr_scheduler,
        num_epochs=25, 
        device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save the final model
    torch.save(model.state_dict(), 'final_vgg_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
