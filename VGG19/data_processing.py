import torchvision
import torch
class DataProcessor:
    """Initializes the data to train the VGG19 net."""
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loaders(self):
        """Returns the data loaders for training and validation."""
        # Define transformations for the training and validatio
        data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        # Create training and validation datasets
        cifar10 = torchvision.datasets.CIFAR10
        image_datasets = {x: cifar10(root=self.data_dir, train=(x == 'train'), download=True,
                                     transform=data_transforms[x])
                          for x in ['train', 'val']}    
        # Create training and validation dataloaders
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size,
                                                     shuffle=True, num_workers=self.num_workers)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        return dataloaders, dataset_sizes, class_names