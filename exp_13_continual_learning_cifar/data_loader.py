import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Define superclasses and their corresponding subclass labels
CIFAR100_SUPERCLASSES = {
    'vehicles_1': [8, 13, 48, 58, 90],  # bicycle, bus, motorcycle, pickup truck, train
    'insects': [4, 30, 55, 72, 95]  # bee, beetle, butterfly, caterpillar, cockroach
}

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])


class Subset(Dataset):
    """
    A dataset that is a subset of another dataset.
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = set(labels)
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in self.labels]
        
        # Create a mapping from old labels to new, contiguous labels
        self.label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(list(self.labels)))}
        
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, self.label_map[label]

    def __len__(self):
        return len(self.indices)


def get_cifar100_superclass_dataloaders(task_name, batch_size=128):
    """
    Creates a dataloader for a specific CIFAR-100 superclass.
    """
    full_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
    labels = CIFAR100_SUPERCLASSES[task_name]
    subset_dataset = Subset(full_dataset, labels)
    
    return DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


if __name__ == '__main__':
    vehicles_loader = get_cifar100_superclass_dataloaders('vehicles_1')
    insects_loader = get_cifar100_superclass_dataloaders('insects')

    print(f"Number of batches in vehicles_1 loader: {len(vehicles_loader)}")
    print(f"Number of batches in insects loader: {len(insects_loader)}")
    
    # Verify a batch
    for images, labels in vehicles_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print(f"Unique labels: {torch.unique(labels)}")
        break
