import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_office_home_dataloaders(batch_size=32):
    """
    Downloads the Office-Home dataset and returns dataloaders for the 'Art' and 'Real World' domains.
    """
    art_dataset = torchvision.datasets.OfficeHome(root='./data', domain='Art', transform=data_transforms['train'], download=True)
    real_world_dataset = torchvision.datasets.OfficeHome(root='./data', domain='Real World', transform=data_transforms['train'], download=True)

    art_loader = DataLoader(art_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    real_world_loader = DataLoader(real_world_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return art_loader, real_world_loader

if __name__ == '__main__':
    art_loader, real_world_loader = get_office_home_dataloaders()
    print(f"Number of batches in Art loader: {len(art_loader)}")
    print(f"Number of batches in Real World loader: {len(real_world_loader)}")
