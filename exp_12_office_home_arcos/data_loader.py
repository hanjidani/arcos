import os
import zipfile
import subprocess
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

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

def download_and_extract_data():
    """
    Downloads and extracts the Office-Home dataset from Kaggle.
    
    Note: This function requires the Kaggle API to be set up.
    You need to have a `kaggle.json` file in your `~/.kaggle/` directory.
    """
    dataset_path = './data/office-home'
    if os.path.exists(dataset_path):
        print("Dataset already downloaded.")
        return

    print("Downloading Office-Home dataset from Kaggle...")
    # It's important to use subprocess here to call the kaggle CLI
    # as the kaggle library itself doesn't have a stable python API for this.
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'karntiwari/home-office-dataset', '-p', './data', '--unzip'], check=True)
    print("Dataset downloaded and extracted successfully.")


class OfficeHomeDataset(Dataset):
    """Custom Dataset for Office-Home."""

    def __init__(self, base_path, domain, transform=None):
        """
        Args:
            base_path (string): Path to the office-home dataset directory.
            domain (string): Domain to load ('Art', 'Real World', etc.).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.domain_path = os.path.join(base_path, domain)
        self.transform = transform
        self.classes = sorted(os.listdir(self.domain_path))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.domain_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                item = (img_path, self.class_to_idx[class_name])
                images.append(item)
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_office_home_dataloaders(batch_size=32):
    """
    Downloads the Office-Home dataset and returns dataloaders for the 'Art' and 'Real World' domains.
    """
    download_and_extract_data()
    base_path = './data/home-office-dataset'
    
    art_dataset = OfficeHomeDataset(base_path=base_path, domain='Art', transform=data_transforms['train'])
    real_world_dataset = OfficeHomeDataset(base_path=base_path, domain='Real World', transform=data_transforms['train'])

    art_loader = DataLoader(art_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    real_world_loader = DataLoader(real_world_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return art_loader, real_world_loader


if __name__ == '__main__':
    art_loader, real_world_loader = get_office_home_dataloaders()
    print(f"Number of batches in Art loader: {len(art_loader)}")
    print(f"Number of batches in Real World loader: {len(real_world_loader)}")
    
    # Verify a batch
    for images, labels in art_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        break
