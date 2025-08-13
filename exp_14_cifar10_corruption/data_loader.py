import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import io

# Standard CIFAR-10 transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# --- Corruption Functions ---

def gaussian_noise(image, severity):
    """Adds Gaussian noise to an image."""
    c, h, w = image.shape
    noise = torch.randn(c, h, w) * severity
    return torch.clamp(image + noise, 0, 1)

def gaussian_blur(image, severity):
    """Applies Gaussian blur to an image."""
    if severity == 0: return image
    kernel_size = int(4 * severity + 1)
    if kernel_size % 2 == 0: kernel_size += 1
    
    blurrer = transforms.GaussianBlur(kernel_size=kernel_size, sigma=severity)
    return blurrer(image)

def brightness(image, severity):
    """Adjusts the brightness of an image."""
    return torch.clamp(image * (1 + severity), 0, 1)

def jpeg_compression(image, severity):
    """Applies JPEG compression artifacts to an image."""
    # Map severity (0.1 -> 0.9) to JPEG quality (90 -> 10)
    quality = int(100 * (1 - severity))
    quality = max(10, min(100, quality)) # Ensure quality is in [10, 100]
    
    # Convert to PIL Image
    pil_image = transforms.ToPILImage()(image)
    
    # Save to an in-memory buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, "JPEG", quality=quality)
    
    # Reload from buffer and convert back to tensor
    buffer.seek(0)
    reloaded_image = Image.open(buffer)
    return transforms.ToTensor()(reloaded_image)


CORRUPTIONS = {
    'noise': gaussian_noise,
    'blur': gaussian_blur,
    'brightness': brightness,
    'jpeg': jpeg_compression
}

class CorruptedCIFAR10(Dataset):
    """Wrapper for CIFAR-10 that applies a corruption on the fly."""
    def __init__(self, dataset, corruption_type, severity):
        self.dataset = dataset
        if corruption_type not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        self.corruption_func = CORRUPTIONS[corruption_type]
        self.severity = severity

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        corrupted_image = self.corruption_func(image, self.severity)
        return corrupted_image, label

def get_dataloaders(corruption_type, severity, batch_size=128):
    """
    Returns dataloaders for clean and corrupted CIFAR-10.
    """
    clean_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    corrupted_dataset = CorruptedCIFAR10(clean_dataset, corruption_type, severity)
    
    clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)
    corrupted_loader = DataLoader(corrupted_dataset, batch_size=batch_size, shuffle=True)
    
    return clean_loader, corrupted_loader

if __name__ == '__main__':
    # Example usage
    clean_loader, corrupted_loader = get_dataloaders('noise', 0.5)
    
    for images, _ in corrupted_loader:
        print(f"Corrupted image batch shape: {images.shape}")
        # You can optionally save or display an image to verify the corruption
        break
