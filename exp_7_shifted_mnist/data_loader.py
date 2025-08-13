import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import rotate, shift, zoom

# --- Main Data Loading Function ---

def get_mnist_loaders(batch_size=128, test_batch_size=1000, transform=None):
    """
    Creates and returns the MNIST train and test data loaders.
    """
    if transform is None:
        transform = transforms.ToTensor()

    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=test_batch_size,
        shuffle=False
    )
    return train_loader, test_loader

# --- Shift Application Functions ---

def apply_shift_to_dataset(dataset, shift_type, shift_params):
    """
    Applies a specified shift to an entire dataset (numpy array).
    Returns a new numpy array with the transformation.
    """
    data = dataset.data.numpy()
    shifted_data = np.zeros_like(data)

    for i in range(len(data)):
        img = data[i]
        if shift_type == 'translation':
            shifted_data[i] = shift(img, shift=(shift_params['dy'], shift_params['dx']), cval=0)
        elif shift_type == 'rotation':
            shifted_data[i] = rotate(img, angle=shift_params['angle'], reshape=False, cval=0)
        elif shift_type == 'scale':
            # SciPy zoom is a bit tricky with padding, let's do it carefully
            h, w = img.shape
            zoom_factor = shift_params['s']
            zoomed_img = zoom(img, zoom_factor)
            zh, zw = zoomed_img.shape
            
            # Pad or crop to original size
            if zoom_factor > 1: # Cropping
                y_start = (zh - h) // 2
                x_start = (zw - w) // 2
                shifted_data[i] = zoomed_img[y_start:y_start+h, x_start:x_start+w]
            else: # Padding
                padded_img = np.zeros_like(img)
                y_start = (h - zh) // 2
                x_start = (w - zw) // 2
                padded_img[y_start:y_start+zh, x_start:x_start+zw] = zoomed_img
                shifted_data[i] = padded_img
        elif shift_type == 'morphological':
            # This requires scikit-image or opencv, let's add a placeholder
            # from skimage.morphology import erosion, dilation, square
            # kernel = square(shift_params['k'])
            # if shift_params['type'] == 'erosion':
            #     shifted_data[i] = erosion(img, kernel)
            # else:
            #     shifted_data[i] = dilation(img, kernel)
            pass # Placeholder
        else:
            raise ValueError(f"Unknown shift_type: {shift_type}")

    # Create a new TensorDataset from the shifted data
    shifted_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(shifted_data).unsqueeze(1).float() / 255.0, # Add channel dim and normalize
        dataset.targets
    )
    return shifted_dataset

if __name__ == '__main__':
    # --- Example Usage ---
    print("Loading original MNIST data...")
    _, test_dataset = datasets.MNIST('../data', train=False, download=True)

    # 1. Translation Example
    print("\nApplying translation shift...")
    translated_dataset = apply_shift_to_dataset(test_dataset, 'translation', {'dx': 5, 'dy': -5})
    # You can now create a DataLoader from this
    translated_loader = DataLoader(translated_dataset, batch_size=1000)
    print(f"Created a translated dataset with {len(translated_dataset)} images.")

    # 2. Rotation Example
    print("\nApplying rotation shift...")
    rotated_dataset = apply_shift_to_dataset(test_dataset, 'rotation', {'angle': 15})
    rotated_loader = DataLoader(rotated_dataset, batch_size=1000)
    print(f"Created a rotated dataset with {len(rotated_dataset)} images.")

    # 3. Scale Example
    print("\nApplying scale shift...")
    scaled_dataset = apply_shift_to_dataset(test_dataset, 'scale', {'s': 0.8})
    scaled_loader = DataLoader(scaled_dataset, batch_size=1000)
    print(f"Created a scaled dataset with {len(scaled_dataset)} images.")
