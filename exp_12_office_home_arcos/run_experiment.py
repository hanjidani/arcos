import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from data_loader import get_office_home_dataloaders, data_transforms
from models import get_resnet_model, get_feature_extractor
import pandas as pd
from src.wasserstein import calculate_wasserstein_nd, calculate_mmd
import numpy as np

def train_model(model, dataloader, num_epochs=25, lr=0.001, weight_decay=0.0):
    """
    Trains a model on the given dataloader.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
    return model

def get_features(model, dataloader):
    """
    Extracts features from a model for a given dataloader.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor = get_feature_extractor(model)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    features_list = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            features = feature_extractor(inputs)
            features_list.append(features.cpu().numpy().reshape(features.size(0), -1))
            
    return np.concatenate(features_list)

def calculate_generalization_error(model, dataloader):
    """
    Calculates the generalization error of a model on a given dataloader.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
    return running_loss / len(dataloader.dataset)

def arcos_decomposition(q_model, q_tilde_model, source_loader, target_loader, use_mmd=False):
    """
    Performs the ARCOS decomposition.
    """
    # 1. Calculate generalization errors
    g_q = calculate_generalization_error(q_model, source_loader)
    g_q_tilde = calculate_generalization_error(q_tilde_model, target_loader)
    
    # 2. Calculate delta R
    source_risk = calculate_generalization_error(q_tilde_model, source_loader)
    target_risk = g_q_tilde
    delta_r = np.abs(target_risk - source_risk)
    
    # 3. Calculate divergence
    source_features = get_features(q_model, source_loader)
    target_features = get_features(q_model, target_loader)
    
    if use_mmd:
        divergence = calculate_mmd(source_features, target_features)
    else:
        divergence = calculate_wasserstein_nd(source_features, target_features)
        
    return {
        'delta_r': delta_r,
        'g_q': g_q,
        'g_q_tilde': g_q_tilde,
        'divergence': divergence
    }

def main(args):
    """
    Main function to run the experiment.
    """
    # Get dataloaders
    art_loader, real_world_loader = get_office_home_dataloaders(batch_size=args.batch_size)
    
    # --- Scenario A: Standard Shift ---
    print("--- Training Scenario A ---")
    # Train source model Q on Art domain
    q_model = get_resnet_model(num_classes=65)
    print("Training source model Q on Art domain...")
    q_model = train_model(q_model, art_loader, num_epochs=args.num_epochs, lr=args.lr, weight_decay=args.weight_decay)
    
    # Train target model Q_tilde on Real-World domain
    q_tilde_model = get_resnet_model(num_classes=65)
    print("\nTraining target model Q_tilde on Real-World domain...")
    q_tilde_model = train_model(q_tilde_model, real_world_loader, num_epochs=args.num_epochs, lr=args.lr, weight_decay=args.weight_decay)
    
    # --- Scenario B: Shift + Poor Generalization ---
    print("\n--- Training Scenario B ---")
    # Train target model Q_tilde_bad on Real-World domain with bad hyperparameters
    q_tilde_bad_model = get_resnet_model(num_classes=65)
    print("\nTraining target model Q_tilde_bad on Real-World domain with bad hyperparameters...")
    q_tilde_bad_model = train_model(q_tilde_bad_model, real_world_loader, num_epochs=args.num_epochs, lr=args.bad_lr, weight_decay=0.0)
    
    # --- ARCOS Decomposition ---
    print("\n--- ARCOS Decomposition ---")
    
    # Scenario A
    print("\nDecomposition for Scenario A (Wasserstein):")
    arcos_a_w = arcos_decomposition(q_model, q_tilde_model, art_loader, real_world_loader, use_mmd=False)
    print(arcos_a_w)
    
    print("\nDecomposition for Scenario A (MMD):")
    arcos_a_mmd = arcos_decomposition(q_model, q_tilde_model, art_loader, real_world_loader, use_mmd=True)
    print(arcos_a_mmd)

    # Scenario B
    print("\nDecomposition for Scenario B (Wasserstein):")
    arcos_b_w = arcos_decomposition(q_model, q_tilde_bad_model, art_loader, real_world_loader, use_mmd=False)
    print(arcos_b_w)
    
    print("\nDecomposition for Scenario B (MMD):")
    arcos_b_mmd = arcos_decomposition(q_model, q_tilde_bad_model, art_loader, real_world_loader, use_mmd=True)
    print(arcos_b_mmd)

    # Save results to CSV
    results = {
        'Scenario A (Wasserstein)': arcos_a_w,
        'Scenario A (MMD)': arcos_a_mmd,
        'Scenario B (Wasserstein)': arcos_b_w,
        'Scenario B (MMD)': arcos_b_mmd,
    }
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('arcos_office_home_results.csv')
    print("\nResults saved to arcos_office_home_results.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARCOS experiment on Office-Home dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for good models.')
    parser.add_argument('--bad_lr', type=float, default=0.1, help='Learning rate for the bad model.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for good models.')
    args = parser.parse_args()
    main(args)
