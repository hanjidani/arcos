import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np

from data_loader import get_cifar100_superclass_dataloaders
from models import get_cnn_model, get_feature_extractor
from src.wasserstein import calculate_wasserstein_nd, calculate_mmd

def train_model(model, dataloader, num_epochs=10, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} completed.")
    return model

def evaluate_model(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def get_features(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor = get_feature_extractor(model)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    features_list = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            features = feature_extractor(inputs)
            features_list.append(features.cpu().numpy())
            
    return np.concatenate(features_list)

def main(args):
    # Get dataloaders
    vehicles_loader = get_cifar100_superclass_dataloaders('vehicles_1', args.batch_size)
    insects_loader = get_cifar100_superclass_dataloaders('insects', args.batch_size)
    
    # 1. Train Q1 on Task 1 (Vehicles)
    print("--- Training Q1 on Vehicles ---")
    q1_model = get_cnn_model(num_classes=5)
    q1_model = train_model(q1_model, vehicles_loader, num_epochs=args.num_epochs, lr=args.lr)
    
    # 2. Train Q2 on Task 2 (Insects), starting from Q1's weights
    print("\n--- Training Q2 on Insects (from Q1) ---")
    q2_model = get_cnn_model(num_classes=5)
    q2_model.load_state_dict(q1_model.state_dict())
    q2_model = train_model(q2_model, insects_loader, num_epochs=args.num_epochs, lr=args.lr)
    
    # 3. Analyze Forgetting and Decompose
    print("\n--- Analyzing Forgetting with ARCOS ---")
    
    # Risks on original task T1
    r_t1_q1 = evaluate_model(q1_model, vehicles_loader)
    r_t1_q2 = evaluate_model(q2_model, vehicles_loader)
    delta_r = abs(r_t1_q1 - r_t1_q2)
    
    # Generalization gaps
    g_q1 = r_t1_q1 # Approx. since train=test for this calculation
    g_q2 = evaluate_model(q2_model, insects_loader)
    
    # Divergence terms
    features_t1 = get_features(q1_model, vehicles_loader)
    features_t2 = get_features(q1_model, insects_loader)
    
    w1_dist = calculate_wasserstein_nd(features_t1, features_t2)
    mmd_dist = calculate_mmd(features_t1, features_t2)
    
    # Output distance
    features_q1_t2 = get_features(q1_model, insects_loader)
    features_q2_t2 = get_features(q2_model, insects_loader)
    output_dist = np.linalg.norm(features_q1_t2 - features_q2_t2, axis=1).mean()
    
    results = {
        'delta_r': [delta_r],
        'g_q1': [g_q1],
        'g_q2': [g_q2],
        'wasserstein_divergence': [w1_dist],
        'mmd_divergence': [mmd_dist],
        'output_distance': [output_dist]
    }
    
    df = pd.DataFrame(results)
    df.to_csv('arcos_continual_learning_results.csv', index=False)
    print("\nResults saved to arcos_continual_learning_results.csv")
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARCOS Continual Learning experiment on CIFAR-100.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for models.')
    args = parser.parse_args()
    main(args)
