import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils_mod import (get_network, get_training_dataloader, get_test_dataloader, 
                       compute_lipschitz_constant, prepare_cifar_data, 
                       CIFAR100_MEAN, CIFAR100_STD)

# ================= Configuration =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LR = 0.01
EPOCHS = 30 # 可根据时间调整，一般30-50个epoch能看到收敛趋势
NET_NAME = 'resnet18' 
NET_ID_MAP = {'resnet18': 0, 'resnet50': 1}

# Sample sizes to test (N)
N_VECTOR = [500, 1000, 2500, 5000, 10000, 25000, 50000]
SEEDS = range(3) # 每个N跑3次，节省时间 (原论文通常多一些)

# Files
RESULT_FILE = "./resnet_experiment_results.csv"
STATE_FILE = "./experiment_state.json"

def train_one_epoch(net, loader, optimizer, loss_fn):
    net.train()
    train_loss = 0.0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        total += images.size(0)
    return train_loss / total

def evaluate(net, loader, loss_fn):
    net.eval()
    test_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return test_loss / total, 1.0 - (correct / total) # Return Loss and Error Rate

def run_experiment():
    print(f"--> Preparing Data...")
    prepare_cifar_data() # 解压数据
    
    print(f"--> Starting Experiment on {DEVICE} using {NET_NAME}")
    
    # Checkpoint: Resume logic
    completed_configs = set()
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            try:
                state = json.load(f)
                # JSON loads lists, convert back to tuple for set lookup
                completed_configs = set([tuple(x) for x in state.get('completed', [])])
                print(f"--> Resuming... {len(completed_configs)} configurations already completed.")
            except:
                print("--> State file corrupted, starting fresh.")

    # Init CSV
    if not os.path.exists(RESULT_FILE):
        df = pd.DataFrame(columns=['n_train', 'm_train', 'train_loss', 'valid_loss', 'test_loss', 'lipschitz', 'net_id', 'seed'])
        df.to_csv(RESULT_FILE, index=False)

    # Configs
    configs = []
    for n in N_VECTOR:
        for seed in SEEDS:
            configs.append((n, seed))
    
    # Progress Bar
    # Only process configs that are NOT in completed_configs
    remaining_configs = [c for c in configs if c not in completed_configs]
    
    if len(remaining_configs) == 0:
        print("All configurations completed!")
        return

    progress_bar = tqdm(remaining_configs, desc="Exp Progress")
    
    for n_train, seed in progress_bar:
        progress_bar.set_description(f"Running N={n_train}, Seed={seed}")
        
        # Set Seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        # Data
        train_loader = get_training_dataloader(
            CIFAR100_MEAN, CIFAR100_STD, 
            batch_size=BATCH_SIZE, 
            n_samples=n_train,
            num_workers=4
        )
        test_loader = get_test_dataloader(
            CIFAR100_MEAN, CIFAR100_STD, 
            batch_size=BATCH_SIZE,
            num_workers=4
        )
        
        # Model & Opt
        net = get_network(NET_NAME).to(DEVICE)
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        loss_fn = nn.CrossEntropyLoss()
        
        final_train_loss = 0
        final_test_loss = 0
        final_test_error = 0
        
        # Epoch Loop
        # leave=False keeps the main bar clean
        epoch_pbar = tqdm(range(EPOCHS), desc=f"Ep (N={n_train})", leave=False)
        for epoch in epoch_pbar:
            train_loss = train_one_epoch(net, train_loader, optimizer, loss_fn)
            test_loss, test_error = evaluate(net, test_loader, loss_fn)
            scheduler.step()
            
            final_train_loss = train_loss
            final_test_loss = test_loss
            final_test_error = test_error
            
            epoch_pbar.set_postfix({
                'TrL': f"{train_loss:.3f}", 
                'TeE': f"{test_error:.3f}"
            })
        
        # Compute Lipschitz
        lipschitz = compute_lipschitz_constant(net)
        
        # Save Result
        result_data = {
            'n_train': n_train,
            'm_train': 1, # Fixed for image classification
            'train_loss': final_train_loss,
            'valid_loss': final_test_loss, 
            'test_loss': final_test_error, # Storing Error Rate
            'lipschitz': lipschitz,
            'net_id': NET_ID_MAP[NET_NAME],
            'seed': seed
        }
        
        # Write to CSV immediately
        pd.DataFrame([result_data]).to_csv(RESULT_FILE, mode='a', header=False, index=False)
        
        # Update Checkpoint State
        completed_configs.add((n_train, seed))
        with open(STATE_FILE, 'w') as f:
            json.dump({'completed': list(completed_configs)}, f)
            
    print("Experiment Completed Successfully!")

if __name__ == '__main__':
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\nExperiment paused by user.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()