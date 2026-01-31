import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import tarfile

# CIFAR100 Mean and Std
CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def prepare_cifar_data(data_root='./data'):
    """
    Check if data exists. If not, try to extract from AutoDL public path.
    """
    target_path = os.path.join(data_root, 'cifar-100-python')
    if os.path.exists(target_path):
        print(f"Data already ready at {target_path}")
        return

    # AutoDL specific paths
    possible_paths = [
        '/root/autodl-pub/cifar-100/cifar-100-python.tar.gz',
        'autodl-pub/cifar-100/cifar-100-python.tar.gz',
        '../autodl-pub/cifar-100/cifar-100-python.tar.gz'
    ]
    
    found_archive = None
    for p in possible_paths:
        if os.path.exists(p):
            found_archive = p
            break
            
    if found_archive:
        print(f"Found AutoDL dataset at {found_archive}. Extracting to {data_root}...")
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        with tarfile.open(found_archive, 'r:gz') as tar:
            tar.extractall(path=data_root)
        print("Extraction complete.")
    else:
        print("AutoDL dataset not found, torchvision will attempt download.")

def compute_lipschitz_constant(model):
    """
    Compute Lipschitz constant via Spectral Norm product.
    """
    lipschitz = 1.0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            if isinstance(module, nn.Conv2d):
                w = w.view(w.size(0), -1)
            # Use svdvals for efficiency on GPU
            if w.is_cuda:
                s = torch.linalg.svdvals(w)
            else:
                # Fallback for CPU if svdvals is not implemented for some versions
                _, s, _ = torch.linalg.svd(w, full_matrices=False)
            
            layer_lip = s[0].item()
            lipschitz *= layer_lip
            
    return lipschitz

def get_network(net_name, num_classes=100):
    if net_name == 'resnet18':
        from resnet import resnet18
        net = resnet18(num_classes)
    elif net_name == 'resnet50':
        from resnet import resnet50
        net = resnet50(num_classes)
    else:
        raise ValueError('Network not supported')
    return net

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, n_samples=None):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # root points to where 'cifar-100-python' folder is located
    cifar_training = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)

    if n_samples is not None:
        total_data = len(cifar_training)
        if n_samples > total_data:
            n_samples = total_data
        
        # Fixed seed for data subsetting consistency across runs
        indices = np.random.RandomState(42).choice(total_data, n_samples, replace=False)
        cifar_training = Subset(cifar_training, indices)

    cifar_training_loader = DataLoader(
        cifar_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=False):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar_test = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)

    cifar_test_loader = DataLoader(
        cifar_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar_test_loader