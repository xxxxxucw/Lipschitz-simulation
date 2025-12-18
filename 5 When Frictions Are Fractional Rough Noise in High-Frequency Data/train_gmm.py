import os, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import numpy as np
import math
import torch  
import torch.nn as nn     
from torch.utils.data import DataLoader, Dataset
import random 
import multiprocessing
import subprocess
import shutil
import pandas as pd
import time
from tqdm import tqdm

def get_gpu_memory(device_id):
    """Retrieve memory usage of GPU."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--id={}".format(device_id), "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"])
        memory_used, memory_total = map(int, output.decode("utf-8").strip().split("\n")[0].split(","))
        return memory_used, memory_total
    except Exception as e:
        return None, None

def get_free_gpu():
    """Auto-select GPU with most free memory."""
    try:
        device_ids = list(range(torch.cuda.device_count()))
        memory_usages = []
        for device_id in device_ids:
            memory_used, memory_total = get_gpu_memory(device_id)
            if memory_used is not None and memory_total is not None:
                memory_free = memory_total - memory_used
                memory_usages.append((device_id, memory_free))
        
        if len(memory_usages) > 0:
            best_device_id = sorted(memory_usages, key=lambda x: x[1])[-1][0]
            return torch.device(f"cuda:{best_device_id}")
        else:
            return torch.device("cuda:0")
    except:
        return torch.device("cuda:0")

class Args:
    def __init__(self, batch_size=32, lr=0.001, nepoch=200, patience=15, wide=100, depth=3, n_train=1, m_train=1):
        self.batch_size = batch_size
        self.lr = lr
        self.nepoch = nepoch 
        self.patience = patience 
        self.wide = wide 
        self.depth = depth 
        self.biaoji = f"w{wide}d{depth}n{n_train}m{m_train}"
        self.n_train = n_train
        self.m_train = m_train

class EarlyStopping():
    def __init__(self, save_path, args, verbose=False, delta=0):
        self.save_path = save_path 
        self.patience = args.patience 
        self.verbose = verbose 
        self.counter = 0 
        self.best_score = None 
        self.early_stop = False 
        self.val_loss_min = np.inf 
        self.delta = delta 

    def __call__(self, model, train_loss, valid_loss, test_error, lipschitz, args, seed):
        score = -valid_loss 
        if self.best_score is None: 
            self.best_score = score 
            self.save_checkpoint(model, train_loss, valid_loss, test_error, lipschitz, args, seed) 
        elif score < self.best_score + self.delta: 
            self.counter += 1 
            if self.counter >= self.patience: 
                self.early_stop = True 
        else:
            self.best_score = score
            self.save_checkpoint(model, train_loss, valid_loss, test_error, lipschitz, args, seed)
            self.counter = 0

    def save_checkpoint(self, model, train_loss, valid_loss, test_error, lipschitz, args, seed):
        os.makedirs(self.save_path, exist_ok=True)
        prefix = f'best{seed}{args.biaoji}'
        torch.save(model.state_dict(), os.path.join(self.save_path, f'{prefix}network.pth'))
        torch.save(train_loss, os.path.join(self.save_path, f'{prefix}train_loss.pth')) 
        torch.save(valid_loss, os.path.join(self.save_path, f'{prefix}valid_loss.pth')) 
        torch.save(test_error, os.path.join(self.save_path, f'{prefix}test_loss.pth')) 
        torch.save(lipschitz, os.path.join(self.save_path, f'{prefix}lipschitz.pth')) 
        self.val_loss_min = valid_loss

class TimeSeriesDataset(Dataset): 
    def __init__(self, x, y):  
        self.x = x 
        self.y = y 

    def __len__(self): 
        return len(self.x) 
    
    def __getitem__(self, index): 
        return {
            "x" : self.x[index], 
            "y" : self.y[index]
        }

class HappyNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_layer): 
        super().__init__()
        layers = []
        layers.append(nn.Linear(n_feature, n_hidden))
        layers.append(nn.ReLU())
        for _ in range(n_layer - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hidden, n_output))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def compute_lipschitz_constant(model):
    lipschitz = 1.0
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach()
            _, s, _ = torch.linalg.svd(W, full_matrices=False)
            lipschitz *= s[0].item()
    return lipschitz

def train_model(x_train, y_train, x_valid, y_valid, x_test, y_test, args, seed, nocuda):
    x_dim = x_train.shape[1] 
    
    if nocuda == 9: device = get_free_gpu()
    elif nocuda == -1: device = torch.device("cpu")
    else: device = torch.device(f"cuda:{nocuda}")

    net = HappyNet(n_feature=x_dim, n_hidden=args.wide, n_output=1, n_layer=args.depth).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) 
    loss_func = nn.MSELoss() 
    
    x_valid_t = torch.from_numpy(x_valid).float().to(device)
    y_valid_t = torch.from_numpy(y_valid).float().to(device)
    x_test_t = torch.from_numpy(x_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)
    
    train_dataset = TimeSeriesDataset(x_train, y_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    save_path = "./resultsv_gmm"
    early_stopping = EarlyStopping(save_path, args=args)
    
    for epoch in range(args.nepoch): 
        net.train()
        train_epoch_loss = []

        for traindata in train_dataloader:
            bx = traindata["x"].float().to(device)
            by = traindata["y"].float().to(device)
            
            output = net(bx)
            loss = loss_func(output.view(-1), by.view(-1))
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            train_epoch_loss.append(loss.item())

        avg_train_loss = np.average(train_epoch_loss)

        with torch.no_grad():
            net.eval() 
            val_out = net(x_valid_t)
            loss_valid = loss_func(val_out.view(-1), y_valid_t.view(-1))
            
            test_out = net(x_test_t)
            error_test = loss_func(test_out.view(-1), y_test_t.view(-1))
            
            current_lip = compute_lipschitz_constant(net)

        early_stopping(net, avg_train_loss, loss_valid.item(), error_test.item(), current_lip, args, seed)
        if early_stopping.early_stop: 
            break
            
    return net

def process_single_seed(n_train, m_train, seed, nocuda):
    """
    Load data, train models, save results.
    包含 Checkpoint 逻辑。
    """
    
    result_file = f"./perf/n{n_train}m{m_train}s{seed}.csv"
    if os.path.exists(result_file):
        return None

    seed_internal = ((seed + 50) * 2024) % 2**31
    torch.manual_seed(seed_internal)
    np.random.seed(seed_internal)
    
    try:
        train_df = pd.read_csv(f"./data/train_data/n{n_train}m{m_train}s{seed}.csv")
        valid_df = pd.read_csv(f"./data/valid_data/n{n_train}m{m_train}s{seed}.csv")
        test_df = pd.read_csv(f"./data/test_data/s{seed}.csv")
    except FileNotFoundError:
        print(f"Data missing for n={n_train}, m={m_train}, s={seed}")
        return None

    def parse_df(df):
        X = df.pivot(index='sample_id', columns='measurement_id', values='feature_1').values
        Y = df.groupby('sample_id')['target'].first().values
        return X, Y

    x_train, y_train = parse_df(train_df)
    x_valid, y_valid = parse_df(valid_df)
    x_test, y_test = parse_df(test_df)

    if n_train < 128: batch_size = 32
    elif n_train < 1024: batch_size = 64
    else: batch_size = 128
    
    net_configs = [
        {'wide': 32, 'depth': 2, 'id': 0},
        {'wide': 64, 'depth': 3, 'id': 1},
        {'wide': 128, 'depth': 4, 'id': 2},
        {'wide': 256, 'depth': 5, 'id': 3},
        {'wide': 512, 'depth': 6, 'id': 4},
        {'wide': 512, 'depth': 8, 'id': 5}
    ]

    results = []

    for config in net_configs:
        args = Args(batch_size=batch_size, lr=0.001, wide=config['wide'], depth=config['depth'], 
                   n_train=n_train, m_train=m_train)
        
        train_model(x_train, y_train, x_valid, y_valid, x_test, y_test, args, seed_internal, nocuda)
        
        try:
            path_prefix = f"./resultsv_gmm/best{seed_internal}{args.biaoji}"
            # [修改] 显式设置 weights_only=False 以消除 FutureWarnings
            train_loss = torch.load(f"{path_prefix}train_loss.pth", weights_only=False)
            valid_loss = torch.load(f"{path_prefix}valid_loss.pth", weights_only=False)
            test_loss = torch.load(f"{path_prefix}test_loss.pth", weights_only=False)
            lipschitz = torch.load(f"{path_prefix}lipschitz.pth", weights_only=False)
            
            results.append({
                'n_train': n_train,
                'm_train': m_train,
                'train_loss': float(train_loss),
                'valid_loss': float(valid_loss),
                'test_loss': float(test_loss),
                'lipschitz': float(lipschitz),
                'net_id': config['id']
            })
        except Exception as e:
            print(f"Error loading results for config {config}: {e}")

    if results:
        df_res = pd.DataFrame(results)
        os.makedirs("./perf", exist_ok=True)
        df_res.to_csv(result_file, index=False)

def worker_wrapper(args):
    return process_single_seed(*args)

if __name__ == '__main__':
    n_values = [100, 200, 400, 800, 1600, 3200]
    m_values = [50]
    seeds = list(range(10)) 
    
    tasks = []
    for n in n_values:
        for m in m_values:
            for s in seeds:
                tasks.append((n, m, s, 9)) 

    print(f"Total tasks: {len(tasks)}")
    
    multiprocessing.set_start_method('spawn', force=True)
    
    with multiprocessing.Pool(processes=8) as pool: 
        for _ in tqdm(pool.imap_unordered(worker_wrapper, tasks), total=len(tasks), desc="Training Models"):
            pass
        
    print("All training tasks completed.")