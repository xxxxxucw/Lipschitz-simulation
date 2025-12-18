import os, gc
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import multiprocessing
import shutil
from tqdm import tqdm 

# 针对 4090 优化
torch.backends.cudnn.benchmark = True

class Args:
    def __init__(self, batch_size=64, lr=0.001, nepoch=500, patience=20, wide=100, depth=3, n_train=1, m_train=1):
        self.batch_size = batch_size
        self.lr = lr
        self.nepoch = nepoch
        self.patience = patience
        self.wide = wide
        self.depth = depth
        self.biaoji = f"w{wide}d{depth}n{n_train}"
        self.n_train = n_train
        self.m_train = m_train

class EarlyStopping:
    def __init__(self, save_path, args, delta=0):
        self.save_path = save_path
        self.patience = args.patience
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
        # 保存临时文件
        torch.save(train_loss, os.path.join(self.save_path, f'best{seed}{args.biaoji}train_loss.pth'))
        torch.save(valid_loss, os.path.join(self.save_path, f'best{seed}{args.biaoji}valid_loss.pth'))
        torch.save(test_error, os.path.join(self.save_path, f'best{seed}{args.biaoji}test_loss.pth'))
        torch.save(lipschitz, os.path.join(self.save_path, f'best{seed}{args.biaoji}lipschitz.pth'))
        self.val_loss_min = valid_loss

class Dataset_TS(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}

class happynet(nn.Module):
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
    """计算各层谱范数的乘积"""
    lipschitz = 1.0
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach()
            s = torch.linalg.svdvals(W)
            lipschitz *= s[0].item()
    return lipschitz

def GPUstrain(x, y, x_valid, y_valid, x_test, y_test, args, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_dim = 1 

    net = happynet(n_feature=x_dim, n_hidden=args.wide, n_output=1, n_layer=args.depth).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    train_dataset = Dataset_TS(x, y)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    x_valid = torch.from_numpy(x_valid).float().view(-1, 1).to(device)
    y_valid = torch.from_numpy(y_valid).float().view(-1).to(device)
    x_test = torch.from_numpy(x_test).float().view(-1, 1).to(device)
    y_test = torch.from_numpy(y_test).float().view(-1).to(device)

    # 每一个进程使用独立的文件夹防止冲突，或者共用一个通过文件名区分
    save_path = "./resultsv"
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(save_path, args=args)

    for epoch in range(args.nepoch):
        net.train()
        train_losses = []
        
        for traindata in train_dataloader:
            x_train = traindata["x"].float().view(-1, 1).to(device)
            y_train = traindata["y"].float().view(-1).to(device)
            output = net(x_train).view(-1)
            loss = loss_func(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        with torch.no_grad():
            net.eval()
            valid_pred = net(x_valid).view(-1)
            loss_valid = loss_func(valid_pred, y_valid).item()
            test_pred = net(x_test).view(-1)
            error_test = loss_func(test_pred, y_test).item()
            current_lipschitz = compute_lipschitz_constant(net)

        early_stopping(net, avg_train_loss, loss_valid, error_test, current_lipschitz, args, seed)
        if early_stopping.early_stop:
            break

def run_experiment_wrapper(args_tuple):
    """
    包装器函数，用于解包参数并检查是否存在
    """
    n_train, seed = args_tuple
    m_train = 1
    
    # === [Check Point] 检查最终结果文件是否已存在 ===
    final_output_path = f'./perf/n{n_train}m{m_train}s{seed}.csv'
    if os.path.exists(final_output_path):
        if os.path.getsize(final_output_path) > 0:
            return f"Skipped n={n_train}, s={seed}"
    
    # 开始实验逻辑
    full_seed = ((seed + 50) * 20000331) % 2**31
    torch.manual_seed(full_seed)
    np.random.seed(full_seed)
    
    try:
        train_df = pd.read_csv(f"./data/train_data/n{n_train}m{m_train}s{seed}.csv")
        valid_df = pd.read_csv(f"./data/valid_data/n{int(n_train*0.25)}m{m_train}s{seed}.csv")
        test_df = pd.read_csv(f"./data/test_data/s{seed}.csv")
    except FileNotFoundError:
        return f"Error: Data missing for n={n_train}, seed={seed}"

    x_train = train_df['feature_1'].values
    y_train = train_df['target'].values
    x_valid = valid_df['feature_1'].values
    y_valid = valid_df['target'].values
    x_test = test_df['feature_1'].values
    y_test = test_df['target'].values

    net_configs = [
        {'wide': 50, 'depth': 2, 'id': 0},
        {'wide': 100, 'depth': 3, 'id': 1},
        {'wide': 200, 'depth': 4, 'id': 2},
        {'wide': 400, 'depth': 5, 'id': 3},
        {'wide': 600, 'depth': 6, 'id': 4},
        {'wide': 800, 'depth': 6, 'id': 5}
    ]

    results = []
    save_path = "./resultsv"
    
    for config in net_configs:
        if n_train < 1000: batch_size = 32
        elif n_train < 10000: batch_size = 128
        else: batch_size = 512
        
        args = Args(lr=0.001, wide=config['wide'], depth=config['depth'], 
                   batch_size=batch_size, n_train=n_train, m_train=m_train)
        
        GPUstrain(x_train, y_train, x_valid, y_valid, x_test, y_test, args, full_seed)
        
        try:
            # === [Fix] 添加 weights_only=False 以消除警告 ===
            res = {
                'n_train': n_train,
                'm_train': m_train,
                'train_loss': float(torch.load(os.path.join(save_path, f'best{full_seed}{args.biaoji}train_loss.pth'), weights_only=False)),
                'valid_loss': float(torch.load(os.path.join(save_path, f'best{full_seed}{args.biaoji}valid_loss.pth'), weights_only=False)),
                'test_loss': float(torch.load(os.path.join(save_path, f'best{full_seed}{args.biaoji}test_loss.pth'), weights_only=False)),
                'lipschitz': float(torch.load(os.path.join(save_path, f'best{full_seed}{args.biaoji}lipschitz.pth'), weights_only=False)),
                'net_id': config['id']
            }
            results.append(res)
            
            # 清理临时文件
            os.remove(os.path.join(save_path, f'best{full_seed}{args.biaoji}train_loss.pth'))
            os.remove(os.path.join(save_path, f'best{full_seed}{args.biaoji}valid_loss.pth'))
            os.remove(os.path.join(save_path, f'best{full_seed}{args.biaoji}test_loss.pth'))
            os.remove(os.path.join(save_path, f'best{full_seed}{args.biaoji}lipschitz.pth'))
            
        except Exception as e:
            # print(f"Error loading results: {e}")
            pass

    if results:
        os.makedirs("./perf", exist_ok=True)
        pd.DataFrame(results).to_csv(final_output_path, index=False)
        return f"Done n={n_train}, s={seed}"
    return f"Failed n={n_train}, s={seed}"

if __name__ == '__main__':
    n_vector = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
    n_repeat = 50
    
    tasks = [(n, s) for n in n_vector for s in range(n_repeat)]
    
    if not os.path.exists("./perf"): os.makedirs("./perf")
    if not os.path.exists("./resultsv"): os.makedirs("./resultsv")
    
    multiprocessing.set_start_method('spawn', force=True)
    
    n_process = 10 
    
    print(f"Starting {len(tasks)} experiments with {n_process} processes...")
    
    with multiprocessing.Pool(processes=n_process) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_experiment_wrapper, tasks), 
            total=len(tasks),
            desc="Training Progress",
            unit="exp"
        ))
    
    print("All experiments completed.")