import os, gc
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random 
import multiprocessing
from tqdm import tqdm  # 引入进度条模块

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# 自动创建输出目录
for d in ['./resultsv', './perf']:
    os.makedirs(d, exist_ok=True)

class Args:
    def __init__(self, batch_size=64, lr=0.001, nepoch=200, patience=15, wide=100, depth=3, n_train=100, m_train=1) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.nepoch = nepoch 
        self.patience = patience 
        self.wide = wide 
        self.depth = depth 
        self.biaoji = f"wide{wide}depth{depth}n{n_train}m{m_train}"
        self.n_train = n_train
        self.m_train = m_train

class Dataset_regression(Dataset): 
    def __init__(self, x, y): super().__init__(); self.x = x; self.y = y 
    def __len__(self): return len(self.x) 
    def __getitem__(self, index): return {"x": self.x[index], "y": self.y[index]}

class FlexibleNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_layer): 
        super().__init__()
        layers = [nn.Linear(n_feature, n_hidden), nn.ReLU()]
        for _ in range(n_layer - 2):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
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

class EarlyStopping():
    def __init__(self, save_path, args, delta=0):
        self.save_path = save_path; self.patience = args.patience 
        self.counter = 0; self.best_score = None 
        self.early_stop = False; self.val_loss_min = np.Inf; self.delta = delta 

    def __call__(self, model, train_loss, valid_loss, test_error, lipschitz, args, seed):
        score = -valid_loss 
        if self.best_score is None: 
            self.best_score = score 
            self.save_checkpoint(model, train_loss, valid_loss, test_error, lipschitz, args, seed) 
        elif score < self.best_score + self.delta: 
            self.counter += 1 
            if self.counter >= self.patience: self.early_stop = True 
        else:
            self.best_score = score
            self.save_checkpoint(model, train_loss, valid_loss, test_error, lipschitz, args, seed)
            self.counter = 0

    def save_checkpoint(self, model, train_loss, valid_loss, test_error, lipschitz, args, seed):
        torch.save(model.state_dict(), os.path.join(self.save_path, f'best{seed}{args.biaoji}network.pth'))
        torch.save(train_loss, os.path.join(self.save_path, f'best{seed}{args.biaoji}train_loss.pth')) 
        torch.save(valid_loss, os.path.join(self.save_path, f'best{seed}{args.biaoji}valid_loss.pth')) 
        torch.save(test_error, os.path.join(self.save_path, f'best{seed}{args.biaoji}test_loss.pth')) 
        torch.save(lipschitz, os.path.join(self.save_path, f'best{seed}{args.biaoji}lipschitz.pth')) 
        self.val_loss_min = valid_loss

def GPUstrain(x, y, x_valid, y_valid, x_test, y_test, args, seed):
    x_dim = x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = FlexibleNet(n_feature=x_dim, n_hidden=args.wide, n_output=1, n_layer=args.depth).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) 
    loss_func = nn.MSELoss() 
    
    train_dataloader = DataLoader(Dataset_regression(x, y), batch_size=args.batch_size, shuffle=True)
    x_valid = torch.from_numpy(x_valid).float().to(device) 
    y_valid = torch.from_numpy(y_valid).float().to(device) 
    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device) 

    early_stopping = EarlyStopping("./resultsv", args=args)

    # 训练循环加上进度条
    pbar = tqdm(range(args.nepoch), desc=f"Training {args.biaoji}", leave=False)
    for epoch in pbar: 
        net.train()
        train_epoch_loss = []
        for traindata in train_dataloader:
            x_train = torch.Tensor(traindata["x"]).float().to(device) 
            y_train = torch.Tensor(traindata["y"]).float().to(device) 
            outputs = net(x_train).view(-1)
            loss = loss_func(outputs, y_train)
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            train_epoch_loss.append(loss.item())
        
        avg_train_loss = np.average(train_epoch_loss)

        with torch.no_grad():
            net.eval() 
            loss_valid = loss_func(net(x_valid).view(-1), y_valid)
            error_test = loss_func(net(x_test).view(-1), y_test)
            current_lipschitz = compute_lipschitz_constant(net)

        pbar.set_postfix({'val_loss': f'{loss_valid.item():.4f}'})

        if epoch > 10:
            early_stopping(net, avg_train_loss, loss_valid.item(), error_test.item(), current_lipschitz, args, seed)
            if early_stopping.early_stop: 
                break 

    gc.collect()

def onedim(n_train, seed):
    m_train = 1
    seed2 = ((seed+50) * 20000331 ) % 2**31
    torch.manual_seed(seed2); np.random.seed(seed2); random.seed(seed2)
    
    train_df = pd.read_csv(f"./data/train_data/n{n_train}m{m_train}s{seed}.csv")
    valid_df = pd.read_csv(f"./data/valid_data/n{n_train}m{m_train}s{seed}.csv")
    test_df = pd.read_csv(f"./data/test_data/s{seed}.csv")

    x_train, y_train = train_df.filter(like='feature_').values, train_df['target'].values
    x_valid, y_valid = valid_df.filter(like='feature_').values, valid_df['target'].values
    x_test, y_test = test_df.filter(like='feature_').values, test_df['target'].values

    batch_size, lr = (64, 0.001) if n_train < 1000 else (256, 0.002)

    net_configs = [
        {'wide': 50, 'depth': 2, 'id': 0},
        {'wide': 100, 'depth': 3, 'id': 1},
        {'wide': 200, 'depth': 4, 'id': 2},
    ]
    
    csv_filename = f'./perf/n{n_train}m{m_train}s{seed}.csv'
    
    # 【断点续训核心逻辑】读取已存在的 CSV
    existing_results = []
    completed_ids = set()
    if os.path.exists(csv_filename):
        existing_df = pd.read_csv(csv_filename)
        existing_results = existing_df.to_dict('records')
        completed_ids = set(existing_df['net_id'].values)

    for config in net_configs:
        if config['id'] in completed_ids:
            print(f"⏩ 跳过: n={n_train}, seed={seed}, net_id={config['id']} (已完成)")
            continue

        args = Args(lr=lr, wide=config['wide'], depth=config['depth'], batch_size=batch_size, n_train=n_train, m_train=m_train)
        
        GPUstrain(x_train, y_train, x_valid, y_valid, x_test, y_test, args, seed2)
        
        try:
            train_loss_val = float(torch.load(os.path.join('./resultsv', f'best{seed2}{args.biaoji}train_loss.pth'), map_location='cpu'))
            valid_loss_val = float(torch.load(os.path.join('./resultsv', f'best{seed2}{args.biaoji}valid_loss.pth'), map_location='cpu'))
            test_loss_val = float(torch.load(os.path.join('./resultsv', f'best{seed2}{args.biaoji}test_loss.pth'), map_location='cpu'))
            lipschitz_val = float(torch.load(os.path.join('./resultsv', f'best{seed2}{args.biaoji}lipschitz.pth'), map_location='cpu'))
            
            existing_results.append({
                'n_train': n_train, 'm_train': m_train,
                'train_loss': train_loss_val, 'valid_loss': valid_loss_val,
                'test_loss': test_loss_val, 'lipschitz': lipschitz_val,
                'net_id': config['id']
            })
            
            # 每跑完一个配置就实时保存，防止突然中断
            pd.DataFrame(existing_results).to_csv(csv_filename, index=False)
            
        except Exception as e:
            print(f"❌ 读取指标失败: {e}")

if __name__ == '__main__': 
    n_values = [100, 500, 1000, 2000, 5000, 10000]
    seeds = list(range(10)) # 先跑10个seed测试
    
    # 生成需要跑的任务列表
    tasks = [(n, s) for n in n_values for s in seeds]
    
    print(f"🚀 总计任务数: {len(tasks)}")
    
    # 采用顺序执行，避免单卡显存爆炸（天垓150通常单卡显存大，但为了稳定推荐按序执行，用tqdm监控总进度）
    for n_train, seed in tqdm(tasks, desc="Total Tasks Progress"):
        onedim(n_train, seed)
    
    print("✅ 所有训练任务已完成！你的 visualize.py 现在可以直接读取 ./perf 目录生成图表了。")