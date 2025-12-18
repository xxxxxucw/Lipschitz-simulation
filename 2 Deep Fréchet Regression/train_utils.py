import os
import gc
import numpy as np
import math
import torch  
import torch.nn as nn     
from torch.utils.data import DataLoader, Dataset
import random 
import pandas as pd
import time
from colorama import init

init(autoreset=True)

# ==========================================
# 辅助类定义
# ==========================================

class Args:
    def __init__(self, batch_size=10, lr =0.001, nepoch = 200, patience = 10, wide = 100, depth = 5, n_train=1, m_train=1) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.nepoch = nepoch 
        self.patience = patience 
        self.wide = wide 
        self.depth = depth 
        self.biaoji = "wide" + str(wide) + "depth" + str(depth) + "n" + str(n_train) + "m" + str(m_train)
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
        # [修改点1] np.Inf 在 NumPy 2.0 中被移除，改为 np.inf
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
        torch.save(model.state_dict(), os.path.join(self.save_path, 'best' + str(seed) + args.biaoji +'network.pth') )
        torch.save(train_loss, os.path.join(self.save_path, 'best'+ str(seed) + args.biaoji +'train_loss.pth')) 
        torch.save(valid_loss, os.path.join(self.save_path, 'best'+ str(seed) + args.biaoji +'valid_loss.pth')) 
        torch.save(test_error, os.path.join(self.save_path, 'best'+ str(seed) + args.biaoji +'test_loss.pth')) 
        torch.save(lipschitz, os.path.join(self.save_path, 'best'+ str(seed) + args.biaoji +'lipschitz.pth')) 
        self.val_loss_min = valid_loss

class Dataset_repeatedmeasurement(Dataset): 
    def __init__(self, x, y) -> None:  
        super().__init__()
        self.x = x 
        self.y = y 
    def __len__(self) -> int: 
        return len(self.x) 
    def __getitem__(self, index): 
        return {"x" : self.x[index], "y" : self.y[index]}

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

# ==========================================
# 核心函数定义
# ==========================================

def compute_lipschitz_constant(model):
    """计算模型的全局Lipschitz常数"""
    lipschitz = 1.0
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach()
            try:
                # 尝试在GPU上计算
                _, s, _ = torch.linalg.svd(W, full_matrices=False)
                lipschitz *= s[0].item()
            except:
                # 失败则回退到CPU
                _, s, _ = torch.linalg.svd(W.cpu(), full_matrices=False)
                lipschitz *= s[0].item()
    return lipschitz

def GPUstrain(x, y, x_valid, y_valid, x_test, y_test, args, seed, device):
    x_dim = x.shape[1] 
    net = happynet(n_feature=x_dim, n_hidden=args.wide, n_output=1, n_layer=args.depth).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.90, 0.999), eps=1e-8) 
    loss_func = nn.MSELoss() 

    train_epochs_loss = [] 
    valid_epochs_loss = [] 
    test_epochs_error = [] 

    # 数据集准备
    train_dataset = Dataset_repeatedmeasurement(x, y)
    # num_workers=0 避免多进程开销
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    x_valid = torch.from_numpy(x_valid).float().to(device) 
    y_valid = torch.from_numpy(y_valid).float().to(device) 
    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device) 

    save_path = "./resultsv" 
    early_stopping = EarlyStopping(save_path, args=args)

    for epoch in range(args.nepoch): 
        net.train()
        train_epoch_loss = []

        for traindata in train_dataloader:
            x_train = traindata["x"].float().to(device)
            y_train = traindata["y"].float().to(device)
            
            outputs = net(x_train) 
            loss = loss_func(outputs.view(-1), y_train.view(-1))
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            
            train_epoch_loss.append(loss.item())

        avg_train_loss = np.mean(train_epoch_loss)        
        train_epochs_loss.append(avg_train_loss)

        # Validation
        with torch.no_grad():
            net.eval() 
            valid_predict = net(x_valid)
            loss_valid = loss_func(valid_predict.view(-1), y_valid.view(-1))
            valid_epochs_loss.append(loss_valid.item())

            test_predict = net(x_test)
            error_test = loss_func(test_predict.view(-1), y_test.view(-1))
            test_epochs_error.append(error_test.item())

            current_lipschitz = compute_lipschitz_constant(net)

        # Early Stopping check
        if epoch > 10 or args.n_train*args.m_train > 200:
            early_stopping(net, avg_train_loss, loss_valid.item(), error_test.item(), current_lipschitz, args, seed)
            if early_stopping.early_stop: 
                break
    
    # 显式清理显存
    del net, optimizer, x_valid, y_valid, x_test, y_test
    torch.cuda.empty_cache()

    return None 

def onedim(n_train, m_train, seed):
    """
    Worker function to be called by multiprocessing pool
    """
    device = torch.device("cuda:0")

    seed2 = ((seed+50) * 20000331 )% 2**31
    torch.manual_seed(seed2) 
    np.random.seed(seed2) 
    random.seed(seed2) 
    
    n_valid = math.ceil(n_train*0.25)
    m_valid = m_train

    # 读取数据
    try:
        train_path = f"./data/train_data/n{n_train}m{m_train}s{seed}.csv"
        valid_path = f"./data/valid_data/n{n_train}m{m_valid}s{seed}.csv"
        test_path = f"./data/test_data/s{seed}.csv"
        
        if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
            print(f"Skipping n={n_train}, m={m_train}, seed={seed}: Data file missing.")
            return None

        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error loading data for n={n_train}, m={m_train}, seed={seed}: {e}")
        return None

    # 提取特征列
    feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
    
    x_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    x_valid = valid_df[feature_cols].values
    y_valid = valid_df['target'].values
    x_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    # Batch size 策略
    total_samples = n_train * m_train
    if total_samples < 128:
       batch_size, lr = min(total_samples, 32), 0.0005
    elif total_samples < 1024:
        batch_size, lr = 64, 0.0005
    elif total_samples < 4096:
        batch_size, lr = 128, 0.001
    elif total_samples < 16384:
        batch_size, lr = 256, 0.002
    else:
        batch_size, lr = 1024, 0.002

    # 网络配置
    net_configs = [
        {'wide': 50, 'depth': 2, 'id': 0},
        {'wide': 100, 'depth': 3, 'id': 1},
        {'wide': 200, 'depth': 4, 'id': 2},
        {'wide': 400, 'depth': 5, 'id': 3},
        {'wide': 600, 'depth': 6, 'id': 4},
        {'wide': 800, 'depth': 6, 'id': 5}
    ]
    
    results = []
    
    for config in net_configs:
        args = Args(lr=lr, wide=config['wide'], depth=config['depth'], 
                   batch_size=batch_size, n_train=n_train, m_train=m_train)
        
        # 训练
        GPUstrain(x_train, y_train, x_valid, y_valid, x_test, y_test, args, seed2, device)
        
        # 加载并汇总结果
        try:
            prefix = os.path.join('./resultsv', f'best{seed2}{args.biaoji}')
            
            # [修改点2] PyTorch 2.6+ 需要 weights_only=False 才能加载 numpy scalar
            train_loss = torch.load(prefix + 'train_loss.pth', map_location='cpu', weights_only=False)
            valid_loss = torch.load(prefix + 'valid_loss.pth', map_location='cpu', weights_only=False)
            test_loss = torch.load(prefix + 'test_loss.pth', map_location='cpu', weights_only=False)
            lipschitz = torch.load(prefix + 'lipschitz.pth', map_location='cpu', weights_only=False)
            
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
            print(f"Error processing results for net {config['id']}: {e}")

    # 保存结果到 CSV
    if results:
        df_res = pd.DataFrame(results)
        os.makedirs("./perf", exist_ok=True)
        output_csv = f'./perf/n{n_train}m{m_train}s{seed}.csv'
        df_res.to_csv(output_csv, index=False)
        return output_csv
        
    return None