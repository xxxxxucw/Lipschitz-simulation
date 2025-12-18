import os, gc
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
from colorama import init, Fore
from tqdm import tqdm # 引入进度条库

# ==========================================
# 4090 显卡设置与优化
# ==========================================
# 默认使用第一张卡，4090通常显存很大，不需要频繁清理
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
torch.backends.cudnn.benchmark = True # 开启加速，针对固定输入尺寸优化

init(autoreset=True)

class Args:
    """参数配置类"""
    def __init__(self, batch_size=10, lr=0.001, nepoch=200, patience=10, wide=100, depth=5, n_train=1, m_train=1):
        self.batch_size = batch_size
        self.lr = lr
        self.nepoch = nepoch 
        self.patience = patience 
        self.wide = wide 
        self.depth = depth 
        # 唯一标识符，用于临时文件命名
        self.biaoji = f"wide{wide}depth{depth}n{n_train}m{m_train}"
        self.n_train = n_train
        self.m_train = m_train

class EarlyStopping():
    """早停机制"""
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
        '''保存模型和指标'''
        # 使用 torch.save 保存所有指标，文件名包含 seed 和配置
        prefix = f'best{seed}{args.biaoji}'
        torch.save(model.state_dict(), os.path.join(self.save_path, f'{prefix}network.pth'))
        torch.save(train_loss, os.path.join(self.save_path, f'{prefix}train_loss.pth')) 
        torch.save(valid_loss, os.path.join(self.save_path, f'{prefix}valid_loss.pth')) 
        torch.save(test_error, os.path.join(self.save_path, f'{prefix}test_loss.pth')) 
        torch.save(lipschitz, os.path.join(self.save_path, f'{prefix}lipschitz.pth')) 
        self.val_loss_min = valid_loss

class Dataset_repeatedmeasurement(Dataset): 
    """数据集类"""
    def __init__(self, x, y):  
        super().__init__()
        self.x = x 
        self.y = y 

    def __len__(self): 
        return len(self.x) 
    
    def __getitem__(self, index): 
        return {
            "x" : self.x[index], 
            "y" : self.y[index]
        }

class happynet(nn.Module):
    """
    通用全连接神经网络 (MLP)
    支持 2-10 层深度配置
    """
    def __init__(self, n_feature, n_hidden, n_output, n_layer): 
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(n_feature, n_hidden))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layer - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(n_hidden, n_output))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def compute_lipschitz_constant(model):
    """
    计算模型的全局Lipschitz常数
    方法：计算每一层权重矩阵的谱范数（最大奇异值），然后将它们相乘。
    """
    lipschitz = 1.0
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach()
            # 使用 svd 计算最大奇异值
            if W.nelement() > 0:
                _, s, _ = torch.linalg.svd(W, full_matrices=False)
                lipschitz *= s[0].item()
    return lipschitz

def load_model_and_compute_lipschitz(model_path, n_feature, n_hidden, n_output, n_layer):
    """加载模型并计算Lipschitz常数"""
    model = happynet(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output, n_layer=n_layer)
    # PyTorch 2.6+ defaults weights_only=True, which blocks numpy scalars. Set to False for local trusted files.
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model, compute_lipschitz_constant(model)

def GPUstrain(x, y, x_valid, y_valid, x_test, y_test, args, seed, device):
    """训练循环"""
    
    # 针对 R 代码移植，输入维度是 3
    x_dim = 3 

    net = happynet(n_feature=x_dim, n_hidden=args.wide, n_output=1, n_layer=args.depth).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.90, 0.999), eps=1e-8, weight_decay=0.) 
    loss_func = nn.MSELoss() 

    train_epochs_loss = [] 
    valid_epochs_loss = [] 
    test_epochs_error = [] 

    # Flatten 数据以输入 MLP
    x = x.reshape(-1, x_dim)
    y = y.reshape(-1)

    train_dataset = Dataset_repeatedmeasurement(x, y)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # 转换为 Tensor 并移至 GPU
    x_valid = torch.from_numpy(x_valid).float().to(device) 
    y_valid = torch.from_numpy(y_valid).float().to(device) 
    x_test = torch.from_numpy(x_test).float().view(-1, 1, x_dim).to(device)
    y_test = torch.from_numpy(y_test).float().view(-1).to(device) 

    save_path = "./resultsv" 
    os.makedirs(save_path, exist_ok=True)
    
    early_stopping = EarlyStopping(save_path, args=args)

    for epoch in range(args.nepoch): 
        net.train()
        train_epoch_loss = []

        for traindata in train_dataloader:
            x_train = traindata["x"].float().view(-1, 1, x_dim).to(device)
            y_train = traindata["y"].float().to(device)
            
            outputs = net(x_train) 
            loss = loss_func(outputs.view(-1), y_train.view(-1))
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            
            train_epoch_loss.append(loss.item())

        avg_train_loss = np.average(train_epoch_loss)        
        train_epochs_loss.append(avg_train_loss)

        # Validation & Test
        with torch.no_grad():
            net.eval() 
            # Valid
            valid_predict = net(x_valid.view(-1, 1, x_dim))
            loss_valid = loss_func(valid_predict.view(-1), y_valid.view(-1))
            valid_epochs_loss.append(loss_valid.item())

            # Test
            test_predict = net(x_test)
            error_test = loss_func(test_predict.view(-1), y_test)
            test_epochs_error.append(error_test.item())

            # Lipschitz (每个 epoch 计算一次用于 monitoring，实际保存最好那个)
            current_lipschitz = compute_lipschitz_constant(net)

        # Early Stopping Check
        early_stopping(net, avg_train_loss, loss_valid.item(), error_test.item(), current_lipschitz, args, seed)
        if early_stopping.early_stop: 
            break 

    return net, train_epochs_loss, valid_epochs_loss, test_epochs_error

def onedim(n_train, m_train, seed, device_id=0):
    """单个实验任务"""
    
    # === Checkpoint 机制 ===
    # 如果结果文件已存在，直接跳过训练
    save_file = f"./perf/n{n_train}m{m_train}s{seed}.csv"
    if os.path.exists(save_file):
        # 返回 Skip 状态供进度条显示（可选）
        return

    # 设置随机种子
    seed2 = ((seed+50) * 20000331 ) % 2**31
    torch.manual_seed(seed2) 
    np.random.seed(seed2) 
    random.seed(seed2) 
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    n_valid = math.ceil(n_train * 0.25)
    m_valid = m_train

    # 读取 CSV 数据
    try:
        train_df = pd.read_csv(f"./data/train_data/n{n_train}m{m_train}s{seed}.csv")
        valid_df = pd.read_csv(f"./data/valid_data/n{n_train}m{m_valid}s{seed}.csv")
        test_df = pd.read_csv(f"./data/test_data/s{seed}.csv")
    except FileNotFoundError:
        print(f"Error: Data file not found for n={n_train}, seed={seed}")
        return None

    # 提取特征和标签 (Feature columns start with 'feature_')
    feat_cols = [c for c in train_df.columns if c.startswith('feature_')]
    
    x_train = train_df[feat_cols].values
    y_train = train_df['target'].values
    x_valid = valid_df[feat_cols].values
    y_valid = valid_df['target'].values
    x_test = test_df[feat_cols].values
    y_test = test_df['target'].values

    # 动态调整 Batch Size 和 LR (根据样本量)
    total_samples = n_train * m_train
    if total_samples < 128:
       batch_size = min(total_samples, 32)
       lr = 0.0005
    elif total_samples < 1024:
        batch_size = 64
        lr = 0.0005
    elif total_samples < 4096:
        batch_size = 128
        lr = 0.001
    elif total_samples < 8192:
        batch_size = 256
        lr = 0.001
    else:
        batch_size = 1024
        lr = 0.002

    # 定义6种网络配置 (对应 visualize.py 的6条线)
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
        
        GPUstrain(x_train, y_train, x_valid, y_valid, x_test, y_test, args, seed2, device)
        
        # 读取保存的最佳结果
        try:
            prefix = f'best{seed2}{args.biaoji}'
            # 加载指标 - set weights_only=False to allow numpy scalars
            train_l = torch.load(f'./resultsv/{prefix}train_loss.pth', map_location='cpu', weights_only=False)
            valid_l = torch.load(f'./resultsv/{prefix}valid_loss.pth', map_location='cpu', weights_only=False)
            test_l = torch.load(f'./resultsv/{prefix}test_loss.pth', map_location='cpu', weights_only=False)
            lip_val = torch.load(f'./resultsv/{prefix}lipschitz.pth', map_location='cpu', weights_only=False)
            
            results.append({
                'n_train': n_train,
                'm_train': m_train,
                'train_loss': float(train_l),
                'valid_loss': float(valid_l),
                'test_loss': float(test_l),
                'lipschitz': float(lip_val),
                'net_id': config['id']
            })
        except Exception as e:
            print(f"Failed to load results for config {config}: {e}")
            results.append({
                'n_train': n_train, 'm_train': m_train,
                'train_loss': 0, 'test_loss': 0, 'lipschitz': 0, 'net_id': config['id']
            })
    
    # 保存结果 CSV
    os.makedirs("./perf", exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(save_file, index=False)
    # print(f"Saved: {save_file}")

# 包装函数，用于 imap 参数解包
def task_wrapper(args):
    return onedim(*args)

# 主程序配置
n_vector = [10, 13, 17, 23, 31, 42, 57, 78, 106, 145, 198, 271, 400]
m_vector = [20] # 固定 m=20
n_repeat = 50   # 50个种子

if __name__ == '__main__': 
    # 清理旧结果 (注意：如果启用了断点续传，通常不应该全部删除，这里保留目录结构)
    if not os.path.exists("./resultsv"):
        os.mkdir("./resultsv")
    if not os.path.exists("./perf"):
        os.mkdir("./perf")
    
    # 统计已有文件数量
    existing_files = [f for f in os.listdir("./perf") if f.endswith(".csv")]
    if len(existing_files) > 0:
        print(f"{Fore.YELLOW}Warning: Found {len(existing_files)} existing files in ./perf. These tasks will be SKIPPED.{Fore.RESET}")

    # 构建任务列表
    tasks = []
    for n in n_vector:
        for m in m_vector:
            for s in range(n_repeat):
                tasks.append((n, m, s))

    # 4090 性能很强，但 Python 的 GIL 限制了多线程效率，且 CUDA 上下文初始化开销大。
    # 策略：使用 multiprocessing 开启几个进程并行跑，每个进程跑串行任务。
    # 注意：如果要多进程同时用 GPU，需要控制并发数以免显存溢出。
    # 对于小模型，4090 可以同时跑 4-8 个进程。
    
    ctx = multiprocessing.get_context('spawn')
    num_processes = 8
    
    print(f"Start training on RTX 4090 with {len(tasks)} tasks using {num_processes} processes...")
    
    # 使用 imap_unordered 配合 tqdm 显示进度
    # 结果将是一个 list (虽然 onedim 没有返回值，但这会强制迭代完成)
    with ctx.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(task_wrapper, tasks), total=len(tasks), desc="Total Progress"))
    
    print("All training finished.")