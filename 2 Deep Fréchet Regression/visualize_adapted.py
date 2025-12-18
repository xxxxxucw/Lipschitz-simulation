import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import os

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=15)
rc('text', usetex=False)

color_tuple = ['#ae1908', '#ec813b', '#05348b', '#9acdc4', '#00FF00', '#0000FF']
n_vector = [10, 13, 17, 23, 31, 42, 57, 78, 106, 145, 198, 271, 400]
fixed_m_train = 20

# 初始化存储矩阵
train_mse_matrix = np.zeros((len(n_vector), 6))
test_mse_matrix = np.zeros((len(n_vector), 6))
lip_matrix = np.zeros((len(n_vector), 6))

print("Loading data for visualization...")

for i, n_train in enumerate(n_vector):
    results = []
    for seed in range(50):
        file_path = f"./perf/n{n_train}m{fixed_m_train}s{seed}.csv"
        if os.path.exists(file_path):
            try:
                data = pd.read_csv(file_path)
                results.append(data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            # print(f"Missing: {file_path}") # Optional: reduce noise
            pass

    if results:
        all_results = pd.concat(results, ignore_index=True)
        # 按网络ID分组计算均值
        for net_id in range(6):
            net_data = all_results[all_results['net_id'] == net_id]
            if not net_data.empty:
                train_mse_matrix[i, net_id] = net_data['train_loss'].mean()
                test_mse_matrix[i, net_id] = net_data['test_loss'].mean()
                lip_matrix[i, net_id] = net_data['lipschitz'].mean()
    else:
        print(f"No valid data for n={n_train}")

# 绘图设置
lines = ['solid']*4 + ['dashed']*2
marker = ['o', 's', '^', 'v', 'D', 'x']
model_name = ['L=2, W=50','L=3, W=100', 'L=4, W=200', 'L=5, W=400', 'L=6, W=600', 'L=6, W=800']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.95, wspace=0.3)

# Plot Training Error
for i in range(6):
    ax1.plot(n_vector, train_mse_matrix[:, i], color=color_tuple[i], linestyle=lines[i],
             label=model_name[i], marker=marker[i])
ax1.set_ylabel(r"$\mathtt{train-mse}$", fontsize=14)
ax1.set_xlabel(r"Training Set Size $n$", fontsize=14)
ax1.set_title(f"Training Error (m={fixed_m_train})", fontsize=14)
ax1.legend()

# Plot Test Error
for i in range(6):
    ax2.plot(n_vector, test_mse_matrix[:, i], color=color_tuple[i], linestyle=lines[i],
             label=model_name[i], marker=marker[i])
ax2.set_ylabel(r"$\mathtt{test-mse}$", fontsize=14)
ax2.set_xlabel(r"Training Set Size $n$", fontsize=14)
ax2.set_title(f"Test Error (m={fixed_m_train})", fontsize=14)
ax2.legend()

# Plot Lipschitz Constant
log_lip = np.log10(lip_matrix + 1e-10) # 加上 epsilon 防止 log(0)
for i in range(6):
    ax3.plot(n_vector, log_lip[:, i], color=color_tuple[i], linestyle=lines[i],
             label=model_name[i], marker=marker[i])
ax3.set_ylabel(r"Lipschitz constant $log_{10}(L)$", fontsize=14)
ax3.set_xlabel(r"Training Set Size $n$", fontsize=14)
ax3.set_title(f"Lipschitz Constant (m={fixed_m_train})", fontsize=14)
ax3.set_xscale("log")
ax3.legend()

save_path = './Simulation/Case1_R_Adapted'
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, 'DenSimu_Result.png'), dpi=300, bbox_inches='tight')
print(f"Figure saved to {save_path}/DenSimu_Result.png")
# plt.show()