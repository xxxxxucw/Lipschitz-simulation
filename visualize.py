import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_benchmark_results():
    # 获取所有的性能监控 CSV 文件
    csv_files = glob.glob('./perf/*.csv')
    if not csv_files:
        print("未在 ./perf/ 目录下找到CSV文件，请确认训练跑完！")
        return

    df_list = [pd.read_csv(f) for f in csv_files]
    df_all = pd.concat(df_list, ignore_index=True)

    # 清理掉之前错误写入的 0.0 脏数据，防止影响新旧数据的混合绘制
    df_clean = df_all[df_all['lipschitz'] > 0.0]
    
    if df_clean.empty:
        print("错误：CSV 中的有效数据全为 0.0，请检查数据是否存在。")
        return

    m_target = df_clean['m_train'].max() 
    df_m = df_clean[df_clean['m_train'] == m_target]
    # 对同一个网络在同一样本量下跨 seed 取平均值
    grouped = df_m.groupby(['net_id', 'n_train']).mean().reset_index()

    # 初始化1行3列的画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 按照你论文的代码定义不同的网络架构标识
    net_labels = {
        0: "L=2, W=50",
        1: "L=3, W=100",
        2: "L=4, W=200",
        3: "L=5, W=400",
        4: "L=6, W=600",
        5: "L=6, W=800"
    }
    
    markers = ['o', 's', '^', 'v', 'D', 'x']
    colors = ['#8B0000', '#D2691E', '#4682B4', '#5F9EA0', '#32CD32', '#0000FF'] 

    for net_id in sorted(grouped['net_id'].unique()):
        data = grouped[grouped['net_id'] == net_id].sort_values('n_train')
        label = net_labels.get(net_id, f"Net {net_id}")
        mkr = markers[int(net_id) % len(markers)]
        clr = colors[int(net_id) % len(colors)]
        
        axes[0].plot(data['n_train'], data['train_loss'], marker=mkr, color=clr, label=label, linestyle='-', linewidth=1.5)
        axes[1].plot(data['n_train'], data['test_loss'], marker=mkr, color=clr, label=label, linestyle='-', linewidth=1.5)
        axes[2].plot(data['n_train'], np.log10(data['lipschitz']), marker=mkr, color=clr, label=label, linestyle='-', linewidth=1.5)

    # 统一格式化子图
    for idx, (ax, title, ylabel) in enumerate(zip(
        axes, 
        [f'Training Error (m={m_target})', f'Test Error (m={m_target})', f'Lipschitz Constant (m={m_target})'],
        ['train - mse', 'test - mse', r'Lipschitz constant $log_{10}(L)$']
    )):
        ax.set_title(title)
        ax.set_xlabel('Training Set Size n')
        ax.set_ylabel(ylabel)
        ax.legend()

    plt.tight_layout()
    plt.savefig('results_plot.png', dpi=300)
    print("✅ 绘图完成！图片已保存为 results_plot.png")

if __name__ == '__main__':
    plot_benchmark_results()