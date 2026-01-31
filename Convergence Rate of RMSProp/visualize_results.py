import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Settings compatible with your previous visualize style
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=15)

COLOR_TUPLE = ['#ae1908', '#05348b'] # Red for ResNet18, Blue for others
LINES = ['solid', 'dashed']
MARKERS = ['o', 's']
MODEL_NAMES = ['ResNet18', 'ResNet50']

def visualize(csv_path='./resnet_experiment_results.csv'):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Run train_experiment.py first.")
        return

    # Read Data
    df = pd.read_csv(csv_path)
    
    # Group by n_train and net_id to get means
    # We want to plot X-axis: n_train
    n_vector = sorted(df['n_train'].unique())
    net_ids = sorted(df['net_id'].unique())
    
    # Prepare matrices for plotting
    # Rows: indices of n_vector, Cols: net_ids
    train_loss_mat = np.zeros((len(n_vector), len(net_ids)))
    test_err_mat = np.zeros((len(n_vector), len(net_ids)))
    lip_matrix = np.zeros((len(n_vector), len(net_ids)))
    
    # Fill Data
    for i, n in enumerate(n_vector):
        for j, net_id in enumerate(net_ids):
            subset = df[(df['n_train'] == n) & (df['net_id'] == net_id)]
            if not subset.empty:
                train_loss_mat[i, j] = subset['train_loss'].mean()
                test_err_mat[i, j] = subset['test_loss'].mean() # This is Test Error Rate
                lip_matrix[i, j] = subset['lipschitz'].mean()
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.95, wspace=0.3)

    # 1. Training Error/Loss
    for j, net_id in enumerate(net_ids):
        ax1.plot(n_vector, train_loss_mat[:, j], color=COLOR_TUPLE[j], linestyle=LINES[j],
                 label=MODEL_NAMES[net_id], marker=MARKERS[j])
    ax1.set_ylabel(r"$\mathtt{train-loss}$", fontsize=14)
    ax1.set_xlabel(r"Training Set Size $n$", fontsize=14)
    ax1.set_title(f"Training Loss", fontsize=14)
    ax1.legend()
    ax1.set_xscale("log") # N varies widely, log scale is better

    # 2. Test Error
    for j, net_id in enumerate(net_ids):
        ax2.plot(n_vector, test_err_mat[:, j], color=COLOR_TUPLE[j], linestyle=LINES[j],
                 label=MODEL_NAMES[net_id], marker=MARKERS[j])
    ax2.set_ylabel(r"$\mathtt{test-error}$", fontsize=14)
    ax2.set_xlabel(r"Training Set Size $n$", fontsize=14)
    ax2.set_title(f"Test Error", fontsize=14)
    ax2.legend()
    ax2.set_xscale("log")

    # 3. Lipschitz Constant
    log_lip = np.log10(lip_matrix)
    for j, net_id in enumerate(net_ids):
        ax3.plot(n_vector, log_lip[:, j], color=COLOR_TUPLE[j], linestyle=LINES[j],
                 label=MODEL_NAMES[net_id], marker=MARKERS[j])
    ax3.set_ylabel(r"Lipschitz constant $log_{10}(L)$", fontsize=14)
    ax3.set_xlabel(r"Training Set Size $n$", fontsize=14)
    ax3.set_title(f"Lipschitz Constant", fontsize=14)
    ax3.set_xscale("log")
    ax3.legend()

    save_path = './resnet_experiment_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    visualize()