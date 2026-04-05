# -*- coding: utf-8 -*-
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter

plt.rc('font', size=15)
rc('text', usetex=False)

COLOR_MAP = {
    'mixing bridge': '#05348b',
}
LINESTYLE_MAP = {
    'mixing bridge': 'dashed',
}
MARKER_MAP = {
    'mixing bridge': 's',
}
DISPLAY_NAME = {
    'mixing bridge': 'Mixing bridge',
}


def plot_with_errorbar(ax, x, y, yerr, method_key):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color=COLOR_MAP[method_key],
        linestyle=LINESTYLE_MAP[method_key],
        marker=MARKER_MAP[method_key],
        label=DISPLAY_NAME[method_key],
        capsize=3,
    )


def summarize_metric(df, methods, n_vector, metric_name, log_transform=False):
    means = {m: [] for m in methods}
    ses = {m: [] for m in methods}
    valid_counts = {m: [] for m in methods}

    for method in methods:
        for n in n_vector:
            sub = df[(df['method'].str.lower() == method) & (df['n_train'] == n)]

            if len(sub) == 0:
                means[method].append(np.nan)
                ses[method].append(np.nan)
                valid_counts[method].append(0)
                continue

            values = sub[metric_name].to_numpy(dtype=float)

            if log_transform:
                values = np.where(np.isnan(values), np.nan, np.log10(np.maximum(values, 1e-12)))

            valid_values = values[~np.isnan(values)]
            valid_counts[method].append(len(valid_values))

            if len(valid_values) == 0:
                means[method].append(np.nan)
                ses[method].append(np.nan)
            elif len(valid_values) == 1:
                means[method].append(float(np.mean(valid_values)))
                ses[method].append(0.0)
            else:
                means[method].append(float(np.mean(valid_values)))
                ses[method].append(float(np.std(valid_values, ddof=1) / np.sqrt(len(valid_values))))

    return means, ses, valid_counts


def force_linear_integer_xaxis(ax, n_vector):
    ax.set_xscale("linear")
    ax.set_xticks(n_vector)
    ax.set_xticklabels([str(n) for n in n_vector])

    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_visible(False)


def print_valid_count_table(valid_counts, methods, n_vector, title):
    print(f"\nValid count table for {title}:")
    header = "method".ljust(22) + "".join([f"{n:>8}" for n in n_vector])
    print(header)
    for method in methods:
        row = method.ljust(22) + "".join([f"{c:>8}" for c in valid_counts[method]])
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf_csv", type=str, default="./mtdr_outputs/perf/results.csv")
    parser.add_argument("--save_path", type=str, default="./mtdr_outputs/figures/mTDR-train-test-lipschitz.png")
    parser.add_argument("--n_list", type=int, nargs="+", default=[500, 1000, 2000, 3000, 5000])
    args = parser.parse_args()

    df = pd.read_csv(args.perf_csv)
    df["method"] = df["method"].str.lower().str.strip()

    methods = ['mixing bridge']
    n_vector = args.n_list

    print("\nRecord count by method and n_train:")
    count_table = (
        df[df["method"] == "mixing bridge"]
        .groupby(["method", "n_train"])
        .size()
        .unstack(fill_value=0)
    )
    print(count_table)

    train_means, train_ses, train_counts = summarize_metric(
        df, methods, n_vector, "train_error", log_transform=False
    )
    test_means, test_ses, test_counts = summarize_metric(
        df, methods, n_vector, "test_error", log_transform=False
    )
    lip_means, lip_ses, lip_counts = summarize_metric(
        df, methods, n_vector, "lipschitz", log_transform=True
    )

    print_valid_count_table(train_counts, methods, n_vector, "train_error")
    print_valid_count_table(test_counts, methods, n_vector, "test_error")
    print_valid_count_table(lip_counts, methods, n_vector, "lipschitz")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.08, right=0.98, wspace=0.45)

    method = "mixing bridge"
    plot_with_errorbar(ax1, n_vector, train_means[method], train_ses[method], method)
    plot_with_errorbar(ax2, n_vector, test_means[method], test_ses[method], method)
    plot_with_errorbar(ax3, n_vector, lip_means[method], lip_ses[method], method)

    force_linear_integer_xaxis(ax1, n_vector)
    force_linear_integer_xaxis(ax2, n_vector)
    force_linear_integer_xaxis(ax3, n_vector)

    ax1.set_ylabel(r"train-error", fontsize=14)
    ax1.set_xlabel(r"Training Set Size $n$", fontsize=14)
    ax1.set_title("Training Error", fontsize=16)
    ax1.legend()

    ax2.set_ylabel(r"test-error", fontsize=14)
    ax2.set_xlabel(r"Training Set Size $n$", fontsize=14)
    ax2.set_title("Test Error", fontsize=16)
    ax2.legend()

    ax3.set_ylabel(r"Lipschitz constant $log_{10}(L)$", fontsize=14)
    ax3.set_xlabel(r"Training Set Size $n$", fontsize=14)
    ax3.set_title("Lipschitz Constant", fontsize=16)
    ax3.legend()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.savefig(args.save_path, dpi=300, bbox_inches='tight', format='png')
    plt.show()
    print(f"Saved figure to: {args.save_path}")


if __name__ == "__main__":
    main()