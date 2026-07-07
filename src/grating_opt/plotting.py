import os
import matplotlib.pyplot as plt
import pandas as pd

from src.grating_opt.utils import calc_crit_ne


def plot_reward_from_csv(csv_filepath: str, output_filepath: str):
    # 1. Load the CSV data
    df = pd.read_csv(csv_filepath)
    
    # Ensure the trials are ordered by trial_index to compute a proper running history
    df = df.sort_values(by='trial_index').reset_index(drop=True)
    
    # 2. Extract all reward values
    all_values = df['reward'].tolist()
    
    # 3. Record the best value trial-by-trial (running maximum)
    best_values = []
    current_best = float('-inf')  # Start with negative infinity for maximization
    
    for val in all_values:
        # If the objective is minimization instead, change the comparison to < and start at float('inf')
        if val > current_best:
            current_best = val
        best_values.append(current_best)
        
    # 4. Plot using the logic provided in plot_reward
    fig, ax = plt.subplots(figsize=(12, 6))
    trials_range = range(0, len(best_values))

    # Scatter plot of all trial values
    ax.scatter(trials_range, all_values, c='b', alpha=0.6, s=50, label='Trial Reward')

    # Plot best values over trials (convergence plot)
    ax.plot(trials_range, best_values, 'b-', linewidth=2, label='Best Value Found')

    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Optimization Progress', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_pareto_frontier(x, y, minimize_x=True, maximize_y=True):
    """
    Return indices of Pareto-optimal points, sorted along x.

    Default assumption: lower x (normalized ne) and higher y (DE) are both
    "better", i.e. we want to minimize x and maximize y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    sign_x = 1 if minimize_x else -1
    sign_y = -1 if maximize_y else 1

    order = np.argsort(sign_x * x)
    pareto_idx = []
    best_y = np.inf
    for i in order:
        if sign_y * y[i] < best_y:
            pareto_idx.append(i)
            best_y = sign_y * y[i]
    return np.array(pareto_idx)


def plot_pareto_frontier(results_path: str, output_filepath: str, wavelength: float):
    trial_indices = []
    norm_ne_values = []
    DE_values = []
    crit_ne = calc_crit_ne(wavelength)

    for entry in os.listdir(results_path):
        trial_dir = os.path.join(results_path, entry)
        if os.path.isdir(trial_dir) and entry.isdigit():
            trial_idx = int(entry)
            fdtd_path = os.path.join(trial_dir, "FDTD_result.csv")
            de_path = os.path.join(trial_dir, "DE_vs_wavelength.csv")
            if os.path.exists(fdtd_path) and os.path.exists(de_path):
                df_fdtd = pd.read_csv(fdtd_path)
                df_de = pd.read_csv(de_path)
                ne_peak = df_fdtd["ne_peak"].values[0]
                norm_ne = ne_peak / crit_ne
                DE_at_wavelength = df_de.loc[
                    (df_de['Wavelength_um'] - wavelength).abs().idxmin(),
                    'DE_reflected_m1'
                ]
                trial_indices.append(trial_idx)
                norm_ne_values.append(norm_ne)
                DE_values.append(DE_at_wavelength)

    norm_ne_values = np.array(norm_ne_values)
    DE_values = np.array(DE_values)
    trial_indices = np.array(trial_indices)

    if len(norm_ne_values) == 0:
        raise ValueError(f"No valid trial results found in {results_path}")

    pareto_idx = compute_pareto_frontier(norm_ne_values, DE_values)
    is_pareto = np.zeros(len(norm_ne_values), dtype=bool)
    is_pareto[pareto_idx] = True

    # --- styling ---
    plt.rcParams.update({
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # dominated points
    ax.scatter(
        norm_ne_values[~is_pareto], DE_values[~is_pareto],
        s=60, c="#9aa5b1", alpha=0.6, edgecolors="white", linewidths=0.5,
        label="Dominated trials", zorder=2,
    )

    # pareto-optimal points, connected by a line
    order = np.argsort(norm_ne_values[pareto_idx])
    pareto_x = norm_ne_values[pareto_idx][order]
    pareto_y = DE_values[pareto_idx][order]
    pareto_trials = trial_indices[pareto_idx][order]

    ax.plot(
        pareto_x, pareto_y,
        color="#444444", linewidth=1.5, linestyle="--", zorder=3,
        label="Pareto frontier",
    )

    # distinct color per frontier point
    n_pareto = len(pareto_x)
    cmap = plt.get_cmap("tab20" if n_pareto > 10 else "tab10")
    colors = [cmap(i % cmap.N) for i in range(n_pareto)]

    for xi, yi, ti, c in zip(pareto_x, pareto_y, pareto_trials, colors):
        ax.scatter(
            xi, yi, s=100, color=c, edgecolors="black", linewidths=0.8,
            zorder=4, label=f"Trial {ti}",
        )

    ax.set_title(f"Pareto Frontier at Wavelength {wavelength:.3f} $\\mu$m", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"Normalized $n_e$ ($n_{e,\mathrm{peak}} / n_{e,\mathrm{crit}}$)")
    ax.set_ylabel("Diffraction Efficiency (central wavelength)")
    ax.grid(True, linestyle=":", alpha=0.5)

    # split legend: general items first, then trial labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, frameon=False, loc="center left",
        bbox_to_anchor=(1.02, 0.5), fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_norm_ne(results_path: str, output_filepath: str, wavelength: float):
    trial_indices = []
    norm_ne_values = []
    crit_ne = calc_crit_ne(wavelength)

    for entry in os.listdir(results_path):
        trial_dir = os.path.join(results_path, entry)
        if os.path.isdir(trial_dir) and entry.isdigit():
            trial_idx = int(entry)
            fdtd_path = os.path.join(trial_dir, "FDTD_result.csv")
            if os.path.exists(fdtd_path):
                df = pd.read_csv(fdtd_path)
                ne_peak = df["ne_peak"].values[0]
                norm_ne = ne_peak / crit_ne
                trial_indices.append(trial_idx)
                norm_ne_values.append(norm_ne)

    sorted_pairs = sorted(zip(trial_indices, norm_ne_values), key=lambda x: x[0])
    trial_indices = [p[0] for p in sorted_pairs]
    norm_ne_values = [p[1] for p in sorted_pairs]

    plt.figure(figsize=(10, 6))
    plt.scatter(trial_indices, norm_ne_values)
    plt.axhline(y=1, linestyle=':', color='r')

    plt.title('Normalized Electron Density vs Trial Index')
    plt.xlabel('Trial Index')
    plt.ylabel('Normalized ne (ne_peak / crit_ne)')
    plt.grid(True)

    plt.savefig(output_filepath)
    plt.close()


def plot_DE_vs_wavelength(csv_filepath: str, trial_indcs: list, output_filepath: str, results_path: str):
    df_main = pd.read_csv(csv_filepath)
    
    plt.figure(figsize=(10, 6))
    
    for trial_idx in trial_indcs:
        trial_row = df_main[df_main['trial_index'] == trial_idx]
        if trial_row.empty:
            continue
        reward = trial_row['reward'].values[0]
        
        trial_csv_path = os.path.join(results_path, f"{trial_idx}", "DE_vs_wavelength.csv")
        df = pd.read_csv(trial_csv_path)
        
        wavelength = df['Wavelength_um'].values
        diffraction_efficiency = df['DE_reflected_m1'].values
        
        plt.plot(wavelength, diffraction_efficiency, marker='o', linestyle='-', label=f'Trial {trial_idx}: {reward}')
    
    plt.title('Diffraction Efficiency vs Wavelength')
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Diffraction Efficiency')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_filepath)
    plt.close()