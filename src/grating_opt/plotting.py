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