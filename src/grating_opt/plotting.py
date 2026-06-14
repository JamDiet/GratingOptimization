from ax.api.client import Client
import matplotlib.pyplot as plt
import pandas as pd

# def plot_frontier(client: Client, filepath: str):
#     frontier = client.get_pareto_frontier()

#     peak_ne = []
#     diff_eff = []

#     # Retrieve objective metrics
#     for _, metrics, _, _ in frontier:
#         peak_ne.append(metrics["peak_ne"][0])
#         diff_eff.append(metrics["diff_eff"][0])
    
#     # Sort points by peak_ne to ensure a clean frontier line
#     sorted_indices = sorted(range(len(peak_ne)), key=lambda k: peak_ne[k])
#     peak_ne = [peak_ne[i] for i in sorted_indices]
#     diff_eff = [diff_eff[i] for i in sorted_indices]
    
#     # Initialize the plot
#     plt.figure(figsize=(10, 6))
    
#     # Plotting the data
#     plt.plot(peak_ne, diff_eff, marker='o', linestyle='--', color='b', label='Pareto Frontier')
    
#     # Apply log scale to the x-axis (Peak Electron Density)
#     plt.xscale('log')
    
#     # Formatting and Labels
#     plt.title('Pareto Frontier: Peak Electron Density vs. Diffraction Efficiency')
#     plt.xlabel('Peak Electron Density (cm^-3)')
#     plt.ylabel('Diffraction Efficiency')
#     plt.grid(True, which="both", ls="-", alpha=0.5)
#     plt.legend()
    
#     plt.savefig(filepath)


# def plot_reward(best_values: list, all_values: list, filepath: str):
#     fig, ax = plt.subplots(figsize=(12, 6))

#     trials_range = range(1, len(best_values) + 1)

#     # Plot best values over trials (convergence plot)
#     ax.plot(trials_range, best_values, 'b-', linewidth=2, label='Best value found')

#     # Scatter plot of all trial values
#     ax.scatter(trials_range, all_values, c='b', alpha=0.6, s=50)

#     ax.set_xlabel('Trial', fontsize=12)
#     ax.set_ylabel('Reward', fontsize=12)
#     ax.set_title('Optimization Progress', fontsize=14)
#     ax.legend(loc='upper right')
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(filepath)


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
    trials_range = range(1, len(best_values) + 1)

    # Scatter plot of all trial values
    ax.scatter(trials_range, all_values, c='b', alpha=0.6, s=50, label='Trial Reward')

    # Plot best values over trials (convergence plot)
    ax.plot(trials_range, best_values, 'b-', linewidth=2, label='Best Value Found')

    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Optimization Progress', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()