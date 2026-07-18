import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.grating_opt.utils import calc_crit_ne


def plot_reward_from_csv(csv_filepath: str, output_filepath: str, prototype_csv_filepath: str = None):
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

    # Reference line for the prototype's reward, if available
    if prototype_csv_filepath is not None and os.path.exists(prototype_csv_filepath):
        prototype_reward = pd.read_csv(prototype_csv_filepath)['reward'].iloc[0]
        ax.axhline(y=prototype_reward, color='r', linestyle=':', linewidth=2, label='Prototype Reward')

    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Optimization Progress', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
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


def plot_pareto_frontier(results_path: str, output_filepath: str, wavelength: float, de_modes: int | list[int] = 1, prototype_dir: str = None):
    """
    Plot normalized electron density against a diffraction-efficiency metric,
    producing one figure per entry in de_modes.

    Each de_mode selects the DE metric plotted on the y-axis:
      1 - DE at the central wavelength (default)
      2 - Peak DE across all wavelengths
      3 - Average DE across all wavelengths

    When de_modes contains more than one entry, output_filepath is used as a
    template: the mode is inserted before the file extension (e.g.
    "pareto.png" -> "pareto_mode1.png", "pareto_mode2.png", ...).

    :return: List of filepaths written, one per de_mode.
    :rtype: list[str]
    """
    if isinstance(de_modes, int):
        de_modes = [de_modes]

    output_filepaths = []
    for de_mode in de_modes:
        if len(de_modes) > 1:
            base, ext = os.path.splitext(output_filepath)
            mode_output_filepath = f"{base}_mode{de_mode}{ext}"
        else:
            mode_output_filepath = output_filepath

        _plot_single_pareto_frontier(
            results_path, mode_output_filepath, wavelength, de_mode, prototype_dir
        )
        output_filepaths.append(mode_output_filepath)

    return output_filepaths


def _plot_single_pareto_frontier(results_path: str, output_filepath: str, wavelength: float, de_mode: int, prototype_dir: str = None):
    if de_mode not in (1, 2, 3):
        raise ValueError(f"de_mode must be 1, 2, or 3, got {de_mode}")

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

                if de_mode == 1:
                    DE_value = df_de.loc[
                        (df_de['Wavelength_um'] - wavelength).abs().idxmin(),
                        'DE_reflected_m1'
                    ]
                elif de_mode == 2:
                    DE_value = df_de['DE_reflected_m1'].max()
                else:
                    DE_value = df_de['DE_reflected_m1'].mean()

                trial_indices.append(trial_idx)
                norm_ne_values.append(norm_ne)
                DE_values.append(DE_value)

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

    fig, ax = plt.subplots(figsize=(7, 6))

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

    # Reference point for the prototype, if available
    if prototype_dir is not None:
        proto_fdtd_path = os.path.join(prototype_dir, "FDTD_result.csv")
        proto_de_path = os.path.join(prototype_dir, "DE_vs_wavelength.csv")
    if prototype_dir is not None and os.path.exists(proto_fdtd_path) and os.path.exists(proto_de_path):
        proto_ne_peak = pd.read_csv(proto_fdtd_path)["ne_peak"].values[0]
        proto_norm_ne = proto_ne_peak / crit_ne
        proto_de_df = pd.read_csv(proto_de_path)

        if de_mode == 1:
            proto_DE = proto_de_df.loc[
                (proto_de_df['Wavelength_um'] - wavelength).abs().idxmin(),
                'DE_reflected_m1'
            ]
        elif de_mode == 2:
            proto_DE = proto_de_df['DE_reflected_m1'].max()
        else:
            proto_DE = proto_de_df['DE_reflected_m1'].mean()

        ax.scatter(
            proto_norm_ne, proto_DE, s=180, marker="*", color="red",
            edgecolors="black", linewidths=1.0, zorder=5, label="Prototype",
        )

    de_mode_title = {
        1: f"DE at Central Wavelength {wavelength/1000:.2f} $\\mu$m",
        2: "Peak DE",
        3: "Average DE",
    }[de_mode]
    de_mode_ylabel = {
        1: "Diffraction Efficiency (central wavelength)",
        2: "Diffraction Efficiency (peak)",
        3: "Diffraction Efficiency (average)",
    }[de_mode]

    ax.set_title(f"Pareto Frontier: {de_mode_title}", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"Normalized $n_e$ ($n_{e,\mathrm{peak}} / n_{e,\mathrm{crit}}$)")
    ax.set_ylabel(de_mode_ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)

    # split legend: general items first, then trial labels
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(
    #     handles, labels, frameon=False, loc="center left",
    #     bbox_to_anchor=(1.02, 0.5), fontsize=9,
    # )
    ax.legend(
        handles, labels, frameon=False, loc="lower right", fontsize=9,
    )

    fig.tight_layout()
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_norm_ne(results_path: str, output_filepath: str, wavelength: float, prototype_dir: str = None):
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
    trial_indices = np.array([p[0] for p in sorted_pairs])
    norm_ne_values = np.array([p[1] for p in sorted_pairs])

    if len(norm_ne_values) == 0:
        raise ValueError(f"No valid trial results found in {results_path}")

    # --- styling ---
    plt.rcParams.update({
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        trial_indices, norm_ne_values,
        s=60, c="#4c72b0", alpha=0.8, edgecolors="white", linewidths=0.5,
        label="Trials", zorder=3,
    )

    ax.axhline(
        y=1, color="#444444", linestyle="--", linewidth=1.5, zorder=2,
        label="Critical density",
    )

    # Reference line for the prototype's normalized ne, if available
    if prototype_dir is not None:
        prototype_fdtd_path = os.path.join(prototype_dir, "FDTD_result.csv")
        if os.path.exists(prototype_fdtd_path):
            prototype_ne_peak = pd.read_csv(prototype_fdtd_path)["ne_peak"].values[0]
            prototype_norm_ne = prototype_ne_peak / crit_ne
            ax.axhline(
                y=prototype_norm_ne, color="red", linestyle=":", linewidth=2,
                zorder=2, label="Prototype",
            )

    ax.set_title("Normalized Electron Density vs Trial Index", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trial Index")
    ax.set_ylabel(r"Normalized $n_e$ ($n_{e,\mathrm{peak}} / n_{e,\mathrm{crit}}$)")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax.legend(frameon=False, loc="upper left", fontsize=9)

    fig.tight_layout()
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_actual(
        predicted: list,
        actual: list,
        output_filepath: str,
        predicted_sem: list = None,
        metric_name: str = "Reward",
):
    """
    Scatter surrogate-predicted merit function values against actual
    (simulated) values, with a y=x reference line marking a perfect
    surrogate. Entries where `actual` is None (e.g. a timed-out simulation)
    are dropped before plotting.
    """
    predicted_arr = np.asarray(predicted, dtype=float)
    actual_arr = np.array([np.nan if v is None else v for v in actual], dtype=float)

    valid = ~(np.isnan(predicted_arr) | np.isnan(actual_arr))
    predicted_arr = predicted_arr[valid]
    actual_arr = actual_arr[valid]

    if predicted_sem is not None:
        predicted_sem = np.asarray(predicted_sem, dtype=float)[valid]

    if len(predicted_arr) == 0:
        raise ValueError("No completed trials with both predicted and actual values to plot.")

    # --- styling ---
    plt.rcParams.update({
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7, 6))

    if predicted_sem is not None:
        ax.errorbar(
            actual_arr, predicted_arr, yerr=predicted_sem,
            fmt="o", ms=6, color="#4c72b0", ecolor="#4c72b0", alpha=0.7,
            elinewidth=1, capsize=3, zorder=3, label="Sampled parameter sets",
        )
    else:
        ax.scatter(
            actual_arr, predicted_arr,
            s=60, c="#4c72b0", alpha=0.7, edgecolors="white", linewidths=0.5,
            zorder=3, label="Sampled parameter sets",
        )

    lo = min(actual_arr.min(), predicted_arr.min())
    hi = max(actual_arr.max(), predicted_arr.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lims = (lo - pad, hi + pad)

    ax.plot(
        lims, lims, color="#444444", linestyle="--", linewidth=1.5,
        zorder=2, label="Perfect surrogate (y = x)",
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    if len(predicted_arr) > 1:
        corr = np.corrcoef(actual_arr, predicted_arr)[0, 1]
        title = f"Surrogate Predicted vs. Actual {metric_name} (r = {corr:.3f})"
    else:
        title = f"Surrogate Predicted vs. Actual {metric_name}"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Actual {metric_name} (simulation)")
    ax.set_ylabel(f"Predicted {metric_name} (surrogate)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    fig.tight_layout()
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def update_ne_peak_csv(wavelength: float, results_path: str, data_root: str, csv_filename: str = "ne_peak_data.csv"):
    """
    Scan every trial's FDTD_result.csv under results_path and (re)write a CSV
    in data_root/csvs recording each trial's raw ne_peak and ne_peak normalized
    by crit_ne. Existing rows for trials found on disk are refreshed in place;
    rows for trials not found on disk are left untouched.
    """
    crit_ne = calc_crit_ne(wavelength)

    rows = []
    for entry in os.listdir(results_path):
        trial_dir = os.path.join(results_path, entry)
        if os.path.isdir(trial_dir) and entry.isdigit():
            fdtd_path = os.path.join(trial_dir, "FDTD_result.csv")
            if os.path.exists(fdtd_path):
                ne_peak = pd.read_csv(fdtd_path)["ne_peak"].values[0]
                rows.append({
                    "trial_index": int(entry),
                    "ne_peak": ne_peak,
                    "norm_ne": ne_peak / crit_ne,
                })

    if not rows:
        raise ValueError(f"No FDTD_result.csv files found under {results_path}")

    new_df = pd.DataFrame(rows)

    full_path = Path(data_root) / "csvs" / csv_filename
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if full_path.exists():
        existing_df = pd.read_csv(full_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset="trial_index", keep="last")
    else:
        combined_df = new_df

    combined_df = combined_df.sort_values("trial_index").reset_index(drop=True)
    combined_df.to_csv(full_path, index=False)


def plot_DE_vs_wavelength(
        csv_filepath: str,
        trial_indcs: list,
        output_filepath: str,
        results_path: str,
        wavelength: float,
        include_prototype: bool = False,
        prototype_dir: str = None,
        prototype_csv_filepath: str = None,
):
    """
    :param wavelength: Central wavelength in nm, used to normalize each
        trial's peak electron density (n_e,peak / n_e,crit), included in each
        curve's legend entry.
    """
    df_main = pd.read_csv(csv_filepath)
    crit_ne = calc_crit_ne(wavelength)

    fig, ax = plt.subplots(figsize=(9, 6))

    for trial_idx in trial_indcs:
        trial_row = df_main[df_main['trial_index'] == trial_idx]
        if trial_row.empty:
            continue
        reward = trial_row['reward'].values[0]

        trial_csv_path = os.path.join(results_path, f"{trial_idx}", "DE_vs_wavelength.csv")
        df = pd.read_csv(trial_csv_path)

        wavelengths = df['Wavelength_um'].values
        diffraction_efficiency = df['DE_reflected_m1'].values

        label = f'Trial {trial_idx} (reward={reward:.3f}'
        fdtd_path = os.path.join(results_path, f"{trial_idx}", "FDTD_result.csv")
        if os.path.exists(fdtd_path):
            ne_peak = pd.read_csv(fdtd_path)["ne_peak"].values[0]
            norm_ne = ne_peak / crit_ne
            label += f', norm $n_e$={norm_ne:.3f}'
        label += ')'

        ax.plot(wavelengths, diffraction_efficiency, marker='o', linestyle='-', label=label)

    if include_prototype and prototype_dir is not None:
        proto_de_path = os.path.join(prototype_dir, "DE_vs_wavelength.csv")

        if os.path.exists(proto_de_path):
            label = 'Prototype'
            if prototype_csv_filepath is not None and os.path.exists(prototype_csv_filepath):
                prototype_reward = pd.read_csv(prototype_csv_filepath)['reward'].iloc[0]
                label += f' (reward={prototype_reward:.3f}'
            else:
                label += ' ('

            proto_fdtd_path = os.path.join(prototype_dir, "FDTD_result.csv")
            if os.path.exists(proto_fdtd_path):
                proto_ne_peak = pd.read_csv(proto_fdtd_path)["ne_peak"].values[0]
                proto_norm_ne = proto_ne_peak / crit_ne
                label += f', norm $n_e$={proto_norm_ne:.3f}'
            label += ')'

            df_proto = pd.read_csv(proto_de_path)
            proto_wavelengths = df_proto['Wavelength_um'].values
            proto_diffraction_efficiency = df_proto['DE_reflected_m1'].values

            ax.plot(
                proto_wavelengths, proto_diffraction_efficiency, marker='*', markersize=12,
                linestyle='--', color='red', linewidth=2, zorder=5, label=label,
            )

    ax.set_title('Diffraction Efficiency vs Wavelength')
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Diffraction Efficiency')
    ax.grid(True)

    # Legend lives outside the axes, in the right margin, so it never
    # overlaps the curves.
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    fig.subplots_adjust(left=0.09, right=0.6, top=0.93, bottom=0.09)

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_filepath)
    plt.close(fig)