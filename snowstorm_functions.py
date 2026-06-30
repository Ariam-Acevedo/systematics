import numpy as np 
import matplotlib.pyplot as plt
import pickle 
import os
import re
import h5flow


def _safe_token(value):
    """Convert a value into a filesystem-safe token."""
    token = str(value).strip()
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_.-")
    return token or "value"


def build_plot_filename(plot_name, varied_variables=None, data_type=None, ext="png"):
    """Build a filename from a plot name, varied variables, and dtype."""
    parts = [_safe_token(plot_name)]

    if varied_variables is not None:
        if isinstance(varied_variables, dict):
            items = [f"{key}-{value}" for key, value in sorted(varied_variables.items())]
        elif isinstance(varied_variables, (list, tuple, set)):
            items = varied_variables
        else:
            items = [varied_variables]
        parts.extend(_safe_token(item) for item in items)

    if data_type is not None:
        parts.append(_safe_token(f"dtype-{data_type}"))

    suffix = ext.lstrip(".")
    return "_".join(parts) + f".{suffix}"


def save_plot(fig, output_dir, plot_name, varied_variables=None, data_type=None, ext="png", dpi=300):
    """Save a matplotlib figure in a folder with a filename derived from plot inputs."""
    os.makedirs(output_dir, exist_ok=True)
    filename = build_plot_filename(
        plot_name=plot_name,
        varied_variables=varied_variables,
        data_type=data_type,
        ext=ext,
    )
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    return file_path



def load_data(file_path, num_events, data_type):
    """
    Load data from a pickle file.

    Parameters:
    file_path (str): Path to the pickle file.

    Returns:
    data: The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data_files = pickle.load(f)
    try:
        data=np.concatenate([event[data_type] for event in data_files[:num_events]])

    except:
        data=np.array([event[data_type] for event in data_files[:num_events]])

    return data


def load_data_ss_up(file_path, num_events, data_type, threshold):
    """
    Load data from a pickle file.

    Parameters:
    file_path (str): Path to the pickle file.

    Returns:
    data: The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data_files = pickle.load(f)
    try:
        data=np.concatenate([event[data_type] for event in data_files[:num_events] if event[data_type] > threshold])

    except:
        data=np.array([event[data_type] for event in data_files[:num_events] if event[data_type] > threshold])

    return data

def load_data_ss_down(file_path, num_events, data_type, threshold):
    """
    Load data from a pickle file.

    Parameters:
    file_path (str): Path to the pickle file.

    Returns:
    data: The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data_files = pickle.load(f)
    try:
        data=np.concatenate([event[data_type] for event in data_files[:num_events] if event[data_type] < threshold])

    except:
        data=np.array([event[data_type] for event in data_files[:num_events] if event[data_type] < threshold])

    return data
 
def snow_storm_caleculation(
    varied_variable,
    num_events,
    range_e,
    n_bins,
    data_type,
    density,
    std,
    sigma,
    output_dir="plots",
    plot_name=None,
    ext="png",
):
    """
    Calculate the snowstorm effect for a given variable.
    """

    if std == 0:
        raise ValueError("std must be non-zero")

    files_nom=load_data('data_files_nom.pkl', num_events, data_type)
    files_nom_ss=load_data('data_files_nom_ss.pkl', num_events, data_type)
    varied_1p=load_data(f'data_files_{varied_variable}_1p.pkl', num_events, data_type)
    varied_1m=load_data(f'data_files_{varied_variable}_1m.pkl', num_events, data_type)

    # Create histograms for each dataset
    hist_nom, bin_edges = np.histogram(files_nom, bins=n_bins, range=range_e, density=density)
    hist_nom_ss, _ = np.histogram(files_nom_ss, bins=n_bins, range=range_e, density=density)
    hist_varied_1p, _ = np.histogram(varied_1p, bins=n_bins, range=range_e, density=density)
    hist_varied_1m, _ = np.histogram(varied_1m, bins=n_bins, range=range_e, density=density)


    # Calculate the snowstorm variations
    snowstorm_val_up=load_data_ss_up(f'data_files_{varied_variable}_1p.pkl', num_events, data_type, 0)
    snowstorm_val_down=load_data_ss_down(f'data_files_{varied_variable}_1m.pkl', num_events, data_type, 0)

    a_val_down, bin_edges_down = np.histogram(snowstorm_val_down, bins=n_bins, range=range_e, density=density)
    a_val_up, bin_edges_up = np.histogram(snowstorm_val_up, bins=n_bins, range=range_e, density=density)

    grad_a=(np.sqrt(np.pi/2)*(a_val_up-a_val_down)*(1/std))

    snowstorm_up=hist_nom_ss+grad_a*sigma
    snowstorm_down=hist_nom_ss-grad_a*sigma

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(bin_centers, hist_nom, where="mid", label="nominal")
    ax.step(bin_centers, hist_nom_ss, where="mid", label="nominal ss")
    ax.step(bin_centers, hist_varied_1p, where="mid", label=f"{varied_variable} +1p")
    ax.step(bin_centers, hist_varied_1m, where="mid", label=f"{varied_variable} -1m")
    ax.fill_between(bin_centers, snowstorm_down, snowstorm_up, alpha=0.25, label="snowstorm band")
    ax.set_xlabel(str(data_type))
    ax.set_ylabel("Density" if density else "Counts")
    ax.set_title(f"Snowstorm comparison for {varied_variable}")
    ax.legend()
    fig.tight_layout()

    file_path = save_plot(
        fig,
        output_dir=output_dir,
        plot_name=plot_name or f"snowstorm_{varied_variable}",
        varied_variables=varied_variable,
        data_type=data_type,
        ext=ext,
    )

    return fig, file_path



