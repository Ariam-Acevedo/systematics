import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re


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


def load_data_ss_up(file_path, num_events, data_type, threshold, varied_variable):
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
        data=np.concatenate([event[data_type] for event in data_files[:num_events] if event[varied_variable] > threshold])

    except:
        data=np.array([event[data_type] for event in data_files[:num_events] if event[varied_variable] > threshold])

    return data

def load_data_ss_down(file_path, num_events, data_type, threshold, varied_variable):
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
        data=np.concatenate([event[data_type] for event in data_files[:num_events] if event[varied_variable] < threshold])

    except:
        data=np.array([event[data_type] for event in data_files[:num_events] if event[varied_variable] < threshold])

    return data
 
def snow_storm_calculation(
    varied_variable,
    num_events,
    range_e,
    n_bins,
    data_type,
    density,
    std,
    sigma,
    mean,
    output_dir="plots",
    plot_name=None,
    ext="png",
):
    """
    Calculate the snowstorm effect for a given variable.
    """

    if std == 0:
        raise ValueError("std must be non-zero")

    # files_nom=load_data('data_files_nom.pkl', num_events, data_type)
    files_nom_ss=load_data('energy_charge_events_streamed_billy.pkl', num_events, data_type)
    print(f"Loaded {len(files_nom_ss)} events from energy_charge_events_streamed_billy.pkl")
    varied_1p=load_data(f'{varied_variable}_1_sig_up.pkl', num_events, data_type)
    varied_1m=load_data(f'{varied_variable}_1_sig_dn.pkl', num_events, data_type)

    # Create histograms for each dataset
    # hist_nom, bin_edges = np.histogram(files_nom, bins=n_bins, range=range_e, density=density)
    hist_nom_ss, bin_edges = np.histogram(files_nom_ss, bins=n_bins, range=range_e, density=density)
    hist_varied_1p, _ = np.histogram(varied_1p, bins=n_bins, range=range_e, density=density)
    hist_varied_1m, _ = np.histogram(varied_1m, bins=n_bins, range=range_e, density=density)


    # Calculate the snowstorm variations
    snowstorm_val_up=load_data_ss_up('energy_charge_events_streamed_billy.pkl', num_events, data_type, mean, varied_variable)
    snowstorm_val_down=load_data_ss_down('energy_charge_events_streamed_billy.pkl', num_events, data_type, mean, varied_variable)

    print(f"Loaded {len(snowstorm_val_up)} events for snowstorm up and {len(snowstorm_val_down)} events for snowstorm down")
    a_val_down, bin_edges_down = np.histogram(snowstorm_val_down, bins=n_bins, range=range_e, density=density)
    a_val_up, bin_edges_up = np.histogram(snowstorm_val_up, bins=n_bins, range=range_e, density=density)

    grad_a=(np.sqrt(np.pi/2)*(a_val_up-a_val_down)*(1/std))

    snowstorm_up=hist_nom_ss+grad_a*sigma
    snowstorm_down=hist_nom_ss-grad_a*sigma
    





    #initial separation plots
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stairs(a_val_up, bin_edges, label="cut up W_ion > 23.6 MeV")
    ax.stairs(a_val_down, bin_edges, label="cut down W_ion < 23.6 MeV")
    ax.stairs(hist_nom_ss / 2, bin_edges, label="Nominal")
    ax.set_xlabel("Energy per hit [MeV]")
    ax.set_ylabel("Counts")
    ax.legend()

    file_path = save_plot(
        fig,
        output_dir=f"{output_dir}/initial",
        plot_name=plot_name or f"snowstorm_",
        varied_variables=varied_variable,
        data_type=data_type,
        ext=ext,
    )

    #gradient plots
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stairs(grad_a, bin_edges, label="Gradient W_ion")
    ax.set_xlabel("Energy per hit [MeV]")
    ax.set_ylabel("Counts")
    ax.legend()

    file_path = save_plot(
        fig,
        output_dir=f"{output_dir}/gradient",
        plot_name=plot_name or f"gradient_",
        varied_variables=varied_variable,
        data_type=data_type,
        ext=ext,
    )


    #final snowstorm plots
    # Figure with ratio panel
    fig, (ax, axr,ax2) = plt.subplots(
        3, 1, figsize=(10, 15), sharex=True,
        gridspec_kw={"height_ratios": [4, 2,2]}
    )

    ax.stairs(hist_nom_ss, bin_edges, label="Nominal", color="black", linewidth=1)
    ax.stairs(snowstorm_up, bin_edges, label="SS +1sig", color="orange", linewidth=1, linestyle="--")
    ax.stairs(snowstorm_down, bin_edges, label="SS -1sig", color="orange", linewidth=1, linestyle="--")
    ax.stairs(hist_varied_1p, bin_edges, label="Fixed +1sig", color="blue", linewidth=1, linestyle="--")
    ax.stairs(hist_varied_1m, bin_edges, label="Fixed -1sig", color="blue", linewidth=1, linestyle="--")
    ax.set_xlabel("Energy per hit (MeV)")
    ax.set_title("Snowstorm vs fixed 1 sigma variation W_ion")
    ax.set_ylabel("Density")
    ax.legend()

    ratio_up = np.divide(
        snowstorm_up,
        hist_varied_1p,
        out=np.zeros_like(snowstorm_up, dtype=float),
        where=(hist_varied_1p > 0),
    )
    axr.stairs(ratio_up, bin_edges, label="SS +1sig / Fixed +1sig")
    axr.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axr.set_ylim((0.5,1.5))
    axr.set_xlabel("Energy per hit (MeV)")
    axr.set_ylabel("Density")
    axr.legend()

    ratio_down = np.divide(
        snowstorm_down,
        hist_varied_1m,
        out=np.zeros_like(snowstorm_down, dtype=float),
        where=(hist_varied_1m > 0),
    )
    ax2.stairs(ratio_down, bin_edges, label="SS -1sig / Fixed -1sig")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax2.set_ylim((0.5,1.5))
    ax2.set_xlabel("Energy per hit (MeV)")
    ax2.set_ylabel("Density")
    ax2.legend()


    file_path = save_plot(
        fig,
        output_dir=f"{output_dir}/final",
        plot_name=plot_name or f"final__",
        varied_variables=varied_variable,
        data_type=data_type,
        ext=ext,
    )














    return fig, file_path


m_wion=2.36e-05
std_wion=4.682722514901835e-07
m_birksk=0.0486
std_birksk=0.008278352043258438
m_birksa=0.800
std_birksa=0.019968389380744415

snow_storm_calculation(
    varied_variable="w_ion",
    num_events=2250,
    range_e=(0, 29),
    n_bins=35,
    data_type="meanQ",
    density=False,
    std=std_wion,
    sigma=std_wion,
    mean=m_wion,
    output_dir="MeanCharge_plots",
    plot_name=None,
    ext="png",
)


snow_storm_calculation(
    varied_variable="birks_k",
    num_events=2250,
    range_e=(0, 29),
    n_bins=35,
    data_type="meanQ",
    density=False,
    std=std_birksk,
    sigma=std_birksk,
    mean=m_birksk,
    output_dir="MeanCharge_plots",
    plot_name=None,
    ext="png",
)

snow_storm_calculation(
    varied_variable="birks_a",
    num_events=2250,
    range_e=(0, 29),
    n_bins=35,
    data_type="meanQ",
    density=False,
    std=std_birksa,
    sigma=std_birksa,
    mean=m_birksa,
    output_dir="MeanCharge_plots",
    plot_name=None,
    ext="png",
)




