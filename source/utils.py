# Collection of useful functions used across all examples
from matplotlib import rcParams

def set_plot_params():
    # Set commonly used plot parameters to desired values
    # Plotting parameters
    rcParams.update({'figure.figsize' : (8,6),
                    'font.size': 16,
                    'figure.titlesize' : 18,
                    'axes.labelsize': 18,
                    'xtick.labelsize': 15,
                    'ytick.labelsize': 15,
                    'legend.fontsize': 15})    
    