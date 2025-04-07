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
                    'legend.fontsize': 12})    
    
def set_plot_params_elsevier():
    # Set commonly used plot parameters to desired values
    # Plotting parameters
    rcParams.update({
                    #'figure.figsize'   : (3.54331,2.6575), # 90 mm width in inches, 4:3 aspect ratio
                    'figure.figsize'    : (8,6), # 90 mm width in inches, 4:3 aspect ratio
                    'font.serif'        : 'Times New Roman',
                    'font.family'       : 'serif',
                    'font.size': 10,
                    'figure.titlesize' : 12,
                    'axes.labelsize': 11,
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    'legend.fontsize': 10})    