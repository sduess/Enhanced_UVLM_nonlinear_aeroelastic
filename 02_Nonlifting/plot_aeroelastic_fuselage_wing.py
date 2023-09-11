"""
Wing Deflection of Fuselage-Wing ConfigurationPlotting Script

This script is used to generate wing deflection plots for fuselage-wing configurations simulations, 
including different stiffness scaling factors.

Usage:
- Modify script parameters as needed to customize the analysis.
- Run the script to generate and save wing deflection plots.

"""

import numpy as np
import os
import matplotlib.pyplot as plt

# Define script directory and result folder
script_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
result_folder = script_dir + '/result_data/'

def get_linestyle(lifting_only, model):
    """
    Get the appropriate linestyle for plotting based on simulation settings.

    Parameters:
    - lifting_only (bool): Whether to consider lifting surfaces only.
    - model (str): Aircraft model name.

    Returns:
    - linestyle (str): Linestyle for the plot.
    """
    if bool(lifting_only):
        return '-'
    elif model == 'low_wing':
        return ':'
    else:
        return '--'

def get_legend(stiffness_scaling_factors):
    """
    Generate legend labels for the plots.

    Parameters:
    - stiffness_scaling_factors (list): List of stiffness scaling factors.

    Returns:
    - list_labels (list): List of legend labels.
    """
    from matplotlib.lines import Line2D
    list_labels = []
    for isigma, sigma in enumerate(stiffness_scaling_factors):
        list_labels.append(Line2D([0], [0], color=list_colors[isigma], label=r"$\sigma$ = {}".format(sigma)))

    list_labels.append(Line2D([0], [0], color='k', linestyle='solid', label='wing only')) 
    list_labels.append(Line2D([0], [0], color='k', linestyle='dashed', label='wing+fuselage - $z/R=0.0$')) 
    list_labels.append(Line2D([0], [0], color='k', linestyle='dotted', label='wing+fuselage - $z/R=-0.5$'))
    return list_labels  

def plot_wing_deflections(list_stiffness_factors):
    """
    Plot wing deflections for different stiffness scaling factors.

    Parameters:
    - list_stiffness_factors (list): List of stiffness scaling factors.
    """
    list_models = ['low_wing', 'mid_wing_1']
    flag_wing_only = False

    for lifting_only in [1, 0]:
        for model in list_models:
            for isigma, sigma in enumerate(list_stiffness_factors):
                if flag_wing_only and bool(lifting_only): 
                    continue

                data = np.loadtxt(os.path.join(result_folder,
                            '{}_coupled_lifting_only{}_sigma{}.txt'.format(
                                model,
                                int(lifting_only),
                                int(sigma*100)
                            )),
                            delimiter=',')
                plt.plot(data[:,1],
                         data[:,2],
                         color=list_colors[isigma],
                         linestyle=get_linestyle(lifting_only, model))
            flag_wing_only = True
                
    # Create legends
    handles = get_legend(list_stiffness_factors)
    plt.legend(handles=handles)
    plt.grid()
    plt.xlabel('y/s')
    plt.ylabel('z/s')
    plt.savefig(os.path.join(result_folder,
                             'wing_deformation_including_fuselage_effects.png'))
    plt.show()
    
if __name__ == '__main__': 
    list_stiffness_factors = [0.25, 0.33, 1.0]
    list_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    plot_wing_deflections(list_stiffness_factors)
