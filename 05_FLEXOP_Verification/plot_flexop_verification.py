"""
FLEXOP Plot Verification Study

This script loads and plots simulation data for the FLEXOP verification cases.
It compares the results from the simulation with reference data for lift distribution and deformations.

Usage:
- Specify your 1g and 5g load case in list_case_names and run the script
- The results are saved in a png-file
"""
# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Get the directory of the current script
script_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# Define half-wing span
flexop_halfwing_span = 7.07/2

def get_reference_data(material='reference'):
    """
    Load reference data from a CSV file.

    Args:
        material (str): Material identifier ('reference' or 'tailored').

    Returns:
        pd.DataFrame: Reference data as a Pandas DataFrame.
    """
    file_reference = os.path.join(script_dir,
        'reference_data',
        'tudelft_results_{}_one_and_five_g.csv'.format(material))
    
    df_reference = pd.read_csv(file_reference, skiprows=2, header=None, sep=";")
    return df_reference

def load_data(file_name, results_folder):
    """
    Load data from a text file.

    Args:
        file_name (str): Name of the file to load.
        results_folder (str): Path to the folder containing the file.

    Returns:
        np.ndarray: Loaded data as a NumPy array.
    """
    # Generate the file path for the data
    file_dir = os.path.join(results_folder, file_name)
    # Load data from the text file
    data = np.loadtxt(file_dir, delimiter=",")
    return data

def plot_data(list_case_names, results_folder):
    """
    Generate and display plots for lift distribution and deformations.

    Args:
        list_case_names (list): List of case names to plot.
        results_folder (str): Path to the folder containing the results.

    """
    # Create subplots for different parameters
    fig, axs = plt.subplots(3, figsize=(10, 6))
    
    df_reference_data = get_reference_data()
    list_load_cases = ['1g', '5g']
    for icase, case_name in enumerate(list_case_names):
        # 1) Lift Distribution
        data_lift_distribution = load_data('lift_distribution_{}.txt'.format(case_name), 
                                           results_folder)
    
        axs[0].plot(data_lift_distribution[1:,0]/flexop_halfwing_span, 
                    data_lift_distribution[1:,1]/np.diff(data_lift_distribution[:,0]),
                    color=list_colors[icase],
                    linestyle='-',
                    label=list_load_cases[icase] + ' - SHARPy')
        idx_reference = 0 + 2* icase
        axs[0].plot(df_reference_data.iloc[:,idx_reference]/flexop_halfwing_span, 
                    df_reference_data.iloc[:,idx_reference + 1], 
                    '--x', 
                    label = list_load_cases[icase] + "- reference")
        
        # 2) OOP Deformation
        data_displacements = load_data('displacements_{}.txt'.format(case_name), 
                                           results_folder)
        
        axs[1].plot(data_displacements[:,0]/flexop_halfwing_span, 
                data_displacements[:,1]/flexop_halfwing_span,
                color=list_colors[icase],
                linestyle='-')      
        idx_reference += 4
        axs[1].plot(df_reference_data.iloc[:,idx_reference]/flexop_halfwing_span, 
                    df_reference_data.iloc[:,idx_reference + 1]/flexop_halfwing_span,
                    '--x') 

        # 3) Torsional Deformation 
        axs[2].plot(data_displacements[:,0]/flexop_halfwing_span,
                    data_displacements[:,2],
                color=list_colors[icase],
                linestyle='-')
        idx_reference += 4
        axs[2].plot(df_reference_data.iloc[:,idx_reference]/flexop_halfwing_span, 
                    df_reference_data.iloc[:,idx_reference + 1], 
                    '--x')  
        
    # Customize plot settings and labels
    for iax in range(len(axs)):
        plt.setp(axs[iax].get_xticklabels(), visible=iax == len(axs) - 1)
        axs[iax].set_ylabel(list_ylabels[iax])
        axs[iax].set_xlim([0., 1])
        axs[iax].grid()
    lgd = axs[0].legend()
    plt.xlabel("y/s")
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_folder, 'flexop_verification.png'), 
                bbox_extra_artists=(lgd,), 
                bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # Define labels, colors, and linestyles
    list_ylabels =['L/ds, N/m', '$z_{tip}$/s', '$\Theta$, deg']
    list_colors = ['tab:blue', 'tab:orange']
    
    # Define the results folder path
    results_folder = os.path.join(script_dir, 'results_data/')

    list_case_names = ['flexop_uinf45_alpha-0400',  # 1g
                       'flexop_uinf45_alpha10400']  # 5g
    # Generate and display the plots
    plot_data(list_case_names, results_folder)
