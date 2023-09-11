"""
(Super)FLEXOP Polars Postprocessing Script

This script is used to postprocess and analyze simulation data from (Super)FLEXOP simulations
to generate aerodynamic polar plots.

Usage:
- Modify script parameters as needed to customize the analysis.
- Run the script to generate polar plots and save results.

"""

# Import necessary modules
import numpy as np
import configobj
import os
import polars_utils   
import matplotlib.pyplot as plt
from aero import area_ref, chord_main_root

# CONSTANTS
S = area_ref  # Reference area
c_ref = chord_main_root  # Reference chord length

def get_pmor_data(path_to_case):
    """
    Extracts simulation data from the .pmor.sharpy file.
    
    Parameters:
    - path_to_case (str): Path to the simulation case folder.

    Returns:
    - case_name (str): Name of the simulation case.
    - path_to_sharpy_pmor (str): Path to the .pmor.sharpy file.
    - pmor (configobj.ConfigObj): Parsed .pmor.sharpy data.

    Credits:
        Thanks goes to @Norberto Goizueta for this function and .pmor handling!
    """
    case_name = path_to_case.split('/')[-1]
    path_to_sharpy_pmor = path_to_case + f'/{case_name}.pmor.sharpy'
    if not os.path.exists(path_to_sharpy_pmor):
        raise FileNotFoundError
    pmor = configobj.ConfigObj(path_to_sharpy_pmor)

    return case_name, path_to_sharpy_pmor, pmor

def process_case(path_to_case):
    """
    Extracts relevant data from a simulation case folder.

    Parameters:
    - path_to_case (str): Path to the simulation case folder.

    Returns:
    - alpha (float): Angle of attack in degrees.
    - Cl (float): Lift coefficient.
    - Cd (float): Drag coefficient.
    - My (float): Pitching moment coefficient.

    Credits:
        Thanks goes to @Norberto Goizueta for this function!
    """
    _, _, pmor = get_pmor_data(path_to_case)
    alpha = pmor['parameters']['alpha']
    inertial_forces = np.loadtxt(f'{path_to_case}/forces/forces_aeroforces.txt',
                                 skiprows=1, delimiter=',', dtype=float)[1:4]
    inertial_moments = np.loadtxt(f'{path_to_case}/forces/moments_aeroforces.txt',
                                  skiprows=1, delimiter=',', dtype=float)[1:4]

    return alpha, inertial_forces[2], inertial_forces[0], inertial_moments[1]

def apply_coefficients(results, q, S, c_ref):
    """
    Applies aerodynamic coefficients to simulation results.

    Parameters:
    - results (numpy.ndarray): Simulation results [alpha, Cl, Cd, My].
    - q (float): Dynamic pressure.
    - S (float): Reference area.
    - c_ref (float): Reference chord length.

    Returns:
    - results (numpy.ndarray): Results with applied coefficients.
    """
    qS = q * S
    results[:, 1:] /= qS  # lift, drag, and moment coefficients
    results[:, 3] /= c_ref  # moment coefficient

    return results

def store_data(alpha, Cl, Cd, My, matrix_data):
    """
    Stores simulation data in a matrix.

    Parameters:
    - alpha (float): Angle of attack in degrees.
    - Cl (float): Lift coefficient.
    - Cd (float): Drag coefficient.
    - My (float): Pitching moment coefficient.
    - matrix_data (numpy.ndarray): Existing data matrix.

    Returns:
    - matrix_data (numpy.ndarray): Updated data matrix.
    """
    data = np.array([alpha, Cl, Cd, My], dtype=float)
    if matrix_data is None:
        return data
    else:
        return np.vstack((matrix_data, data))

def write_results(data, case, result_folder):  
    """
    Writes processed simulation results to a text file.

    Parameters:
    - data (numpy.ndarray): Processed simulation data.
    - case (str): Name of the simulation case.
    - result_folder (str): Path to the folder where results will be saved.
    """
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    np.savetxt(os.path.join(os.path.join(result_folder, case + '.txt')), 
               data,
               fmt='%10e,' * (np.shape(data)[1] - 1) + '%10e', 
               delimiter=", ", 
               header= 'alpha, Cl, Cd, My')

def postprocess_polars(list_alpha_deg):  
    """
    Postprocesses (Super)FLEXOP simulation results to generate aerodynamic polar plots.

    Parameters:
    - list_alpha_deg (list of float): List of angles of attack in degrees.
    """
    u_inf = 45  # Freestream velocity

    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    SHARPY_output_folder  = './output/' + 'superflexop_uinf_{}_polars/'.format(u_inf)
    result_folder = route_dir + '/results_data/'
    
    for lifting_only in [True]:
        for use_polars in [False, True]:
            matrix_data = None
            for alpha_deg in list_alpha_deg:              
                case_name = polars_utils.get_case_name(u_inf, alpha_deg, use_polars, lifting_only)
                alpha, Cl, Cd, My = process_case(SHARPY_output_folder + case_name)
                matrix_data = store_data(alpha, Cl, Cd, My, matrix_data)
            matrix_data = np.sort(matrix_data, 0)
            matrix_data = apply_coefficients(matrix_data, 0.5 * 1.1336 * u_inf ** 2, S, c_ref)
            data_name = "superflexop_uinf{}_p{}_f{}".format(int(u_inf), int(use_polars), int(not lifting_only))
            write_results(matrix_data, data_name, result_folder)
            
def plot_polars():
    """
    Plots aerodynamic polar data.

    This function reads and plots the aerodynamic polar data generated by the script.
    """
    u_inf = 45
    list_results = []
    list_labels = []

    fig, ax = plt.subplots(1, 3, figsize=(16,9))
    for use_polars in [False, True]:
        for lifting_only in [True, False]:
            data_name = "superflexop_uinf{}_p{}_f{}".format(int(u_inf),int(use_polars), int(not lifting_only))
            results = np.loadtxt(os.path.join(result_folder, data_name + '.txt'),
                                           delimiter=',')
            label = "p{} f{}".format(int(use_polars), int(not lifting_only))
    
            ax[0].plot(results[:,0],
                        results[:,1],
                        label=label)
            ax[1].plot(results[:,2],
                        results[:,1],
                        label=label)
            ax[2].plot(results[:,0],
                        results[:,3],
                        label=label)
    for iax in ax:
        iax.grid()
    ax[2].legend()
    ax[0].set_xlabel('alpha, deg')
    ax[0].set_ylabel('$C_L$')
    ax[1].set_xlabel('$C_D$')
    ax[1].set_ylabel('$C_L$')
    ax[2].set_xlabel('alpha, deg')
    ax[2].set_ylabel('$C_M$')
    fig.tight_layout()
    plt.savefig('{}/superflexop_steady_polar.png'.format(result_folder))
    plt.show()

if __name__ == '__main__': 
    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    result_folder = route_dir + '/results_data/'
    plot_polars()
