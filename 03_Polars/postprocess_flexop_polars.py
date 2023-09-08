"""
(Super)FLEXOP Polars Postprocessing Script

This script is used to postprocess the results of (Super)FLEXOP simulations and calculate aerodynamic coefficients (Cl, Cd, My).
It reads the PMOR files generated during simulations and processes the aerodynamic forces and moments.


Usage:
- Modify the parameters in the `main()` function to match your specific simulation results and desired analysis.
- Run the script to process the data, calculate coefficients, and save the results.

"""

# Import necessary modules
import numpy as np
import configobj
from aero import area_ref, chord_main_root
import os
import polars_utils

# CONSTANTS
S = area_ref
c_ref = chord_main_root

def get_pmor_data(path_to_case):
    """
    Load and parse PMOR data from a (Super)FLEXOP simulation case.

    Parameters:
    - path_to_case (str): Path to the simulation case folder.

    Returns:
    - case_name (str): Name of the simulation case.
    - path_to_sharpy_pmor (str): Path to the SHARPy PMOR file.
    - pmor (configobj.ConfigObj): Parsed PMOR data.
    """
    case_name = path_to_case.split('/')[-1]
    path_to_sharpy_pmor = path_to_case + f'/{case_name}.pmor.sharpy'
    if not os.path.exists(path_to_sharpy_pmor):
        raise FileNotFoundError
    pmor = configobj.ConfigObj(path_to_sharpy_pmor)

    return case_name, path_to_sharpy_pmor, pmor

def process_case(path_to_case):
    """
    Process aerodynamic forces and moments data from a (Super)FLEXOP simulation case.

    Parameters:
    - path_to_case (str): Path to the simulation case folder.

    Returns:
    - alpha (float): Angle of attack (in radians).
    - Cl (float): Lift coefficient.
    - Cd (float): Drag coefficient.
    - My (float): Pitching moment coefficient.
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
    Calculate aerodynamic coefficients (Cl, Cd, My) and apply scaling.

    Parameters:
    - results (numpy.ndarray): Raw aerodynamic force and moment data.
    - q (float): Dynamic pressure (0.5 * rho * u_inf^2).
    - S (float): Reference area.
    - c_ref (float): Reference chord length.

    Returns:
    - results_scaled (numpy.ndarray): Scaled aerodynamic coefficients.
    """
    qS = q * S
    results[:, 1:] /= qS  # lift, drag, and moment
    results[:, 3] /= c_ref  # moment only

    return results

def store_data(alpha, Cl, Cd, My, matrix_data):
    """
    Store aerodynamic coefficient data in a matrix.

    Parameters:
    - alpha (float): Angle of attack (in radians).
    - Cl (float): Lift coefficient.
    - Cd (float): Drag coefficient.
    - My (float): Pitching moment coefficient.
    - matrix_data (numpy.ndarray): Existing data matrix (or None).

    Returns:
    - data (numpy.ndarray): Updated data matrix.
    """
    data = np.array([alpha, Cl, Cd, My], dtype=float)
    if matrix_data is None:
        return data
    else:
        return np.vstack((matrix_data, data))
        
     
def write_results(data, case, result_folder):  
    """
    Write processed aerodynamic coefficient data to a text file.

    Parameters:
    - data (numpy.ndarray): Processed aerodynamic coefficient data.
    - case (str): Name of the simulation case.
    - result_folder (str): Path to the folder where processed results will be saved.
    """
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    np.savetxt(os.path.join(os.path.join(result_folder, case + '.txt')), 
               data,
               fmt='%10e,' * (np.shape(data)[1] - 1) + '%10e', 
               delimiter=", ", 
               header='alpha, Cl, Cd, My')
    
def postprocess_polars(angle_of_attacks_in_deg):  
    """
    Postprocess (Super)FLEXOP simulation results for different angles of attack.

    Parameters:
    - angle_of_attacks_in_deg (numpy.ndarray): Array of angles of attack (in degrees) to postprocess.
    """
    u_inf = 45 
    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    result_folder = route_dir + '/results_data/'
    
    for lifting_only in [False, True]:
        SHARPY_output_folder  = './output/' + 'superflexop_uinf_{}_polars_liftingonly{}'.format(u_inf, int(lifting_only)) + '/'
        for use_polars in [False, True]:
            matrix_data = None
            for alpha_deg in angle_of_attacks_in_deg:              
                case_name = polars_utils.get_case_name(u_inf, alpha_deg, use_polars, lifting_only)
                alpha, Cl, Cd, My = process_case(SHARPY_output_folder + case_name)
                matrix_data = store_data(alpha, Cl, Cd, My, matrix_data)
            matrix_data = apply_coefficients(matrix_data, 0.5 * 1.1336 * u_inf ** 2, S, c_ref)
            data_name = "superflexop_uinf{}_p{}_f{}".format(int(u_inf), int(use_polars), int(not lifting_only))
            write_results(matrix_data, data_name, result_folder)

if __name__ == '__main__': 
    import polars_utils   
    angle_of_attacks_in_deg = np.arange(-5, 31, 1)
    postprocess_polars(angle_of_attacks_in_deg)
