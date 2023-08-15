import numpy as np
import configobj
from aero import area_ref, chord_main_root
import os
import polars_utils


# CONSTANTS
S = area_ref
c_ref = chord_main_root


def get_pmor_data(path_to_case):
    case_name = path_to_case.split('/')[-1]
    path_to_sharpy_pmor = path_to_case + f'/{case_name}.pmor.sharpy'
    if not os.path.exists(path_to_sharpy_pmor):
        raise FileNotFoundError
    pmor = configobj.ConfigObj(path_to_sharpy_pmor)

    return case_name, path_to_sharpy_pmor, pmor


def process_case(path_to_case):
    _, _, pmor = get_pmor_data(path_to_case)
    alpha = pmor['parameters']['alpha']
    inertial_forces = np.loadtxt(f'{path_to_case}/forces/forces_aeroforces.txt',
                                 skiprows=1, delimiter=',', dtype=float)[1:4]
    inertial_moments = np.loadtxt(f'{path_to_case}/forces/moments_aeroforces.txt',
                                  skiprows=1, delimiter=',', dtype=float)[1:4]

    return alpha, inertial_forces[2], inertial_forces[0], inertial_moments[1]



def apply_coefficients(results, q, S, c_ref):
    qS = q * S
    results[:, 1:] /= qS  # lift drag and moment
    results[:, 3] /= c_ref  # moment only

    return results

def store_data(alpha, Cl, Cd, My, matrix_data):
    data = np.array([alpha, Cl, Cd, My], dtype=float)
    print("data: ", data)
    if matrix_data is None:
        return data
    else:
        # np.vstack(())
        print("matrix data = ", np.vstack((matrix_data, data)))
        return np.vstack((matrix_data, data))


     
def write_results(data, case, result_folder):  
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    np.savetxt(os.path.join(os.path.join(result_folder, case + '.txt')), 
               data,
               fmt='%10e,' * (np.shape(data)[1] - 1) + '%10e', 
               delimiter=", ", 
               header= 'alpha, Cl, Cd, My')

    
def postprocess_polars(list_alpha_deg):  
    u_inf = 45 

    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    SHARPY_output_folder  = './output/' + 'superflexop_uinf_{}_polars/'.format(u_inf)
    result_folder = route_dir + '/results_data/'
    
    for lifting_only in [True]:
        for use_polars in [False, True]:
            matrix_data =None
            for alpha_deg in list_alpha_deg:              
                case_name = polars_utils.get_case_name(u_inf,alpha_deg, use_polars, lifting_only)
                alpha, Cl, Cd, My = process_case(SHARPY_output_folder + case_name)
                matrix_data = store_data(alpha, Cl, Cd, My, matrix_data)
            matrix_data = np.sort(matrix_data, 0)
            matrix_data = apply_coefficients(matrix_data, 0.5 * 1.1336 * u_inf ** 2, S, c_ref)
            data_name = "superflexop_uinf{}_p{}_f{}".format(int(u_inf),int(use_polars), int(not lifting_only))
            write_results(matrix_data, data_name, result_folder)
            
def plot_polars():
    u_inf = 45
    list_results = []
    list_labels = []

    fig, ax = plt.subplots(1, 3)
    for use_polars in [False, True]:
        for lifting_only in [True]:
            data_name = "superflexop_uinf{}_p{}_f{}".format(int(u_inf),int(use_polars), int(not lifting_only))
            results = np.loadtxt(os.path.join(result_folder, data_name + '.txt'),
                                           delimiter=',')
            label = "p{} f{}".format(int(use_polars), int(not lifting_only))
    
    
            ax[0].plot(results[:,0],
                        results[:,1],
                        label = label)
            ax[1].plot(results[:,2],
                        results[:,1],
                        label = label)
            ax[2].plot(results[:,0],
                        results[:,3],
                        label = label)
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
    import polars_utils   
    import matplotlib.pyplot as plt
    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    result_folder = route_dir + '/results_data/'
    plot_polars()
