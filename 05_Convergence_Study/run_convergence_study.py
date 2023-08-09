import numpy as np
import os
import matplotlib.pyplot as plt
import sys

sys.path.insert(1,'../../01_Case_Generator')
from generate_flexop_case import generate_flexop_case 

def run_modal_convergence_study(plot_only=False):
    # Convergence of modes due to spanwise number of nodes
    flow = [
        'BeamLoader', 
        'AerogridLoader',
        'BeamPlot',
        'Modal',
        ]
    
    # Set Flight Conditions
    u_inf = 45 # cruise flight speed
    rho = 1.1336 # corresponds to an altitude of 800  m
    alpha_rad = 6.796482976011756182e-03

    # Set initial cruise parameter
    num_chord_panels = 4
    wake_length = 1
    horseshoe = True
    default_simulation_settings['horseshoe'] = horseshoe
    default_simulation_settings['mstar'] = 1
    default_simulation_settings['wing_only'] = True
    default_simulation_settings['free_flight'] = False
    num_modes= 20 + 10 * int(default_simulation_settings['free_flight'])
    num_modes_selected = 6 
    convergence_type = 'beam_discretisation'
    flag_first_run = True
    results_frequencies = None
    list_n_elem_multiplier = [1, 2.3, 4, 6, 8, 10] 
    list_n_spanwise_nodes = [] 
 
    for n_elem_multiplier in list_n_elem_multiplier:

        case_name = '{}_convergence_{}_alpha{}_uinf{}_m{}_nelem{}_w{}_nmodes{}'.format(get_aircraft_name(default_simulation_settings["sigma"]),
                                                                                       convergence_type,
                                                                                    int(np.rad2deg(alpha_rad*100)),
                                                                                    int(u_inf),
                                                                                    num_chord_panels,
                                                                                    n_elem_multiplier,
                                                                                    wake_length,
                                                                                    num_modes)

        default_simulation_settings['n_elem_multiplier'] = n_elem_multiplier
        flexop_model = generate_flexop_case(u_inf,
                                            rho,
                                            flow,
                                            set_initial_trim_values(alpha_rad=alpha_rad),
                                            case_name,
                                            num_modes=num_modes,
                                            **default_simulation_settings)
        
        list_n_spanwise_nodes.append(flexop_model.structure.n_node_main-1)

        if not plot_only: 
            flexop_model.run()
        

        # get frequencies
        frequencies = np.unique(np.loadtxt(os.path.join(output_route, case_name, 'beam_modal_analysis/frequencies.dat'),
                                           delimiter='|',
                                           skiprows=10*int(default_simulation_settings["free_flight"])))
        frequencies /= 2 * np.pi
        if plot_only:
            frequencies = frequencies[:num_modes_selected]

        results_frequencies = store_data(flag_first_run, frequencies, results_frequencies)
        flag_first_run = False

    error_frequencies, relative_error_frequencies= compute_error(results_frequencies)
    plot_data(error_frequencies, list_n_spanwise_nodes, 
              ylabel="$\Delta$ f, Hz", 
              xlabel='number of spanwise nodes', 
              legend_label_prefix = "mode ",
              filename = '{}_convergence_spanwise_nodes_abs_error'.format(get_aircraft_name(default_simulation_settings["sigma"])))
    plot_data(relative_error_frequencies, list_n_spanwise_nodes, ylabel="$\Delta$ f, %", xlabel='number of spanwise nodes',
              filename = '{}convergence_spanwise_nodes_rel_error'.format(get_aircraft_name(default_simulation_settings["sigma"])))

def get_results(output_route, list_case_names, frequencies=False):
    num_cases = len(list_case_names)
    list_return_results = []
    for i_case, case_name in enumerate(list_case_names):
        if frequencies:
            frequencies = np.unique(np.loadtxt(os.path.join(output_route, case_name, 'beam_modal_analysis/frequencies.dat')))
            results_frequencies = store_data(i_case==0, frequencies, results_frequencies)
            if i_case == num_cases - 1:
                list_return_results.append(results_frequencies)

    return list_return_results # use * to unpack list?
                
def compute_error(results_frequencies, index_reference_column = -1):
    # assume finest grid is at the last position
    finest_result = results_frequencies[:,index_reference_column]
    error_frequencies = results_frequencies.copy()
    relative_error_frequencies = results_frequencies.copy()
    for icase in range(np.shape(results_frequencies)[1]):
        error_frequencies[:,icase] -= finest_result
        for irow in range(np.shape(results_frequencies)[0]):
            relative_error_frequencies[irow,icase] =  error_frequencies[irow,icase] /finest_result[irow]
    relative_error_frequencies *= 100
    return error_frequencies, relative_error_frequencies

def plot_data(error_data, xaxis, ylabel="", xlabel="",legend_label_prefix = "", filename=None):
    for imode in range(np.shape(error_data)[0]):
        plt.plot(xaxis, error_data[imode,:], 'o-', label = legend_label_prefix+str(imode))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # plt.legend()
    plt.grid()
    if filename is not None:
        import tikzplotlib
        tikzplotlib.save(os.path.join(result_folder,"{}.tex".format(filename)))
        plt.savefig(os.path.join(result_folder,"{}.png".format(filename)))
    plt.show()

def get_aircraft_name(sigma):
    """
        Returns the aircraft name depending on sigma.

        Args:
            sigma (float): Stiffness scaling factor.

        Returns:
            Name of aircraft (str)
    """
    if sigma == 1:
        return "flexop"
    elif sigma == 0.3:
        return "superflexop"
    else:
        return "flexop_sigma{}".format(int(sigma*10))
    
def store_data(flag_first_run, data, data_storage):
    if flag_first_run:
        return data
    else:
        return np.column_stack((data_storage, data))
    

def set_initial_trim_values(alpha_rad=0,elevator_deflection=0, thrust=0):
    """
        Creates and return a dict containing the initial trim values.

        Args:
            alpha_rad: Aircraft angle of attack in radians.
            elevator_deflection: Elevator deflection in radians.
            thrust: Aircraft thrust in Newton.

        Returns:
            initial_trim_values: Dict storing all necessary aircraft trim values.

    """
    initial_trim_values = {'alpha': alpha_rad, 
                        'delta':elevator_deflection,
                        'thrust': thrust}
    return initial_trim_values

if __name__ == "__main__":
    # Define Folders
    cases_route = '../01_case_files/'
    output_route = './output/'

    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    result_folder = route_dir + '/results_data/'

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    # Define Simulation settings
    default_simulation_settings = {
    'lifting_only': True, # ignore nonlifting bodies
    'wing_only': False, # Wing only or full configuration (wing+tail)
    'dynamic': False, # unsteady simulation
    'wake_discretisation': False, # cfl not zero and use variable wake discretisation scheme
    'gravity': True,
    'horseshoe':  False, 
    'use_polars': False, # Apply polar corrections
    'free_flight': False, # False: clamped
    'use_trim': False, # Trim aircraft
    'mstar': 80, # number of streamwise wake panels
    'num_chord_panels': 8, # Chordwise lattice discretisation
    'n_elem_multiplier': 2, # multiplier for spanwise node discretisation
    'n_tstep': 1000,    # number of simulation timesteps
    'num_cores': 8, # number of cores used for parallelization
    'sigma': 0.3, # stiffness scaling factor, sigma = 1: FLEXOP, sigma = 0.3 SuperFLEXOP
    }
    run_modal_convergence_study(plot_only=False)
            
