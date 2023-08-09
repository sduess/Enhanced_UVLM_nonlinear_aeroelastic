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
    
def run_chordwise_discretisation_convergence_study(plot_only=False):
    flow = [
        'BeamLoader', 
        'AerogridLoader',
        'StaticCoupled',
        'AerogridPlot',
        'WriteVariablesTime',
        'AeroForcesCalculator'
        ]
    

    u_inf = 45 # cruise flight speed
    rho = 1.1336 # corresponds to an altitude of 800  m
    default_simulation_settings['use_trim'] = 'StaticTrim' in flow
    if default_simulation_settings['use_trim']:
        alpha_rad =  6.796482976011756182e-03 
        elevator_deflection=-3.325087601649625961e-03
        thrust=2.052055145318664842e+00
    else:
        alpha_rad = np.deg2rad(1)
        elevator_deflection = 0.
        thrust = 0.
    default_simulation_settings['horseshoe'] = True
    default_simulation_settings['wing_only'] = True
    default_simulation_settings['gravity'] = True
    default_simulation_settings['n_elem_multiplier'] = 2
    convergence_type = 'chordwise_discretisation'
    flag_first_run = True
    # horseshoe = True
    list_num_chord_panels = np.array([4, 8, 16, 24],dtype=int)
    results_deformation = None
    results_aeroforces = None
    results_trim = None
    for num_chord_panels in list_num_chord_panels:
        case_name = '{}_convergence_{}_alpha{}_uinf{}_m{}_nelem{}_w{}_nmodes{}'.format(get_aircraft_name(default_simulation_settings["sigma"]),
                                                                                       convergence_type,
                                                                                    int(np.rad2deg(alpha_rad*100)),
                                                                                    int(u_inf),
                                                                                    num_chord_panels,
                                                                                    default_simulation_settings['n_elem_multiplier'],
                                                                                    1,
                                                                                    0)
        
        default_simulation_settings['num_chord_panels'] = num_chord_panels
        flexop_model = generate_flexop_case(u_inf,
                                            rho,
                                            flow,
                                            set_initial_trim_values(alpha_rad=alpha_rad,
                                                                    elevator_deflection=elevator_deflection, 
                                                                    thrust=thrust),
                                            case_name,
                                            **default_simulation_settings)
        if not plot_only:
            flexop_model.run()

        # get deformation
        tip_deformation = np.loadtxt(os.path.join(output_route, 
                                                               case_name,
                                                               'WriteVariablesTime',
                                                               'struct_pos_node{}.dat'.format(flexop_model.structure.n_node_main-1)
                                                               ))[3]
        aeroforces = np.transpose(np.loadtxt(os.path.join(output_route, 
                                                               case_name,
                                                               'forces',
                                                               'forces_aeroforces.txt'                                                               
                                                               ),
                                                               delimiter=','))[3]
        aeroforces /= 0.5 * rho  *u_inf**2 * 2.54
        if default_simulation_settings['use_trim']:
            trim_conditions = np.loadtxt(os.path.join(output_route, 
                                                    case_name,
                                                    'statictrim',
                                                    'trim_values.txt'
                                                    ))
            results_trim = store_data(flag_first_run, trim_conditions, results_trim)

        results_deformation = store_data(flag_first_run, tip_deformation, results_deformation)
        results_aeroforces = store_data(flag_first_run, aeroforces, results_aeroforces)
        flag_first_run = False

    error_aeroforces, relative_error_aeroforces = compute_error(results_aeroforces, index_reference_column=-1)

    plot_data(results_aeroforces, list_num_chord_panels, ylabel="$F, N", xlabel='number of chordwise panels')
    plot_data(error_aeroforces, list_num_chord_panels, ylabel="$\Delta$ F, N", xlabel='number of chordwise panels')
    plot_data(relative_error_aeroforces, list_num_chord_panels, ylabel="$\Delta$ F, %", xlabel='number of chordwise panels', filename='rel_error_Fz_chordwise_convergence')
    if default_simulation_settings['use_trim']:
        for i in range(3):
            plt.plot(list_num_chord_panels, results_trim[i, :])
            plt.show()

def run_wake_length_convergence_study(plot_only=False):
    flow = [
        'BeamLoader', 
        'AerogridLoader',
        'StaticCoupled',
        'WriteVariablesTime',
        'AeroForcesCalculator'
        ]
    u_inf = 45 # cruise flight speed
    rho = 1.1336 # corresponds to an altitude of 800  m
    alpha_rad = 6.796482976011756182e-03

    num_chord_panels = 16
    default_simulation_settings['num_chord_panels'] = num_chord_panels
    convergence_type = 'wake_discretisation'
    flag_first_run = True
    results_deformation = None
    results_aeroforces = None
    list_wake_length_factor = [1, 5, 10, 15, 20]
    list_wake_panel = np.array(list_wake_length_factor) 
    list_wake_panel[1:] *= num_chord_panels
    
    for wake_length_factor in list_wake_length_factor:
        if wake_length_factor == 1:            
            default_simulation_settings['mstar'] = 1
            default_simulation_settings['horseshoe'] = True
        else:
            default_simulation_settings['mstar'] = wake_length_factor * num_chord_panels
        default_simulation_settings['horseshoe'] = False
        case_name = '{}_convergence_{}_alpha{}_uinf{}_m{}_nelem{}_w{}_nmodes{}'.format(get_aircraft_name(default_simulation_settings["sigma"]),
                                                                                    convergence_type,
                                                                                    int(np.rad2deg(alpha_rad*100)),
                                                                                    int(u_inf),
                                                                                    num_chord_panels,
                                                                                    default_simulation_settings['n_elem_multiplier'],
                                                                                    wake_length_factor,
                                                                                    0)
       
        'mstar'
        flexop_model = generate_flexop_case(u_inf,
                                            rho,
                                            flow,
                                            set_initial_trim_values(alpha_rad=alpha_rad),
                                            case_name,
                                            **default_simulation_settings)
        if not plot_only:
            flexop_model.run()
        
        # get deformation
        tip_deformation = np.loadtxt(os.path.join(output_route, 
                                                               case_name,
                                                               'WriteVariablesTime',
                                                               'struct_pos_node{}.dat'.format(flexop_model.structure.n_node_main-1)
                                                               ))[3]
        aeroforces = np.transpose(np.loadtxt(os.path.join(output_route, 
                                                               case_name,
                                                               'forces',
                                                               'forces_aeroforces.txt'                                                               
                                                               ),
                                                               delimiter=','))[3]

        aeroforces /= 0.5 * rho  *u_inf**2 * 2.54
        results_deformation = store_data(flag_first_run, tip_deformation, results_deformation)
        results_aeroforces = store_data(flag_first_run, aeroforces, results_aeroforces)
        flag_first_run = False

    error_aeroforces, relative_error_aeroforces = compute_error(results_aeroforces) #, index_reference_column=0)

    plot_data(relative_error_aeroforces, list_wake_panel, ylabel="$\Delta$ F, %", xlabel='number of wake panels', filename='wake_length_convergence_rel_force')

       
def get_time_history(output_folder, case):
    import h5py as h5
    file = os.path.join(output_folder,
                         case, 
                        'savedata', 
                        case + '.data.h5')   
    with h5.File(file, "r") as f:
            ts_max = len(f['data']['structure']['timestep_info'].keys())-2
            dt = float(str(np.array(f['data']['settings']['DynamicCoupled']['dt'])))
            matrix_data = np.zeros((ts_max, 2)) # parameters: time, tip displacement
            matrix_data[:,0] = np.array(list(range(ts_max))) * dt
            node_tip = np.argmax(np.array(f['data']['structure']['timestep_info']['00000']['pos'])[:,1])
            half_wingspan = 7.07/2
            for its in range(1,ts_max):
                ts_str = f'{its:05d}'
                matrix_data[its, 1] = np.array(f['data']['structure']['timestep_info'][ts_str]['pos'])[node_tip, 2]/half_wingspan*100 
    return matrix_data

def run_convergence_dynamic_gust_response_study(plot_only=False):
    # TODO: Save Parameters with WriteVariablesTime- insteady of SaveData
    flow = [
        'BeamLoader', 
        'AerogridLoader',
        'StaticTrim',
        'DynamicCoupled',
        ]
    u_inf = 45 # cruise flight speed
    rho = 1.1336 # corresponds to an altitude of 800  m
    alpha_rad = 6.796482976011756182e-03
    default_simulation_settings['n_elem_multiplier'] = 2

    convergence_type = 'dynamic_chordwise_discretisation'
    flag_first_run = True
    results_trim= None
    list_chordwise_panels = [4, 8, 16] #, 24]    
    default_simulation_settings["postprocessors_dynamic"] = ['WriteVariablesTime']
    default_simulation_settings['use_trim'] = 'StaticTrim' in flow
    if default_simulation_settings['use_trim']:
        alpha_rad =  6.796482976011756182e-03 
    else:
        alpha_rad = np.deg2rad(1)
    default_simulation_settings['horseshoe'] = False
    default_simulation_settings['wing_only'] = False
    dict_results_deformation = dict()
    # Gust velocity field
    gust_settings ={'use_gust': True,
                    'gust_shape': '1-cos',
                    'gust_length': 10.,
                    'gust_intensity': 0.1,
                    'gust_offset': 10}  
    wake_length_factor = 10
    simulation_time = 0.25
    list_result_max_peak = []
    list_result_time_peak = []
    for num_chord_panels in list_chordwise_panels:
        if default_simulation_settings["sigma"] == 0.3:
            alpha_rad = np.deg2rad(0.38)
        else:
            alpha_rad = np.deg2rad(1.)
        dt = 0.471 / num_chord_panels / u_inf
        default_simulation_settings['n_tstep'] = int(simulation_time / dt)
        default_simulation_settings['mstar'] = wake_length_factor * num_chord_panels
        default_simulation_settings['num_chord_panels'] = num_chord_panels
        case_name = '{}_convergence_{}_alpha{}_uinf{}_m{}_nelem{}_w{}_nmodes{}_d1'.format(get_aircraft_name(default_simulation_settings["sigma"]),
                                                                                    convergence_type,
                                                                                    int(np.rad2deg(alpha_rad*100)),
                                                                                    int(u_inf),
                                                                                    num_chord_panels,
                                                                                    default_simulation_settings['n_elem_multiplier'],
                                                                                    wake_length_factor,
                                                                                    0)
       
        'mstar'
        flexop_model = generate_flexop_case(u_inf,
                                            rho,
                                            flow,
                                            set_initial_trim_values(alpha_rad=alpha_rad,
                                                                    elevator_deflection=-3.325087601649625961e-03, 
                                                                    thrust=2.052055145318664842e+00),
                                            case_name,
                                            gust_settings=gust_settings,
                                            **default_simulation_settings)
        if not plot_only:
            flexop_model.run()
        
        if default_simulation_settings['use_trim']:
            trim_conditions = np.loadtxt(os.path.join(output_route, 
                                                    case_name,
                                                    'statictrim',
                                                    'trim_values.txt'
                                                    ))
            results_trim = store_data(flag_first_run, trim_conditions, results_trim)
        # get deformation
        try:  
            tip_deformation = np.loadtxt(os.path.join(output_route, 
                                                                case_name,
                                                                'WriteVariablesTime',
                                                                'struct_pos_node{}.dat'.format(flexop_model.structure.n_node_main-1)
                                                                ))[:,3]
                                                                
        except:
            tip_deformation = get_time_history(output_route, case_name)
            tip_deformation = tip_deformation[:,1]
        time_signal = np.array(list(range((np.shape(tip_deformation)[0])))) * dt

        plt.plot(time_signal, tip_deformation)
        dict_results_deformation[case_name] = dict()
        dict_results_deformation[case_name]['time_history'] = np.column_stack((time_signal,tip_deformation))
        dict_results_deformation[case_name]['peak_deflection'] = np.max(tip_deformation)
        dict_results_deformation[case_name]['timestep_peak'] = time_signal[np.argmax(tip_deformation)]
        list_result_max_peak.append(dict_results_deformation[case_name]['peak_deflection'])
        list_result_time_peak.append(dict_results_deformation[case_name]['timestep_peak'])
        flag_first_run = False
    
    plt.xlabel('time, s')
    plt.ylabel('tip deflection, %')
    plt.show()

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
    run_chordwise_discretisation_convergence_study(plot_only=False)
    run_wake_length_convergence_study(plot_only=False)
