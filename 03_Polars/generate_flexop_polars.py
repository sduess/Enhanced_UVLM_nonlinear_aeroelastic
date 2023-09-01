import sys
import numpy as np
import polars_utils
sys.path.insert(1,'../../01_Case_Generator')
from generate_flexop_case import generate_flexop_case 


def run_polars(list_alpha_deg):
    u_inf = 45 # cruise flight speed
    rho = 1.1336 # corresponds to an altitude of 800  m
    alpha_rad = 6.406771329255241468e-03 #0.389 deg

    simulation_settings = {
        'lifting_only': True, # ignore nonlifting bodies
        'wing_only': False, # Wing only or full configuration (wing+tail)
        'dynamic': False, # unsteady simulation
        'wake_discretisation': False, # cfl not zero and use variable wake discretisation scheme
        'gravity': True,
        'horseshoe':  False, # Check why not working
        'use_polars': False, # Apply polar corrections
        'free_flight': False, # False: clamped
        'use_trim': False, # Trim aircraft
        'mstar': 80, # number of streamwise wake panels
        'num_chord_panels': 8, # Chordwise lattice discretisation
        'n_elem_multiplier': 2, # multiplier for spanwise node discretisation
        'num_radial_panels': 36, # Radial Fuselage Discretisation
        'n_elem_multiplier_fuselage': 2, # multiplier for spanwise node discretisation
        'num_cores': 8, # number of cores used for parallelization
        'sigma': 0.3, # stiffness scaling factor, sigma = 1: FLEXOP, sigma = 0.3 SuperFLEXOP
    }
    
    # Set Flow
    flow = ['BeamLoader', 
            'AerogridLoader',
            'NonliftingbodygridLoader',
            'AerogridPlot',
            'StaticCoupled',
            'AeroForcesCalculator',
            'AerogridPlot',
            'WriteVariablesTime',
            'SaveParametricCase'
            # 'SaveData'
            ] 
    
    if simulation_settings["lifting_only"]:
        flow.remove('NonliftingbodygridLoader')
    for use_polar in [False, True]:
        simulation_settings['use_polars'] = use_polar
        for alpha_deg in list_alpha_deg:
            if alpha_deg >= 20:
                n_load_steps = 10
            else:
                n_load_steps = 5
            # Set initial cruise parameter
            initial_trim_values = {'alpha': np.deg2rad(alpha_deg),
                                   'delta': 0,
                                   'thrust': 0,}
        
            case_name = polars_utils.get_case_name(u_inf,
                                                   alpha_deg, 
                                                   simulation_settings['use_polars'],
                                                   simulation_settings['lifting_only'])
            output_folder = './output/' + 'superflexop_uinf_{}_polars_liftingonly{}'.format(u_inf, int(simulation_settings["lifting_only"])) + '/'
            # Generate model and start simulation
            flexop_model = generate_flexop_case(u_inf,
                                                rho,
                                                flow,
                                                initial_trim_values,
                                                case_name,
                                                **simulation_settings,
                                                output_folder=output_folder,
                                                n_load_steps=n_load_steps,
                                                nonlifting_interactions=bool(not simulation_settings["lifting_only"]))
            
            flexop_model.run()

if __name__ == '__main__':
    list_alpha_deg = np.arange(-5,31, 1)
    run_polars(list_alpha_deg)