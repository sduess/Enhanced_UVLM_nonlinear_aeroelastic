"""
(Super)FLEXOP Steady Simulation Script with applied Polar Corrections

This script is used to perform a series of (Super)FLEXOPsimulations with varying angles of attack and sectional polar corrections.
It generates cases for different alpha values, runs simulations, and saves the results.

Usage:
- Modify the parameters in the `main()` function to match your specific simulation requirements.
- Run the script to generate and run (Super)FLEXOP simulations for the specified alpha values and polar correction settings.

"""

# Import necessary modules
import sys
import numpy as np
import polars_utils
sys.path.insert(1, '../../01_Case_Generator')
from generate_flexop_case import generate_flexop_case 

def run_polars(angle_of_attacks_in_deg):
    """
    Perform a series of (Super)FLEXOP simulations for varying angles of attack and polar correction settings.

    Parameters:
    - angle_of_attacks_in_deg (numpy.ndarray): Array of angles of attack (in degrees) to simulate.
    """
    u_inf = 45  # Cruise flight speed
    rho = 1.1336  # Corresponds to an altitude of 800 m

    simulation_settings = {
        'lifting_only': True,  # Ignore nonlifting bodies
        'wing_only': False,  # Wing only or full configuration (wing+tail)
        'dynamic': False,  # Unsteady simulation
        'wake_discretisation': False,  # CFL not zero and use variable wake discretization scheme
        'gravity': True,
        'horseshoe':  False,  # Check why not working
        'use_polars': False,  # Apply polar corrections
        'free_flight': False,  # False: clamped
        'use_trim': False,  # Trim aircraft
        'mstar': 80,  # Number of streamwise wake panels
        'num_chord_panels': 8,  # Chordwise lattice discretization
        'n_elem_multiplier': 2,  # Multiplier for spanwise node discretization
        'num_radial_panels': 36,  # Radial Fuselage Discretization
        'n_elem_multiplier_fuselage': 2,  # Multiplier for spanwise node discretization
        'num_cores': 8,  # Number of cores used for parallelization
        'sigma': 0.3,  # Stiffness scaling factor, sigma = 1: FLEXOP, sigma = 0.3 SuperFLEXOP
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
            ] 
    
    if simulation_settings["lifting_only"]:
        flow.remove('NonliftingbodygridLoader')
    
    # Loop over angle of attacks with and without applied polar corrections
    for use_polar in [False, True]:
        simulation_settings['use_polars'] = use_polar
        for alpha_deg in angle_of_attacks_in_deg:
            # Define sufficient number of load steps for convergence
            if alpha_deg >= 17:
                n_load_steps = 10
            else:
                n_load_steps = 5
            # Set initial cruise parameter
            initial_trim_values = {'alpha': np.deg2rad(alpha_deg),
                                   'delta': 0,
                                   'thrust': 0}
        
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
    # Define a range of angle of attack values to simulate
    angle_of_attacks_in_deg = np.arange(-5, 31, 1)
    
    # Run (Super)FLEXOP simulations for these specified values
    run_polars(angle_of_attacks_in_deg)
