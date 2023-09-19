"""
FLEXOP Steady Simulation Script

This script is used for setting up and running steady-state simulations with FLEXOP. 
It defines simulation parameters, settings, and the SHARPy flow sequence to simulate aircraft behavior 
at different angles of attack. The results are later used to verifiy the implemented model with
reference results.

Usage:
- Modify the simulation parameters and settings as needed.
- Run the script to perform steady-state simulations for specified angles of attack.

"""

# Import necessary modules
import sys
import numpy as np
sys.path.insert(1,'../../01_Case_Generator')
from generate_flexop_case import generate_flexop_case
from helper_functions.flexop_utils import get_flexop_aircraft_version

# Define simulation parameters

u_inf = 45  # Cruise flight speed in m/s
rho = 1.1336  # Air density in kg/m^3 (corresponding to an altitude of 800m)
alpha_rad = 6.406771329255241468e-03  # Angle of attack in radians (approximately 0.389 degrees)

# Simulation settings
simulation_settings = {
    'lifting_only': True,  # Ignore nonlifting bodies
    'wing_only': True,  # Simulate the full configuration (wing+tail)
    'dynamic': False,  # Perform unsteady simulation
    'gravity': True,  # Include gravitational effects
    'horseshoe': False,  # Disable horseshoe wake modeling
    'use_polars': False,  # Apply polar corrections
    'use_trim': False,  # Enable aircraft trim
    'mstar': 80,  # Number of streamwise wake panels
    'num_chord_panels': 16,  # Chordwise lattice discretization
    'n_elem_multiplier': 2,  # Multiplier for spanwise node discretization
    'num_cores': 4,  # Number of CPU cores used for parallelization
    'sigma': 1,  # Stiffness scaling factor (1 for FLEXOP, 0.3 for SuperFLEXOP)
}

# Initial trim values
initial_trim_values = {
    'alpha': 0.,
    'delta': 0,
    'thrust': 0
}

# Define the flow sequence
flow = [
    'BeamLoader',
    'AerogridLoader',
    'NonliftingbodygridLoader',
    'AerogridPlot',
    'BeamPlot',
    'StaticCoupled',
    'LiftDistribution',
    'SaveData',
    'AeroForcesCalculator'
]

# Remove certain steps based on simulation settings
if simulation_settings['lifting_only']:
    flow.remove('NonliftingbodygridLoader')

# List of angles of attack to simulate 
list_angles_of_attack_in_deg = [-0.4, 10.4] 

# Loop over specified angles of attack 
for alpha_in_deg in list_angles_of_attack_in_deg:
    initial_trim_values['alpha'] = np.deg2rad(alpha_in_deg)

    # Generate a case name based on simulation settings
    case_name = '{}_uinf{}_alpha{:05d}'.format(
        get_flexop_aircraft_version(simulation_settings["sigma"]),
        int(u_inf),
        int(alpha_in_deg*1000)
    )
    
    # Include 'nonlifting' in the case name if nonlifting bodies are considered
    if not simulation_settings["lifting_only"]:
        case_name += '_nonlifting'
    
    # Generate the FLEXOP model and start the simulation
    flexop_model = generate_flexop_case(
        u_inf,
        rho,
        flow,
        initial_trim_values,
        case_name,
        **simulation_settings,
        nonlifting_interactions=bool(not simulation_settings["lifting_only"])
    )
    flexop_model.run()
