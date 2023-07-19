import sys

sys.path.insert(1,'../../01_Case_Generator')
from generate_flexop_case import generate_flexop_case 

u_inf = 45 # cruise flight speed
rho = 1.1336 # corresponds to an altitude of 800  m
alpha_rad = 6.406771329255241468e-03 #0.389 deg

simulation_settings = {
    'lifting_only': True, # ignore nonlifting bodies
    'wing_only': False, # Wing only or full configuration (wing+tail)
    'dynamic': True, # unsteady simulation
    'wake_discretisation': True, # cfl not zero and use variable wake discretisation scheme
    'gravity': True,
    'horseshoe':  False, 
    'use_polars': False, # Apply polar corrections
    'free_flight': True, # False: clamped
    'use_trim': True, # Trim aircraft
    'mstar': 80, # number of streamwise wake panels
    'num_chord_panels': 8, # Chordwise lattice discretisation
    'n_elem_multiplier': 2, # multiplier for spanwise node discretisation
    'n_tstep': 2000,    # number of simulation timesteps
    'num_cores': 8, # number of cores used for parallelization
    'sigma': 0.3, # stiffness scaling factor, sigma = 1: FLEXOP, sigma = 0.3 SuperFLEXOP
}
 

# Set initial cruise parameter
initial_trim_values = {'alpha': alpha_rad, 
                    'delta':-3.325087601649625961e-03,
                    'thrust': 2.052055145318664842e+00}

# Gust velocity field
gust_settings ={'use_gust': True,
                'gust_shape': '1-cos',
                'gust_length': 10.,
                'gust_intensity': 0.1,
                'gust_offset': 10}  

# set wake shape inputs if needed for variable wake discretisation
if simulation_settings['wake_discretisation']:
    dict_wake_shape =  {'dx1': 0.471/simulation_settings['num_chord_panels'],
                        'ndx1': 23,
                        'r':1.6,
                        'dxmax':5*0.471}
    simulation_settings['mstar'] = 35
    print(simulation_settings)
else:
    dict_wake_shape = None
# Set Flow
flow = ['BeamLoader', 
        'AerogridLoader',
        'AerogridPlot',
        'BeamPlot',
        'StaticCoupled',
        'StaticTrim',
        'BeamPlot',
        'AerogridPlot',
        'AeroForcesCalculator',
        'DynamicCoupled',
        ]      

if not simulation_settings['dynamic']:
    flow.remove('DynaimcCoupled')

if not simulation_settings['use_trim']:
    flow.remove('StaticTrim')
else:
    flow.remove('StaticCoupled')

# Loop over various gust lengths    
list_gust_lengths = [10] #[5, 10, 20, 40, 80, 100]

for gust_length in list_gust_lengths:
    gust_settings['gust_length'] = gust_length
  
    case_name = 'superflexop_free_gust_L_{}_I_{}_p_{}_cfl_{}'.format(gust_settings['gust_length'],
                                                                    int(gust_settings['gust_intensity']*100),
                                                                    int(simulation_settings['use_polars']),
                                                                    int(not simulation_settings['wake_discretisation']))

    # Generate model and start simulation
    flexop_model = generate_flexop_case(u_inf,
                                        rho,
                                        flow,
                                        initial_trim_values,
                                        case_name,
                                        gust_settings=gust_settings,
                                        dict_wake_shape=dict_wake_shape,
                                        **simulation_settings)
    flexop_model.run()