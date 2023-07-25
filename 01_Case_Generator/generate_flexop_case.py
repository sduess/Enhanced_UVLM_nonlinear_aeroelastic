import os
import flexop as aircraft
import numpy as np
import sharpy.utils.algebra as algebra
from helper_functions.get_settings import get_settings

cases_route = '../../cases/'
output_route = './output/'



def generate_flexop_case(u_inf,
                        rho,
                        flow,
                        initial_trim_values,
                        case_name,
                        **kwargs):
    


    # Set Aircraft trim
    alpha =  initial_trim_values['alpha'] 
    cs_deflection = initial_trim_values['delta']
    thrust = initial_trim_values['thrust']   

    # Set Inputs for Polar
    data_polars = None
    use_polars = kwargs.get('use_polars', False)
    if use_polars:
        flexop_directory = os.path.abspath(aircraft.FLEXOP_DIRECTORY)
        airfoil_polars = {
                            0: flexop_directory + '/src/airfoil_polars_alpha_30_/xfoil_seq_re1300000_root.txt',
                            1: flexop_directory + '/src/airfoil_polars_alpha_30_/xfoil_seq_re1300000_naca0012.txt',
                        }
        data_polars = generate_polar_arrays(airfoil_polars)

    # setup FLEXOP model
    flexop_model = aircraft.FLEXOP(case_name, cases_route, output_route)
    flexop_model.clean()
    flexop_model.init_structure(sigma=kwargs.get('sigma', 0.3), 
                                n_elem_multiplier=kwargs.get('n_elem_multiplier', 2),
                                n_elem_multiplier_fuselage = 1, 
                                lifting_only=kwargs.get('lifting_only', True),
                                wing_only = kwargs.get('wing_only', False))
    flexop_model.init_aero(m=kwargs.get('num_chord_panels', 8),
                           cs_deflection = cs_deflection,
                           polars = data_polars) 
    flexop_model.structure.set_thrust(thrust)

    flexop_model.generate()
    flexop_model.structure.calculate_aircraft_mass()

    # Other parameters
    CFL = 1
    dt = CFL * flexop_model.aero.chord_main_root / flexop_model.aero.m / u_inf

    # Setup gust
    gust_settings = kwargs.get('gust_settings', {'use_gust': False})
    if gust_settings['use_gust']:
        gust_settings['gust_offset'] *= dt * u_inf # TODO: define offset independed of dt and velocity!

    # numerics
    structural_relaxation_factor = 0.6
    tolerance = 1e-6 
    fsi_tolerance = 1e-4 
    newmark_damp = 0.5e-4
                    
    # Get settings dict
    settings = get_settings(flexop_model,
                            flow,
                            dt,
                            gust =  gust_settings['use_gust'],
                            gust_settings = gust_settings,
                            alpha = alpha,
                            cs_deflection = cs_deflection,
                            u_inf = u_inf,
                            rho = rho,
                            thrust = thrust,
                            wake_length = kwargs.get('wake_length', 10),
                            free_flight = kwargs.get('free_flight',True),
                            num_cores = kwargs.get('num_cores', 2),
                            tolerance = tolerance,
                            fsi_tolerance = fsi_tolerance,
                            structural_relaxation_factor = structural_relaxation_factor,
                            newmark_damp = newmark_damp,
                            n_tstep = kwargs.get('n_tstep', 1),
                            variable_wake = kwargs.get('wake_discretisation', False),
                            dict_wake_shape = kwargs.get('dict_wake_shape', None),
                            use_polars=use_polars,
                            cs_deflection_initial=cs_deflection,
                            mstar=kwargs.get('mstar', 80)
                            )

    flexop_model.create_settings(settings)
    return flexop_model

def generate_polar_arrays(airfoils):
    # airfoils = {name: filename}
    # Return a aoa (rad), cl, cd, cm for each airfoil
    out_data = [None] * len(airfoils)
    for airfoil_index, airfoil_filename in airfoils.items():
        out_data[airfoil_index] = np.loadtxt(airfoil_filename, skiprows=12)[:, :4]
        if any(out_data[airfoil_index][:, 0] > 1):
            # provided polar is in degrees so changing it to radians
            out_data[airfoil_index][:, 0] *= np.pi / 180
    return out_data
