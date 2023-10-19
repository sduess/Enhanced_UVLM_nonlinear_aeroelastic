import os
import flexop as aircraft
import numpy as np
from helper_functions.get_settings import get_settings

cases_route = '../../cases/'

def generate_flexop_case(u_inf,
                        rho,
                        flow,
                        initial_trim_values,
                        case_name,
                        **kwargs):
    """
    Generate and configure a FLEXOP aircraft model for a simulation case.

    Parameters:
    - u_inf (float): Cruise flight speed in m/s.
    - rho (float): Air density in kg/m^3.
    - flow (list): List of flow modules to be used in the simulation.
    - initial_trim_values (dict): Initial trim values including 'alpha', 'delta', and 'thrust'.
    - case_name (str): Unique name for the simulation case.
    - **kwargs (optional): Additional simulation settings.

    Returns:
    - flexop_model (FLEXOP): Configured FLEXOP aircraft model.

    Additional Simulation Settings (in **kwargs):
    - use_polars (bool): Whether to apply polar corrections (default: False).
    - output_folder (str): Output folder for simulation results (default: './output/').
    - sigma (float): Stiffness scaling factor (default: 0.3).
    - n_elem_multiplier (int): Multiplier for spanwise node discretization (default: 2).
    - n_elem_multiplier_fuselage (int): Multiplier for fuselage spanwise node discretization (default: 3).
    - lifting_only (bool): Consider only lifting bodies (default: True).
    - wing_only (bool): Consider the wing only (default: False).
    - num_chord_panels (int): Chordwise lattice discretization (default: 8).
    - num_radial_panels (int): Radial fuselage discretization (default: 36).
    - num_cores (int): Number of cores used for parallelization (default: 8).
    - wake_length (float): Length of the wake (default: 10.0).
    - free_flight (bool): Perform free flight simulation (default: True).
    - n_tstep (int): Number of simulation time steps (default: 1).
    - variable_wake (bool): Use variable wake discretization (default: False).
    - dict_wake_shape (dict): Dictionary specifying wake shape parameters (default: None).
    - gust_settings (dict): Dictionary specifying gust settings (default: {'use_gust': False}).
    - mstar (int): Number of streamwise wake panels (default: 80).
    - num_modes (int): Number of modes (default: 20).
    - postprocessors_dynamic (list): List of postprocessors for dynamic analysis (default: ['BeamLoads', 'SaveData']).
    - n_load_steps (int): Number of load steps (default: 5).
    - nonlifting_body_interactions (bool): Include interactions with nonlifting bodies (default: False).

    Note: Detailed descriptions of these parameters can be found in the function body.
    """

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
    output_route = kwargs.get('output_folder', './output/')
    # setup FLEXOP model
    flexop_model = aircraft.FLEXOP(case_name, cases_route, output_route)
    flexop_model.clean()
    flexop_model.init_structure(sigma=kwargs.get('sigma', 0.3), 
                                n_elem_multiplier=kwargs.get('n_elem_multiplier', 2),
                                n_elem_multiplier_fuselage = kwargs.get('n_elem_multiplier_fuselage',3), 
                                lifting_only=kwargs.get('lifting_only', True),
                                wing_only = kwargs.get('wing_only', False))
    flexop_model.init_aero(m=kwargs.get('num_chord_panels', 8),
                           cs_deflection = cs_deflection,
                           polars = data_polars) 
    nonlifting_body_interactions = kwargs.get("nonlifting_interactions", False)
    if nonlifting_body_interactions:
        flexop_model.init_fuselage(m=kwargs.get('num_radial_panels', 24))
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

    # Numerics
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
                            horseshoe = kwargs.get('horseshoe', False),
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
                            mstar=kwargs.get('mstar', 80),
                            num_modes=kwargs.get('num_modes',20),
                            postprocessors_dynamic=kwargs.get('postprocessors_dynamic', ['BeamLoads', 'SaveData']),
                            n_load_steps=kwargs.get('n_load_steps', 5),
                            nonlifting_body_interactions=nonlifting_body_interactions,
                            use_dynamic_thrust_input=kwargs.get('use_dynamic_thrust_input', False)
                            )
    if kwargs.get('thrust_input_settings', None):
        thrust_input_settings = kwargs.get('thrust_input_settings', {'thrust_input_file':None})

        if thrust_input_settings['thrust_input_file'] is None:
            raise
        thrust_input = np.loadtxt(thrust_input_settings['thrust_input_file'])
        print("thrust input = ", thrust_input)
        generate_thrust_input(case_name, cases_route,thrust_input)

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

def generate_thrust_input(case_name, route, thrust_timeseries):
    import h5py as h5
    dynamic_forces_time = np.zeros((np.shape(thrust_timeseries)[0], 1, 6))
    dynamic_forces_time[:, 0, 0] = thrust_timeseries
    with h5.File(route + '/' + case_name + '.dyn.h5', 'a') as h5file:
        h5file.create_dataset('dynamic_forces', data=dynamic_forces_time)