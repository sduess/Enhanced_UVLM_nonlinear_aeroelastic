import numpy as np
import sharpy.utils.algebra as algebra
import json
import os

def get_settings(flexop_model, flow, dt, **kwargs):
    """
    Generate settings dictionary for SHARPy simulation.

    Args:
        flexop_model (FlexopModel): An instance of the FlexopModel class containing the aircraft model.
        flow (str): Flow solver to use.
        dt (float): Time step size.
        **kwargs: Additional keyword arguments for simulation parameters.

    Returns:
        dict: A dictionary containing settings for the SHARPy simulation.
    """
    # Simulation Parameters
    alpha = kwargs.get('alpha', 0.)  # Angle of attack in radians
    cs_deflection = kwargs.get('cs_deflection_initial', 0.)  # Control surface deflection in radians
    u_inf = kwargs.get('u_inf', 10.)  # Freestream velocity in m/s
    rho = kwargs.get('rho', 1.225)  # Air density in kg/m^3
    thrust = kwargs.get('thrust', 0.)  # Thrust force
    gust = kwargs.get('gust', False)  # Gust simulation flag
    variable_wake = kwargs.get('variable_wake', False)  # Variable wake configuration flag

    use_polars = kwargs.get('use_polars', False)  # Use polars for corrections

    horseshoe = kwargs.get('horseshoe', False)  # Horseshoe vortex method flag
    gravity = kwargs.get('gravity', True)  # Gravity effect flag
    wake_length = kwargs.get('wake_length', 10)  # Length of wake
    free_flight = kwargs.get('free_flight', False)  # Free-flight simulation flag
    num_modes = kwargs.get('num_modes', 10)  # Number of modes

    num_cores = kwargs.get('num_cores', 2)  # Number of CPU cores
    tolerance = kwargs.get('tolerance', 1e-6)  # Tolerance for convergence
    n_load_steps = kwargs.get('n_load_steps', 5)  # Number of load steps
    fsi_tolerance = kwargs.get('fsi_tolerance', 1e-4)  # FSI tolerance
    structural_relaxation_factor = kwargs.get('structural_relaxation_factor', 0)  # Structural relaxation factor
    relaxation_factor = kwargs.get('relaxation_factor', 0)  # Relaxation factor
    newmark_damp = kwargs.get('newmark_damp', 0.5e-4)  # Newmark damping factor

    n_tstep = kwargs.get('n_tstep', 1)  # Number of time steps

    # Initialize settings dictionary
    settings = {}
    settings['SHARPy'] = {
        'case': flexop_model.case_name,
        'route': flexop_model.case_route,
        'flow': flow,
        'write_screen': 'on',
        'write_log': 'on',
        'log_folder': flexop_model.output_route,
        'log_file': flexop_model.case_name + '.log'
    }

    # BeamLoader Settings
    settings['BeamLoader'] = {
        'unsteady': 'on',
        'orientation': algebra.euler2quat(np.array([0., alpha, 0.]))
    }

    # AeroForcesCalculator Settings
    settings['AeroForcesCalculator'] = {
        'write_text_file': True,
        'coefficients': False
    }

    # NonLinearStatic Settings
    settings['NonLinearStatic'] = {
        'print_info': 'off',
        'max_iterations': 150,
        'num_load_steps': 1,
        'delta_curved': 1e-1,
        'min_delta': tolerance,
        'gravity_on': gravity,
        'gravity': 9.81
    }

    # StaticUvlm Settings
    settings['StaticUvlm'] = {
        'print_info': 'on',
        'horseshoe': horseshoe,
        'num_cores': num_cores,
        'n_rollup': 0,  # int(wake_length * flexop_model.aero.m),
        'velocity_field_generator': 'SteadyVelocityField',
        'velocity_field_input': {
            'u_inf': u_inf,
            'u_inf_direction': [1., 0, 0]
        },
        'rho': rho,
        'cfl1': bool(not variable_wake),
        'nonlifting_body_interactions': kwargs.get("nonlifting_body_interactions", False)
    }

    # StaticCoupled Settings
    settings['StaticCoupled'] = {
        'print_info': 'off',
        'structural_solver': 'NonLinearStatic',
        'structural_solver_settings': settings['NonLinearStatic'],
        'aero_solver': 'StaticUvlm',
        'aero_solver_settings': settings['StaticUvlm'],
        'max_iter': 100,
        'n_load_steps': n_load_steps,
        'tolerance': fsi_tolerance,
        'relaxation_factor': structural_relaxation_factor,
        'nonlifting_body_interactions': kwargs.get("nonlifting_body_interactions", False)
    }

    # StaticTrim Settings
    settings['StaticTrim'] = {
        'solver': 'StaticCoupled',
        'solver_settings': settings['StaticCoupled'],
        'initial_alpha': alpha,
        'initial_deflection': cs_deflection,
        'initial_thrust': thrust,
        'tail_cs_index': [4, 10],
        'thrust_nodes': [0],
        'fz_tolerance': 1e-6,
        'fx_tolerance': 1e-6,
        'm_tolerance': 1e-6,
        'max_iter': 200,
        'save_info': True
    }
    
    # AerogridLoader Settings
    settings['AerogridLoader'] = {
        'unsteady': 'on',
        'aligned_grid': 'on',
        'mstar': kwargs.get('mstar', wake_length * flexop_model.aero.m),
        'wake_shape_generator': 'StraightWake',
        'wake_shape_generator_input': {
            'u_inf': u_inf,
            'u_inf_direction': [1., 0., 0.],
            'dt': dt,
        }
    }
    
    # Handle special cases for horseshoe and variable wake
    if horseshoe:
        settings['AerogridLoader']['mstar'] = 1
    if variable_wake:
        print("mstar = ", settings['AerogridLoader']['mstar'])
        settings['AerogridLoader']['wake_shape_generator_input'] = kwargs.get('dict_wake_shape', {
            'dx1': flexop_model.aero.chord_main_tip / flexop_model.aero.m,
            'ndx1': 23,
            'r': 1.6,
            'dxmax': 5 * flexop_model.aero.chord_main_root
        })

    # NonliftingbodygridLoader Settings
    settings['NonliftingbodygridLoader'] = {}

    # NonLinearDynamicCoupledStep Settings
    settings['NonLinearDynamicCoupledStep'] = {
        'print_info': 'off',
        'max_iterations': 950,
        'delta_curved': 1e-1,
        'min_delta': tolerance,
        'newmark_damp': newmark_damp,
        'gravity_on': gravity,
        'gravity': 9.81,
        'num_steps': n_tstep,
        'dt': dt,
        'initial_velocity': u_inf,
    }

    # NonLinearDynamicPrescribedStep Settings
    settings['NonLinearDynamicPrescribedStep'] = {
        'print_info': 'off',
        'max_iterations': 950,
        'delta_curved': 1e-1,
        'min_delta': tolerance,
        'newmark_damp': newmark_damp,
        'gravity_on': gravity,
        'gravity': 9.81,
        'num_steps': n_tstep,
        'dt': dt,
    }

    # BeamPlot Settings
    settings['BeamPlot'] = {}

    # AerogridPlot Settings
    settings['AerogridPlot'] = {
        'include_rbm': 'off',
        'include_applied_forces': 'on',
        'minus_m_star': 5,
        'u_inf': u_inf,
        'plot_nonlifting_surfaces': kwargs.get("nonlifting_body_interactions", False)
    }

    # BeamLoads Settings
    settings['BeamLoads'] = {'csv_output': True}

    # StepUvlm Settings
    settings['StepUvlm'] = {
        'num_cores': num_cores,
        'convection_scheme': 2,
        'gamma_dot_filtering': 7,
        'cfl1': bool(not variable_wake),
        'velocity_field_generator': 'SteadyVelocityField',
        'velocity_field_input': {
            'u_inf': u_inf * int(not free_flight),
            'u_inf_direction': [1., 0, 0]
        },
        'rho': rho,
        'n_time_steps': n_tstep,
        'dt': dt,
        'nonlifting_body_interactions': kwargs.get("nonlifting_body_interactions", False)
    }

    # Handle gust settings if applicable
    if gust:
        gust_settings = kwargs.get('gust_settings', {
            'gust_shape': '1-cos',
            'gust_length': 10.,
            'gust_intensity': 0.01,
            'gust_offset': 0.
        })
        settings['StepUvlm']['velocity_field_generator'] = 'GustVelocityField'
        settings['StepUvlm']['velocity_field_input'] = {
            'u_inf': u_inf,
            'u_inf_direction': [1., 0, 0],
            'relative_motion': bool(not free_flight),
            'offset': gust_settings['gust_offset'],
            'gust_shape': gust_settings['gust_shape'],
            'gust_parameters': {
                'gust_length': gust_settings['gust_length'],
                'gust_intensity': gust_settings['gust_intensity'] * u_inf,
            }
        }

    # Determine structural solver based on free_flight
    if free_flight:
        structural_solver = 'NonLinearDynamicCoupledStep'
    else:
        structural_solver = 'NonLinearDynamicPrescribedStep'

    # SaveData Settings
    settings['SaveData'] = {
        'save_aero': True,
        'save_struct': True,
    }

    # Include linear settings if applicable
    if 'LinearAssembler' in flow:
        settings['SaveData']['save_linear'] = True
        settings['SaveData']['save_linear_uvlm'] = True
        unsteady_force_distribution = False
    else:
        unsteady_force_distribution = True

    # DynamicCoupled Settings
    settings['DynamicCoupled'] = {
        'structural_solver': structural_solver,
        'structural_solver_settings': settings[structural_solver],
        'aero_solver': 'StepUvlm',
        'aero_solver_settings': settings['StepUvlm'],
        'fsi_substeps': 200,
        'fsi_tolerance': fsi_tolerance,
        'relaxation_factor': relaxation_factor,
        'minimum_steps': 1,
        'relaxation_steps': 150,
        'final_relaxation_factor': 0.05,
        'n_time_steps': n_tstep,
        'dt': dt,
        'include_unsteady_force_contribution': unsteady_force_distribution,
        'postprocessors': kwargs.get('postprocessors_dynamic', ['BeamLoads', 'SaveData']),
        'postprocessors_settings': dict(),
        'nonlifting_body_interactions': kwargs.get("nonlifting_body_interactions", False),
    }

    # WriteVariablesTime Settings
    settings['WriteVariablesTime'] = {
        'structure_variables': ['pos'],
        'structure_nodes': [flexop_model.structure.n_node_main - 1],
        'cleanup_old_solution': 'on',
    }

    # Populate postprocessors settings
    for postprocessor in settings['DynamicCoupled']['postprocessors']:
        if postprocessor in settings.keys():
            settings_postprocessor = settings[postprocessor]
        else:
            settings_postprocessor = {}  # Default settings

        settings['DynamicCoupled']['postprocessors_settings'][postprocessor] = settings_postprocessor

    # Handle closed-loop network settings
    if kwargs.get('closed-loop', False):
        settings['DynamicCoupled']['network_settings'] = kwargs.get('netowrk_settings', {})

    # Modal Settings
    settings['Modal'] = {
        'print_info': True,
        'use_undamped_modes': True,
        'NumLambda': num_modes,
        'rigid_body_modes': free_flight,
        'write_modes_vtk': 'on',
        'print_matrices': 'on',
        'continuous_eigenvalues': 'off',
        'dt': dt,
        'plot_eigenvalues': False,
    }

    # Include LinearAssembler settings if applicable
    if 'LinearAssembler' in flow:
        settings['LinearAssembler'] = {
            'linear_system': 'LinearAeroelastic',
            'inout_coordinates': 'nodes',
            'linear_system_settings': {
                'beam_settings': {
                    'modal_projection': True,
                    'inout_coords': 'modes',
                    'discrete_time': True,
                    'newmark_damp': newmark_damp,
                    'discr_method': 'newmark',
                    'dt': dt,
                    'proj_modes': 'undamped',
                    'num_modes': num_modes,
                    'print_info': 'on',
                    'gravity': gravity,
                },
                'aero_settings': {
                    'dt': dt,
                    'integr_order': 2,
                    'density': rho,
                    'remove_predictor': True,
                    'use_sparse': 'off',
                    'remove_inputs': ['u_gust'],
                    'gust_assembler': 'LeadingEdge',  # 'leading_edge',
                },
                'track_body': free_flight,
                'use_euler': free_flight,
            }
        }
        
        # Handle ROM settings if applicable
        rom_settings = kwargs.get('rom_settings', {'use': False})
        if rom_settings['use']:
            settings['SaveData']['save_rom'] = True
            settings['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method'] = rom_settings['rom_method']
            settings['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method_settings'] = rom_settings['rom_method_settings']

    # Adjust rigid body modes for free flight
    if not free_flight:
        settings['Modal']['rigid_body_modes'] = False

    # Update settings for polar corrections if required
    if use_polars:
        settings = update_settings_for_polar_corrections(settings)

    # AsymptoticStability Settings
    settings['AsymptoticStability'] = {
        'print_info': 'on',
        'frequency_cutoff': 0,
        'export_eigenvalues': 'on',
        'modes_to_plot': num_modes,
        'velocity_analysis': [20, 80, 13]
    }

    # LiftDistribution Settings
    settings['LiftDistribution'] = {'rho': rho}

    # AeroForcesCalculator Settings
    settings['AeroForcesCalculator'] = {
        'write_text_file': 'on',
        'nonlifting_body': kwargs.get("nonlifting_body_interactions", False),
        'coefficients': 'off',
        'S_ref': flexop_model.reference_area,
        'q_ref': 0.5 * rho * u_inf ** 2
    }

    # WriteVariablesTime Settings (structure_variables: ['pos', 'psi'])
    settings['WriteVariablesTime'] = {
        'structure_variables': ['pos', 'psi'],
        'structure_nodes': list(range(flexop_model.structure.n_node_main + 1)),
        'cleanup_old_solution': 'on',
        'delimiter': ','
    }

    # SaveParametricCase Settings
    settings['SaveParametricCase'] = {
        'parameters': {'alpha': np.rad2deg(alpha), 'u_inf': u_inf},
        'save_case': 'off'
    }

    return settings


def update_settings_for_polar_corrections(settings):
    """
    Update settings for polar corrections.

    Args:
        settings (dict): Existing settings dictionary.

    Returns:
        dict: Updated settings with polar correction settings.
    """
    print("update polar settings!")
    aoa_cl_deg = [-3.28415340783741, 0]
    for solver in ['StaticCoupled', 'DynamicCoupled']:
        settings[solver]['correct_forces_method'] = 'PolarCorrection'
        settings[solver]['correct_forces_settings'] = {
            'cd_from_cl': 'off',
            'correct_lift': 'on',
            'moment_from_polar': 'on',
            'skip_surfaces': [],
            'aoa_cl0': np.deg2rad(aoa_cl_deg),
            'write_induced_aoa': False,
        }
    return settings

def get_skipped_attributes(list_to_be_saved_attr):
    """
    Get a list of skipped attributes from a predefined list.

    Args:
        list_to_be_saved_attr (list): List of attributes to be saved.

    Returns:
        list: List of skipped attributes.
    """
    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    with open(route_dir + '/list_aero_and_structural_ts_attributes.json', 'r') as f:
        list_of_all_attributes = json.load(f)
    
    for attribute in list_to_be_saved_attr:
        list_of_all_attributes.remove(attribute)

    return list_of_all_attributes
