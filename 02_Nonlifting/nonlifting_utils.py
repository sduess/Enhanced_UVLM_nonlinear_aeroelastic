import numpy as np
import os

def tearDown(route_dir):
    """
        Removes all created files within this test.
    """
    import shutil
    folders = ['cases', 'output']
    for folder in folders:
        shutil.rmtree(route_dir + '/' + folder)


def define_folder(route_dir):
    """
        Initializes all folder path needed and creates case folder.
    """
    case_route = route_dir + '/cases/'
    output_route = route_dir + '/output/'

    results_folder = route_dir + '/result_data/'

    if not os.path.exists(case_route):
        os.makedirs(case_route)
    return case_route, output_route, results_folder, results_folder


def get_timestep(model, u_inf):
    return model.aero.chord_wing / model.aero.num_chordwise_panels / u_inf

def load_deformation_distribution(output_folder, n_node_wing, n_tsteps=0):
    """
        Loads the resulting beam deformation saved in dat-files.
    """
    result_deformation = np.zeros((n_node_wing, 3))
    for inode in range(n_node_wing):
        result_deformation[inode, :] = np.loadtxt(os.path.join(output_folder,
                                            'WriteVariablesTime',
                                            'struct_pos_node{}.dat'.format(int(inode)))
                                            )[1:]
    return result_deformation

def load_lift_distribution(output_folder, n_node_wing, n_tsteps=0):
    """
        Loads the resulting pressure coefficients saved in txt-files.
    """
    lift_distribution = np.loadtxt(output_folder + '/liftdistribution/liftdistribution_ts{}.txt'.format(str(n_tsteps)), delimiter=',')
    return lift_distribution[:n_node_wing,[1,-1]]
    
def get_geometry_parameters(model_name,route_dir, fuselage_length=10):
    """
        Geometry parameters are loaded from json init file for the specified model. 
        Next, final geoemtry parameteres, depending on the fuselage length are 
        calculated and return within a dict.
    """
    import json
    with open(route_dir + '/geometry_parameter_models.json', 'r') as fp:
        parameter_models = json.load(fp)[model_name]

    geometry_parameters = {
        'fuselage_length': fuselage_length,         
        'max_radius': fuselage_length/parameter_models['length_radius_ratio'],
        'fuselage_shape': parameter_models['fuselage_shape'],
        'alpha_zero_deg': parameter_models['alpha_zero_deg']
    }
    geometry_parameters['chord']=geometry_parameters['max_radius']/parameter_models['radius_chord_ratio']
    geometry_parameters['half_wingspan'] = geometry_parameters['max_radius']/parameter_models['radius_half_wingspan_ratio']
    geometry_parameters['offset_nose_wing'] = parameter_models['length_offset_nose_to_wing_ratio'] * fuselage_length
    geometry_parameters['vertical_wing_position'] = parameter_models['vertical_wing_position'] * geometry_parameters['max_radius']
    return geometry_parameters

def generate_model(case_name, 
                    dict_geometry_parameters,
                    dict_discretisation, 
                    lifting_only,
                    case_route,
                    output_route,
                    sigma=1.):
    """
        Aircraft model object is generated and structural and aerodynamic (lifting and nonlifting)
        input files are generated.
    """
    from sharpy.cases.templates.fuselage_wing_configuration.fuselage_wing_configuration import Fuselage_Wing_Configuration
    aircraft_model = Fuselage_Wing_Configuration(case_name, case_route, output_route)
    aircraft_model.init_aeroelastic(lifting_only=lifting_only,
                                **dict_discretisation,
                                **dict_geometry_parameters,
                                sigma=sigma)
    aircraft_model.generate()
    return aircraft_model


def generate_simulation_settings(flow, 
                                    aircraft_model, 
                                    alpha_deg, 
                                    u_inf, 
                                    lifting_only,
                                    n_tsteps=0,
                                    horseshoe=True,
                                    nonlifting_only=False,
                                    phantom_test=False,
                                    dynamic_structural_solver='NonLinearDynamicPrescribedStep'
                                    ):
    """
        Simulation settings are defined and written to the ".sharpy" input file.
    """
    from sharpy.cases.templates.fuselage_wing_configuration.fwc_get_settings import define_simulation_settings
    settings = define_simulation_settings(flow, 
                                            aircraft_model, 
                                            alpha_deg, 
                                            u_inf, 
                                            dt=get_timestep(aircraft_model, u_inf),
                                            n_tsteps=n_tsteps,
                                            lifting_only=lifting_only, 
                                            phantom_test=phantom_test, 
                                            nonlifting_only=nonlifting_only, 
                                            horseshoe=horseshoe,
                                            dynamic_structural_solver=dynamic_structural_solver)
    aircraft_model.create_settings(settings)


def get_label(lifting_only):
    label = 'SHARPy'
    if lifting_only:
        label += ' - wing only'
    else:
        label += ' - wing and fuselage'
    return label

