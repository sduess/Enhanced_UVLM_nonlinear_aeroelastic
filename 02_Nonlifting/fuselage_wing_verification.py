import os
import numpy as np
from sharpy.cases.templates.fuselage_wing_configuration.fuselage_wing_configuration import Fuselage_Wing_Configuration
from sharpy.cases.templates.fuselage_wing_configuration.fwc_get_settings import define_simulation_settings
import json
import matplotlib.pyplot as plt
import nonlifting_utils

def verifiy_fuselage_wing_configuration():
    """
        Test spanwise lift distribution on a fuselage-wing configuration.
        
        The lift distribution on low wing configuration is computed considering
        fuselage effects. The final results are compared to a previous solution 
        (backward compatibility) that matches the experimental lift distribution for this case. 
    """
    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    case_route, output_route, results_folder = nonlifting_utils.define_folder(route_dir)
    model = 'mid_wing' #'low_wing'
    fuselage_length = 10
    dict_geometry_parameters = get_geometry_parameters(model,
                                                       route_dir,
                                                       fuselage_length=fuselage_length)

    # Freestream Conditions
    if model == 'low_wing':
        alpha_zero_deg = 2.9
    else:
        alpha_zero_deg = 5.8

    alpha_deg = 0.
    u_inf = 10

    # Discretization
    dict_discretization = {
        'n_elem_per_wing': 40,
        'n_elem_fuselage': 30,
        'num_chordwise_panels': 8,
        'num_radial_panels': 36
    }

    # Simulation settings
    horseshoe = False
    phantom_test = False
    lifting_only = False
    # Simlation Solver Flow
    flow = ['BeamLoader',
            'AerogridLoader',
            'NonliftingbodygridLoader',
            'StaticUvlm',
            'BeamLoads',
            'LiftDistribution',
            'AerogridPlot',
                ]
    for lifting_only in [True, False]:
        case_name = '{}_lifting{}'.format(model, int(lifting_only))
        flow_case = flow.copy()
        if lifting_only:
            flow_case.remove('NonliftingbodygridLoader')
        wing_fuselage_model = nonlifting_utils.generate_model(case_name, 
                                                            case_route,
                                                            output_route,
                                                            dict_geometry_parameters,
                                                            dict_discretization, 
                                                            lifting_only)

        nonlifting_utils.generate_simulation_settings(flow_case, 
                                                    wing_fuselage_model, 
                                                    alpha_deg, 
                                                    u_inf, 
                                                    lifting_only,
                                                    horseshoe=horseshoe,
                                                    phantom_test=phantom_test)
        # run simulation
        if not lifting_only:
            wing_fuselage_model.run()
        # get results
        lift_distribution = nonlifting_utils.load_lift_distribution(
            output_route + case_name,
            wing_fuselage_model.structure.n_node_right_wing,
            dimensionalize=True
            )
        
        label = nonlifting_utils.get_label(lifting_only)
        plt.scatter(lift_distribution[:,0], lift_distribution[:,1], label = label)
    # check results
    lift_distribution_test = np.loadtxt(route_dir + "/test_data/experimental_lift_distribution_{}.csv".format(model), skiprows=1, delimiter=',')  # TODO: Exchange with experimental results

    plt.scatter(lift_distribution_test[:,0], lift_distribution_test[:,1], label = 'Experiments')
    plt.legend()
    plt.xlabel('y/s')
    plt.ylabel('$c_l$')
    plt.savefig('{}/result_data/spanwise_lift_{}.png'.format(route_dir, model))
    plt.show()

if __name__ == '__main__':
    verifiy_fuselage_wing_configuration()
