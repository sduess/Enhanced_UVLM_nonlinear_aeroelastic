import os
import numpy as np
import matplotlib.pyplot as plt
import nonlifting_utils

def verifiy_fuselage_wing_configuration(model):
    """
    Test spanwise lift distribution on a fuselage-wing configuration.

    This function computes the spanwise lift distribution for a fuselage-wing configuration, 
    accounting for fuselage effects, and compares the results to experimental measurements.

    Parameters:
        model (str): The name of the aircraft model.

    """
    # Get the directory paths
    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    case_route, output_route, results_folder = nonlifting_utils.define_folder(route_dir)

    # Define fuselage length
    fuselage_length = 10
    dict_geometry_parameters = nonlifting_utils.get_geometry_parameters(model,
                                                       route_dir,
                                                       fuselage_length=fuselage_length)

    # Freestream Conditions
    if model == 'mid_wing':
        alpha_deg = 5.8
    else:
        alpha_deg = 2.9
    u_inf = 10
    
    # Discretization
    dict_discretization = {
        'n_elem_per_wing': 20,
        'n_elem_fuselage': 30,
        'num_chordwise_panels': 16,
        'num_radial_panels': 36
    }

    # Simulation settings
    horseshoe = False
    phantom_test = False

    # Simulation Solver Flow
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
                                                            dict_geometry_parameters,
                                                            dict_discretization, 
                                                            lifting_only,
                                                            case_route,
                                                            output_route)

        nonlifting_utils.generate_simulation_settings(flow_case, 
                                                    wing_fuselage_model, 
                                                    alpha_deg, 
                                                    u_inf, 
                                                    lifting_only,
                                                    horseshoe=horseshoe,
                                                    phantom_test=phantom_test)
        # Run simulation
        wing_fuselage_model.run()
        # Get results
        lift_distribution = nonlifting_utils.load_lift_distribution(
            output_route + case_name,
            wing_fuselage_model.aero.aero_node,
            wing_fuselage_model.structure.n_node_right_wing,
            dimensionalize=True
            )
        
        label = nonlifting_utils.get_label(lifting_only)
        plt.scatter(lift_distribution[:,0], lift_distribution[:,1], label=label)

    # Check results against experimental data
    lift_distribution_test = np.loadtxt(route_dir + "/test_data/experimental_lift_distribution_{}.csv".format(model), skiprows=1, delimiter=',')  # TODO: Exchange with experimental results

    plt.scatter(lift_distribution_test[:,0], lift_distribution_test[:,1], label='Experiments')
    plt.legend()
    plt.xlabel('y/s')
    plt.ylabel('$c_l$')
    plt.savefig('{}/spanwise_lift_{}.png'.format(results_folder, model))
    plt.show()

    nonlifting_utils.tearDown(route_dir)

if __name__ == '__main__':
    model = 'low_wing' 
    verifiy_fuselage_wing_configuration(model)
