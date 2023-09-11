import os
import numpy as np
import nonlifting_utils
import matplotlib.pyplot as plt

def run_fuselage_wing_configuration_coupled(model, plot_only=False):
    """
    Computes the wing deformation for the steady fuselage-wing configuration.
    
    The aeroelastic deformation for the steady fuselage-wing configurations is 
    computed and plotted.

    Parameters:
        model (str): The name of the aircraft model.
        plot_only (bool, optional): If True, only plot the results, otherwise, run the simulation and plot the results.

    """
    # Get the directory paths
    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    case_route, output_route, results_folder = nonlifting_utils.define_folder(route_dir)

    # Define fuselage geometry
    fuselage_length = 10
    
    dict_geometry_parameters = nonlifting_utils.get_geometry_parameters(model,
                                                       route_dir,
                                                       fuselage_length=fuselage_length)

    # Freestream Conditions
    alpha_deg = 2.9
    u_inf = 10

    # Discretization
    dict_discretization = {
        'n_elem_per_wing': 10, 
        'n_elem_fuselage': 20, 
        'num_chordwise_panels': 4,
        'num_radial_panels': 24,
    }

    # Simulation settings
    horseshoe = False
    phantom_test = False
    lifting_only = False

    # Simulation Solver Flow
    flow = ['BeamLoader',
            'AerogridLoader',
            'NonliftingbodygridLoader',
            'AerogridPlot',
            'StaticCoupled',
            'BeamLoads',
            'LiftDistribution',
            'AerogridPlot',
            'WriteVariablesTime'
                ]
    
    # Set list of different stiffness scaling factors sigma
    list_sigmas =  np.array([0.25, 0.33, 1.])/10

    for lifting_only in [True, False]:
        # TODO: Choose number depending on model discretisation
        ignore_first_x_nodes_in_force_calculation = 5 * int(lifting_only)
        for isigma, sigma in enumerate(list_sigmas):
            case_name = '{}_coupled_lifting_only{}_sigma{}'.format(model, int(lifting_only), sigma*1000)
            # Generate SHARPy model input files
            wing_fuselage_model = nonlifting_utils.generate_model(case_name, 
                                                dict_geometry_parameters,
                                                dict_discretization, 
                                                lifting_only,
                                                case_route,
                                                output_route,
                                                sigma=sigma)
            
            # Generate SHARPy setting input
            flow_case = flow.copy()
            if lifting_only:
                flow_case.remove('NonliftingbodygridLoader')
            nonlifting_utils.generate_simulation_settings(flow_case, 
                                                wing_fuselage_model, 
                                                alpha_deg, 
                                                u_inf, 
                                                lifting_only,
                                                horseshoe=horseshoe,
                                                phantom_test=phantom_test,
                                                writeWingPosVariables=True,
                                                ignore_first_x_nodes_in_force_calculation=ignore_first_x_nodes_in_force_calculation)
            
            # Run simulation
            if not plot_only:
                wing_fuselage_model.run()
            
            # Get results
            deformation = nonlifting_utils.load_deformation_distribution(
                output_route + case_name,
                wing_fuselage_model.structure.n_node_right_wing
                )
            
            deformation[:,2] -= deformation[0, 2]
            deformation /= dict_geometry_parameters['half_wingspan']

            # Plot results
            if lifting_only:
                linestyle = '-'
                label = None
            else:
                linestyle = '-'
                label ="$\Lambda$ = {}".format(sigma)
            plt.plot(deformation[:,1], deformation[:,2], color=list_colors[isigma], linestyle=linestyle, label=label)


    plt.legend()
    plt.grid()
    plt.xlabel("y/s")
    plt.ylabel("z/s")
    plt.savefig(results_folder + '/{}_wing_deformation.png'.format(model))
    plt.show()

    # Clean up
    nonlifting_utils.tearDown(route_dir)

if __name__ == '__main__':
    list_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    run_fuselage_wing_configuration_coupled('mid_wing_1', plot_only=False) #low_wing
