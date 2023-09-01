import os
import numpy as np
import nonlifting_utils
import json
import matplotlib.pyplot as plt

def run_fuselage_wing_configuration_coupled(model):
    """
        Test spanwise lift distribution on a fuselage-wing configuration.
        
        The lift distribution on low wing configuration is computed considering
        fuselage effects. The final results are compared to a previous solution 
        (backward compatibility) that matches the experimental lift distribution for this case. 
    """

    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    case_route, output_route, results_folder = nonlifting_utils.define_folder(route_dir)
    model = 'mid_wing'
    fuselage_length = 10
    dict_geometry_parameters = nonlifting_utils.get_geometry_parameters(model,
                                                       route_dir,
                                                        fuselage_length=fuselage_length)

    # Freestream Conditions
    alpha_deg = 2.9
    u_inf = 10
    # sigma = 10

    # Discretization
    dict_discretization = {
        'n_elem_per_wing': 10,
        'n_elem_fuselage': 30,
        'num_chordwise_panels': 8,
        'num_radial_panels': 36
    }

    # Simulation settings
    horseshoe = True
    phantom_test = False
    lifting_only = False
    # Simlation Solver Flow
    flow = ['BeamLoader',
            'AerogridLoader',
            'NonliftingbodygridLoader',
            # 'StaticUvlm',
            'StaticCoupled',
            'BeamLoads',
            'LiftDistribution',
            'AerogridPlot',
            'WriteVariablesTime'
                ]
    list_sigmas = [0.075, 0.1, 0.15, 0.25, 0.5, 1.5]
    for lifting_only in [False, True]:
        if lifting_only:
            linestyle = '-'
        else:
            linestyle = '--'
        for sigma in list_sigmas:
            case_name = '{}_coupled_lifting_only{}_sigma{}'.format(model, int(lifting_only), sigma*1000)
            wing_fuselage_model = nonlifting_utils.generate_model(case_name, 
                                                dict_geometry_parameters,
                                                dict_discretization, 
                                                lifting_only,
                                                case_route,
                                                output_route,
                                                sigma=sigma)

            flow_case = flow.copy()
            if lifting_only:
                flow_case.remove('NonliftingbodygridLoader')
            nonlifting_utils.generate_simulation_settings(flow_case, 
                                                wing_fuselage_model, 
                                                alpha_deg, 
                                                u_inf, 
                                                lifting_only,
                                                horseshoe=horseshoe,
                                                phantom_test=phantom_test)
            # run simulation
            wing_fuselage_model.run()
            # get results
            deformation = nonlifting_utils.load_deformation_distribution(
                output_route + case_name,
                wing_fuselage_model.structure.n_node_right_wing
                )
            deformation /= np.max(deformation[:,1])
            deformation[:,2] -= deformation[0, 2]
            # plot results
            if lifting_only:
                label ="$\Lambda$ = {}".format(sigma)
            else:
                label = None
            plt.plot(deformation[:,1], deformation[:,2], linestyle=linestyle, label = label)
    plt.legend()
    plt.savefig(results_folder + '/')
    plt.show()

    # clean up
    # nonlifting_utils.tearDown(route_dir)

if __name__ == '__main__':
    run_fuselage_wing_configuration_coupled('low_wing') #'mid_wing'