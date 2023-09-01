import os
import numpy as np
import matplotlib.pyplot as plt 
import sharpy
import nonlifting_utils

from sharpy.cases.templates.fuselage_wing_configuration.fuselage_wing_configuration import Fuselage_Wing_Configuration
from sharpy.cases.templates.fuselage_wing_configuration.fwc_get_settings import define_simulation_settings


def test_ellipsoid():
    """ 
    Computes the pressure distribution over an ellipsoid. The results should
    match the analyitcal solution according to potential flow theory (see Chapter 5
    Katz, J., and Plotkin, A., Low-speed aerodynamics, Vol. 13, Cambridge University 
    Press, 2001).
    """

    # define model variables
    radius_ellipsoid = 0.2
    length_ellipsoid = 2.
    u_inf = 10 
    alpha_deg = 0
    n_elem = 30
    num_radial_panels  = 24
    lifting_only= False
    fuselage_shape = 'ellipsoid'
    fuselage_discretisation = 'uniform'
    # define case name and folders
    route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    case_route, output_route, results_folder = nonlifting_utils.define_folder(route_dir)

    case_name = 'ellipsoid'
    enforce_uniform_fuselage_discretisation = True

    # generate ellipsoid model
    ellipsoidal_body = Fuselage_Wing_Configuration(case_name, case_route, output_route) 
    ellipsoidal_body.init_aeroelastic(lifting_only=lifting_only,
                                    max_radius=radius_ellipsoid,
                                fuselage_length=length_ellipsoid,
                                offset_nose_wing=length_ellipsoid/2,
                                n_elem_fuselage=n_elem,
                                num_radial_panels=num_radial_panels,
                                fuselage_shape=fuselage_shape,
                                enforce_uniform_fuselage_discretisation=enforce_uniform_fuselage_discretisation,
                                fuselage_discretisation=fuselage_discretisation)
    ellipsoidal_body.generate()
    
    # define settings
    flow = ['BeamLoader',
            'NonliftingbodygridLoader',
            'StaticUvlm',
            'WriteVariablesTime'
            ]
    settings = define_simulation_settings(flow, ellipsoidal_body, alpha_deg, u_inf, lifting_only=False, nonlifting_only=True, horseshoe=True)
    ellipsoidal_body.create_settings(settings)

    # run simulation
    ellipsoidal_body.run()

    # postprocess
    cp_distribution_SHARPy, cp_distribution_analytical, x_collocation_points = get_results(output_route, case_name, ellipsoidal_body)
    
    # plot
    plt.plot(x_collocation_points, cp_distribution_analytical, color = 'tab:blue', label = 'Analytical')
    plt.scatter(x_collocation_points, cp_distribution_SHARPy, color = 'tab:orange', label = 'SHARPy')
    plt.legend()
    
    plt.savefig(os.path.join(results_folder,
                             'ellipsoid_cp.png'))

    # clean up
    nonlifting_utils.tearDown(route_dir)

def load_pressure_distribution(output_folder, n_collocation_points):
    """
        Loads the resulting pressure coefficients saved in txt-files.
    """
    cp_distribution = np.zeros((n_collocation_points,))
    for i_collocation_point in range(n_collocation_points):
        cp_distribution[i_collocation_point] = np.loadtxt(output_folder + 'nonlifting_pressure_coefficients_panel_isurf0_im0_in{}.dat'.format(i_collocation_point))[1]
    return cp_distribution

def get_analytical_pressure_distribution(radius, x_coordinates):
    """
        Computes the analytical solution of the pressure distribution over
        an ellipsoid in potential flow for the previous specified ellipsoid
        model. Equations used are taken from
            https://www.symscape.com/examples/panel/potential_flow_ellipsoid
    """
    a = np.sqrt(1 - radius**2)
    b = 2 * ((1-a**2)/a**3) * (np.arctanh(a)-a)
    u = 2./(2.-b) * np.sqrt((1-x_coordinates**2)/(1-x_coordinates**2 * a**2))
    return 1-u**2

def get_results(output_route, case_name, ellipsoidal_body):
    """
        Loads the pressure distribution computed by SHARPy as well as the analytical solution
        on the collocation points of the ellipsoidal model.
    """
    cp_distribution_SHARPy = load_pressure_distribution(output_route + '/' + case_name + '/WriteVariablesTime/', 
                                                            ellipsoidal_body.structure.n_node_fuselage)
    
    length_ellipsoid = ellipsoidal_body.structure.fuselage_length
    dx = length_ellipsoid/(ellipsoidal_body.structure.n_node_fuselage-1)
    x_collocation_points = np.linspace(-length_ellipsoid/2+dx/2, length_ellipsoid/2-dx/2, ellipsoidal_body.structure.n_node_fuselage)
    cp_distribution_analytical = get_analytical_pressure_distribution(ellipsoidal_body.fuselage.max_radius, x_collocation_points)
    return cp_distribution_SHARPy, cp_distribution_analytical, x_collocation_points




if __name__ == '__main__':
    test_ellipsoid()
