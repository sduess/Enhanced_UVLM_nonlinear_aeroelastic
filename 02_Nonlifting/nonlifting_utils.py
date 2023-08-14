
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
    import os
    case_route = route_dir + '/cases/'
    output_route = route_dir + '/output/'

    results_folder = route_dir + '/result_data/'

    if not os.path.exists(case_route):
        os.makedirs(case_route)
    return case_route, output_route, results_folder
    