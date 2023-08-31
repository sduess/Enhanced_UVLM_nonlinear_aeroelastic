import os
import h5py as h5
import numpy as np
from sharpy.utils.algebra import quat2euler

route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def get_time_history(output_folder, case):
    file = os.path.join(output_folder,
                         case, 
                        'savedata', 
                        case + '.data.h5')   
    with h5.File(file, "r") as f:
            ts_max = len(f['data']['structure']['timestep_info'].keys())-2
            dt = float(str(np.array(f['data']['settings']['DynamicCoupled']['dt'])))
            matrix_data = np.zeros((ts_max, 5)) # parameters: time, tip displacement, wing root OOP and torsional bending, pitch
            matrix_data[:,0] = np.array(list(range(ts_max))) * dt
            node_tip = np.argmax(np.array(f['data']['structure']['timestep_info']['00000']['pos'])[:,1])
            half_wingspan = 7.07/2
            node_root = 0
            for its in range(1,ts_max):
                ts_str = f'{its:05d}'
                matrix_data[its, 1] = np.array(f['data']['structure']['timestep_info'][ts_str]['pos'])[node_tip, 2]/half_wingspan*100 # normalised tip z position in percent
                matrix_data[its, 2] = np.array(f['data']['structure']['timestep_info'][ts_str]['postproc_cell']['loads'])[node_root,4] # OOP
                matrix_data[its, 3] = np.array(f['data']['structure']['timestep_info'][ts_str]['postproc_cell']['loads'])[node_root,3] # OOP
                matrix_data[its, 4] = np.rad2deg(quat2euler(np.array(f['data']['structure']['timestep_info'][ts_str]['quat'])))[1] # Pitch
    return matrix_data

def get_header_with_parameter_and_unit(dict_parameters_info):
    header_parameter = 'time'
    for ilabel in range(len(dict_parameters_info['para_labels'])):
         header_parameter += ', {}'.format(dict_parameters_info['para_labels'][ilabel])
    return header_parameter
     
def write_results(data, case,dict_parameters_info, result_folder):  
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    np.savetxt(os.path.join(os.path.join(result_folder, case + '.txt')), 
               data,
               fmt='%10e,' * (np.shape(data)[1] - 1) + '%10e', 
               delimiter=", ", 
               header= get_header_with_parameter_and_unit(dict_parameters_info))

def main():    
    u_inf = 45    
    case_string_format =  'superflexop_free_gust_L_{:g}_I_{:g}_p_{:g}_cfl_{:g}'
    if u_inf == 20:
        case_string_format += 'u_inf20'
    list_gust_lengths = [5, 10, 20, 40, 80, 100]
    list_gust_intensity = [10]
    list_cfl1 = [1, 0]
    list_polars = [0]
    
    SHARPY_output_folder = './output/'
    result_folder = route_dir + '/results_data/'
    dict_parameters_info = {
                                'para_labels': ['z/s', 'OOP', 'MT', 'Pitch'],
                           }
    for gust_length in list_gust_lengths:
        for gust_intensity in list_gust_intensity:
            for cfl1 in list_cfl1:
                for polars in list_polars:
                    case_name = case_string_format.format(gust_length, gust_intensity, polars, cfl1)
                    data = get_time_history(SHARPY_output_folder,  case_name)
                    write_results(data, case_name,dict_parameters_info, result_folder)


if __name__ == '__main__':
    main()
