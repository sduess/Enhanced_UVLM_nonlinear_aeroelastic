import numpy as np
import matplotlib.pyplot as plt
import os

route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def plot_data(list_gust_lengths, list_gust_intensity, list_cfl1, list_polars, case_string_format,results_folder):
    
    # TODO: labels (dict with keys to find units and latex parameter), legend
    fig, axs = plt.subplots(4)
    for gust_length in list_gust_lengths:
        for gust_intensity in list_gust_intensity:
            for cfl1 in list_cfl1:
                for polars in list_polars:
                    file = results_folder +case_string_format.format(gust_length, gust_intensity, polars, cfl1) + '.txt'
                    data = np.loadtxt(file,
                                      delimiter=",")
                    for iparameter in range(np.shape(data)[1] - 1):
                        axs[iparameter].plot(data[:,0], data[:,iparameter + 1])
    plt.savefig(results_folder + 'dynamic_gust_response.png')
                    

def main():         
    results_folder = route_dir + '/results_data/'
    case_string_format =  'superflexop_free_gust_L_{:g}_I_{:g}_p_{:g}_cfl_{:g}'
    list_gust_lengths = [10] # [5, 10, 20, 40, 80, 100]
    list_gust_intensity = [10]
    list_cfl1 = [0] #[0, 1]
    list_polars = [0] #[0, 1]
    plot_data(list_gust_lengths, list_gust_intensity, list_cfl1, list_polars, case_string_format, results_folder)


    

if __name__ == '__main__':
    main()