import numpy as np
import matplotlib.pyplot as plt
import os

route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def plot_data(list_gust_lengths, list_gust_intensity, list_cfl1, list_polars, case_string_format,results_folder):
    fig, axs = plt.subplots(4)
    for i_gust_length, gust_length in enumerate(list_gust_lengths):
        for gust_intensity in list_gust_intensity:
            for cfl1 in list_cfl1:
                for polars in list_polars:
                    file = results_folder +case_string_format.format(gust_length, gust_intensity, polars, cfl1) + '.txt'
                    data = np.loadtxt(file,
                                      delimiter=",")
                    data[1:,1] /= 2
                    for iparameter in range(np.shape(data)[1] - 1):
                        axs[iparameter].plot(data[1:,0], 
                                             data[1:,iparameter + 1], 
                                             color = list_colors[i_gust_length],
                                             linestyle = get_linestyle(cfl1, polars), 
                                             label= get_label(cfl1, polars, iparameter, gust_length))
    for iax in range(len(axs)):
        plt.setp(axs[iax].get_xticklabels(), visible=iax == len(axs)-1)
        axs[iax].set_ylabel(list_ylabels[iax])
        axs[iax].set_xlim([0.,2.5])
    plt.xlabel("time, s")
    plt.tight_layout()
    axs[0].legend(ncols=len(list_gust_lengths), loc='upper center', bbox_to_anchor=(0.7, 1.2))
    plt.savefig(results_folder + 'dynamic_gust_response.png')
    plt.show()
                    
def get_label(cfl1, polars, iparameter, gust_length):
    if cfl1 or polars or iparameter >0:
        return None
    else:
        return 'H = {} m'.format(gust_length)
    

def get_linestyle(cfl1, polar):
    if bool(cfl1) and not bool(polar):
        return '-'
    elif bool(polar):
        return ':'
    else:
        return '--'
def main():         
    results_folder = route_dir + '/results_data/'
    case_string_format =  'superflexop_free_gust_L_{:g}_I_{:g}_p_{:g}_cfl_{:g}'

    list_gust_lengths = [5, 10, 20, 40, 80, 100]
    list_gust_intensity = [10]
    list_cfl1 = [0, 1] 
    list_polars = [0] 
    plot_data(list_gust_lengths, list_gust_intensity, list_cfl1, list_polars, case_string_format, results_folder)


    

if __name__ == '__main__':
    list_ylabels = ['$z_{tip}/s$, %', '$M_{OOP}$, Nm$^2$', '$M_{T}$, Nm$^2$', '$\Theta$, deg']
    list_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    list_linestyles = ['-', '--', ':']
    main()