import numpy as np
import matplotlib.pyplot as plt
import os

route_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def plot_data(list_gust_lengths, list_gust_intensity, list_cfl1, list_polars, case_string_format,results_folder):
    fig, axs = plt.subplots(4,figsize=(10,6))
    for i_gust_length, gust_length in enumerate(list_gust_lengths):
        for gust_intensity in list_gust_intensity:
            for cfl1 in list_cfl1:
                for polars in list_polars:
                    file = results_folder +case_string_format.format(gust_length, gust_intensity, polars, cfl1) + '.txt'
                    data = np.loadtxt(file,
                                      delimiter=",")
                    for iparameter in range(np.shape(data)[1] - 1):
                        axs[iparameter].plot(data[2:,0], 
                                             data[2:,iparameter + 1], 
                                             color = list_colors[i_gust_length],
                                             linestyle = get_linestyle(cfl1, polars))
    for iax in range(len(axs)):
        plt.setp(axs[iax].get_xticklabels(), visible=iax == len(axs)-1)
        axs[iax].set_ylabel(list_ylabels[iax])
        axs[iax].set_xlim([0.,2.5])
        axs[iax].grid()
    handles = get_legend(list_gust_lengths, 1 in list_polars)

    plt.xlabel("time, s")
    plt.tight_layout()
    lgd = axs[1].legend(ncols=1, #len(list_gust_lengths), 
                        loc='upper right', 
                        handles = handles[:len(list_gust_lengths)],
                        bbox_to_anchor=(1.0, 1.2))
                        
    lgd = axs[0].legend(ncols=2, loc='upper right', handles = handles[-2:],
                        bbox_to_anchor=(1.0, 1.))
    str_name_extension = ''
    if 1 in list_polars:
        str_name_extension += '_polars'

    plt.savefig(results_folder + 'dynamic_gust_response{}.png'.format(str_name_extension), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

def get_legend(list_gust_length, flag_polar_study=False):
    from matplotlib.lines import Line2D
    list_labels = []
    for igust, gust_length in enumerate(list_gust_length):
        list_labels.append(Line2D([0], [0],color=list_colors[igust], label = "H = {} m".format(gust_length)))
    if flag_polar_study:
        list_labels.append(Line2D([0], [0],color='k', linestyle='dashed', label='SHARPy')) 
        list_labels.append(Line2D([0], [0],color='k', linestyle='solid', label='SHARPy + Polar Corrections'))

    else:
        list_labels.append(Line2D([0], [0],color='k', linestyle='dashed', label='uniform wake discretisation')) 
        list_labels.append(Line2D([0], [0],color='k', linestyle='solid', label='variable wake discretisation'))
    return list_labels

def get_linestyle(cfl1, polar):
    if bool(cfl1) or bool(polar):
        return '-'
    else:
        return '--'
def main():         
    results_folder = route_dir + '/results_data/'
    case_string_format =  'superflexop_free_gust_L_{:g}_I_{:g}_p_{:g}_cfl_{:g}u_inf20'

    list_gust_lengths = [5, 10, 20, 40, 80] #, 100]
    list_gust_intensity = [10]
    list_cfl1 = [0] 
    list_polars = [0,1] 
    plot_data(list_gust_lengths, list_gust_intensity, list_cfl1, list_polars, case_string_format, results_folder)


    

if __name__ == '__main__':
    list_ylabels = ['$z_{tip}/s$, %', '$M_{OOP}$, Nm$^2$', '$M_{T}$, Nm$^2$', '$\Theta$, deg']
    list_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    list_linestyles = ['-', '--', ':']
    main()