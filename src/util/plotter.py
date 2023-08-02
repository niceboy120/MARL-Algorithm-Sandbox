

    
    
# system libs
import sys
import math

from random import Random
random = Random(1337)

# dependencies - numpy, matplotlib
import numpy as np 
from numpy.random import rand
from numpy import NaN, log, dot, e
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage,AnnotationBbox

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# local files
# from utility import *

np.set_printoptions(threshold=sys.maxsize)


    
    
def plot_algo_compare(plot_info, results):
    """Plot a graph comparing the performance of a set of MARL algorithms


    Args:
        plot_info (dict): contains values for experiment parameters
        results (list): a list of results from a series of MARL experiments
    """
    env_name = plot_info["env_name"]
    algos = plot_info["algorithms"]
    num_episodes = plot_info["max_episodes"]

    # x_axis = ['value_1', 'value_2', 'value_3', ...]
    # y_axis = ['value_1', 'value_2', 'value_3', ...]

    episode_i, scores, steps = [],[],[]

    # plot
    plt.clf()
    plt.suptitle('Total Rewards Received During Episode')
    plt.title(f"averaged over 5 tests")
    plt.xlabel('Episodes')
    plt.ylabel('Sum of episode rewards')
    #plt.ylim(y_lim)

    log_interval = 100

    x = [i for i in range(num_episodes // log_interval)]
    
    for result in results:
        plot_data = np.load(f"../results/{result}")
        print(plot_data.shape)
        # max_plot_data = np.max(plot_data, axis=0)
        # avg_plot_data = np.mean(plot_data, axis=0).transpose()
        # std_plot_data = np.std(plot_data, axis=0).transpose()
        # min_plot_data = np.min(plot_data, axis=0)
        # print(plot_data[0])
        y = plot_data.transpose()[0]

        # for datum in plot_data[1::]:
        #     episode_i.append(datum[0])
        #     scores.append(datum[1])
        #     steps.append(datum[2])
        
        # y = scores

        # y = avg_plot_data[1]
        # y_std = std_plot_data[1]

        plt.plot(x, y)
        # plt.fill_between(x, y+y_std, y-y_std, facecolor='yellow', alpha=0.5)

    # # plt.axhline(y=hbar, color='y', linestyle='--')
    # plt.legend([r"$\mu$", r"$\pm\sigma$"])
    # plt.grid()
    # # plt.show()
    # plt.savefig(f"./results/rewards_{rel_str}_{strat_str}_{custom_env_options['grid_size']}_grid_{custom_sim_options['num_runs']}_runs.svg", format="svg")

    # plt.clf()
    # plt.suptitle('Total Steps to Complete Episode')
    # plt.title(f"averaged over {custom_sim_options['num_runs']} runs")
    # plt.xlabel('Episodes')
    # plt.ylabel('Step count')
    # #plt.ylim(y_lim)



    #     plt.plot(x, y2)
    #     # plot errors
    #     #plt.errorbar(x, y, yerr=asymmetric_error, fmt='-o')
    #     plt.fill_between(x, y2+y2_std, y2-y2_std, facecolor='yellow', alpha=0.5)

    #     # plt.axhline(y=hbar, color='y', linestyle='--')
    #     plt.legend([r"$\mu$", r"$\pm\sigma$"])
    #     plt.grid()
    #     # plt.show()
    #     plt.savefig(f"./results/steps_{rel_str}_{strat_str}_{custom_env_options['grid_size']}_grid_{custom_sim_options['num_runs']}_runs.svg", format="svg")

    # plt.title('title name')
    # plt.xlabel('x_axis name')
    # plt.ylabel('y_axis name')

    # plt.show()


# def get_int_from_binary_array(bin_array):
#     # convert he binary array into a string
#     bin_array_str = str(bin_array).strip()
#     # loop over certain sub characters and replace them with empty string
#     sub_chars = " [],"
#     for char in sub_chars:
#         bin_array_str = bin_array_str.replace(char,'')
#     return int(bin_array_str, 2)


# class MatplotGraph:
#     x_series = []

#     def __init__(self, rows, cols, sup_title):
#         self.rows = rows
#         self.cols = cols
#         self.fig, self.axes = plt.subplots(rows, cols, constrained_layout=True)
#         self.fig.suptitle(sup_title, fontsize=16)
                 
#     def plot_subgraph(self, x_data, func, x_label, y_label, title, axes_num=0, has_images=False, sort_mode="alphanumeric", sort_var=0):
#         axis = self.axes
#         if self.rows > 1 or self.cols > 1:
#             axis = self.axes[axes_num]
#         y_series = []
#         self.x_series = []
#         for x_val in x_data: 
#             x, y = func(x_val)
#             y_series.append(y)
#             self.x_series.append(str(x))
#         series_data = zip(self.x_series, y_series)
#         series_list = list(series_data)
#         if(sort_mode is "binary"):
#             series_list.sort(key=lambda x: get_int_from_binary_array(x[sort_var]))
#         else: 
#             series_list.sort(key=lambda x:x[0], reverse=True)
#             series_list.sort(key=lambda x:x[sort_var])
#         unzipped_data = list(zip(*series_list))
#         rects = axis.bar(unzipped_data[0], unzipped_data[1])
#         axis.tick_params('x', labelrotation=90)
#         axis.set_title(title)
#         axis.set_xlabel(x_label)
#         axis.set_ylabel(y_label)
#         # axis.yaxis.set_label_coords(-6,1.00)
#         # add text to the graph if the x_series isnt too large
#         for i, rect in enumerate(rects):
#             text_y_offset = 0.0
#             # if height > 0:
#             #     text_y_offset = 1.1*height
#             # else:
#             #     text_y_offset = 0.0
#             text_x_offset = rect.get_x() + rect.get_width()/2.
#             text_y_offset = rect.get_height()/2.
#             axis.text(text_x_offset, text_y_offset,str(round(float(series_list[i][1]),1)), ha='center', va='bottom')

#         if has_images:
#             # for i, site_str in enumerate(self.x_series):
#             sorted_x = list(zip(*series_list))[0]
#             for i, site_str in enumerate(sorted_x):
#                 site_config_label = site_str[::-1].strip(' []').replace(',','').replace(' ','')
#                 min_y = np.min(y_series)
#                 base_offset = 0.0
#                 if(min_y < 0.0):
#                     base_offset = min_y
#                 offset_image(i, site_config_label, axis, base_offset)
                
#     def set_figure_size(self, h, w):
#         self.fig.set_figheight(h)
#         self.fig.set_figwidth(w)

#     def save_to_file(self, file_name):
#         plt.tight_layout()
#         plt.savefig(file_name)
 

# def get_site_image(name):
#     # path = "D:\\UML_PhD_Program\\Exalabs_Research\\traffic_observability_project\\figures\\site_configurations\\1x\\{}.png".format(name)
#     path = "./figures/site_configurations/1x/{}.png".format(name)
#     im = plt.imread(path)
#     return im

# def offset_image(coord, name, ax, base_offset=0.0):
#     img = get_site_image(name)
#     im = OffsetImage(img, zoom=0.52)
#     im.image.axes = ax

#     y_offset = 0.0
#     if len(name) > 5:
#         y_offset = -45.0
#     elif len(name) > 3:
#         y_offset = -35.0
#     else:
#         y_offset = -25.0
#     ab = AnnotationBbox(im, (coord, base_offset),  xybox=(0., y_offset), frameon=False,
#                         xycoords='data',  boxcoords="offset points", pad=0)
#     # ax.axes.xaxis.set_visible(False)
#     # ax.axes.xaxis.set_ticklabels([])
#     ax.add_artist(ab)