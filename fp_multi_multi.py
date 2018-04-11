# fp_multi_multi.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import streamplot as sp
import multiplot_data as mpd
import heapq
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rcParams['axes.titlesize'] = 10

plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

plot_col = ['#801515', '#c90d0d', '#d26618', '#dfa524', '#cdeb14', '#7d9d10', '#1ff675', '#139a72', '#359ab5', '#075fd2', '#151fa4', '#3c33a3', '#7f05d4', '#b100de', '#ff8ac2']

#todo: parameters
dx_blocks = 20

#hack: path

# sub_dir = "s= " + print_s + ", h= " + print_h + ", q= " + print_q
# print sub_dir
# linear_dir_path = "../output/revival/local_fp_output/par_s_" + print_s + "_h_" + print_h +"/par_q_" + print_q + "/"
# outpath = linear_dir_path

param_s_nums = np.array([150.0])
param_s_strings = ['150']

param_h_nums = np.array([200.0, 400.0])
param_h_strings = ['200', '400']

# param_h_nums = np.array([200.0])
# param_h_strings = ['200']

param_q_nums = np.array([1.0, 3.0, 5.0, 10.0, 30.0, 50.0])
param_q_strings = ['1.0', '3.0', '5.0', '10.0', '30.0', '50.0']

#hack: big arrays
temp_top_linspace = np.zeros([dx_blocks,len(param_s_nums),len(param_h_nums),len(param_q_nums)])
temp_bottom_linspace = np.zeros([dx_blocks,len(param_s_nums),len(param_h_nums),len(param_q_nums)])

#hack: gen_path
gen_path = '../output/revival/local_fp_output/'

for m in range(len(param_s_nums)):
    for mm in range(len(param_h_nums)):
        for mmm in range(len(param_q_nums)):

            txt_path = "../output/revival/local_fp_output/par_s_" + param_s_strings[m] + "_h_" + param_h_strings[mm] +"/par_q_" + param_q_strings[mmm] + "/"

            km_linspace = np.loadtxt(txt_path + 'z_km_linspace.txt',delimiter='\n')
            age_linspace = np.loadtxt(txt_path + 'z_age_linspace.txt',delimiter='\n')
            temp_top_linspace[:,m,mm,mmm] = np.loadtxt(txt_path + 'z_temp_top_linspace.txt',delimiter='\n')
            temp_bottom_linspace[:,m,mm,mmm] = np.loadtxt(txt_path + 'z_temp_bottom_linspace.txt',delimiter='\n')



#todo: trial_fig
fig=plt.figure(figsize=(14.0,8.0))

ax=fig.add_subplot(2, 3, 1, frameon=True)

n_color = 0
for m in range(len(param_s_nums)):
    for mm in [0]:
        for mmm in range(len(param_q_nums)):
            plt.plot(km_linspace,temp_top_linspace[:,m,mm,mmm],label="s: " + param_s_strings[m] + ", h: " + param_h_strings[mm] +", q: " + param_q_strings[mmm], lw=2.0, color=plot_col[n_color])
            plt.plot(km_linspace,temp_bottom_linspace[:,m,mm,mmm], linestyle='-', lw=1.0,color=plot_col[n_color])
            n_color = n_color + 1

plt.ylim([0.0,120.0])
plt.xlabel('distance from inflow [km]',fontsize=9)
plt.ylabel('temp [C]',fontsize=9)
plt.title('param_h = 200')

plt.legend(fontsize=8,bbox_to_anchor=(-0.3, 0.7))



ax=fig.add_subplot(2, 3, 2, frameon=True)

for m in range(len(param_s_nums)):
    for mm in [1]:
        for mmm in range(len(param_q_nums)):
            plt.plot(km_linspace,temp_top_linspace[:,m,mm,mmm],label="s: " + param_s_strings[m] + ", h: " + param_h_strings[mm] +", q: " + param_q_strings[mmm], lw=2.0, color=plot_col[n_color])
            plt.plot(km_linspace,temp_bottom_linspace[:,m,mm,mmm], linestyle='-', lw=1.0,color=plot_col[n_color])
            n_color = n_color + 1

plt.ylim([0.0,120.0])
plt.xlabel('distance from inflow [km]',fontsize=9)
plt.ylabel('temp [C]',fontsize=9)
plt.title('param_h = 400')






ax=fig.add_subplot(2, 3, 4, frameon=True)

n_color = 0
for m in range(len(param_s_nums)):
    for mm in range(len(param_h_nums)):
        for mmm in range(len(param_q_nums)):
            plt.plot(km_linspace,temp_top_linspace[:,m,mm,mmm],label="s: " + param_s_strings[m] + ", h: " + param_h_strings[mm] +", q: " + param_q_strings[mmm], lw=1.25, color=plot_col[n_color])
            n_color = n_color + 1

plt.ylim([0.0,120.0])
plt.xlabel('distance from inflow [km]',fontsize=9)
plt.ylabel('temp [C]',fontsize=9)
plt.title('temp_top_linspace')
plt.legend(fontsize=8,bbox_to_anchor=(-0.3, 0.7))



ax=fig.add_subplot(2, 3, 5, frameon=True)

n_color = 0
for m in range(len(param_s_nums)):
    for mm in range(len(param_h_nums)):
        for mmm in range(len(param_q_nums)):
            plt.plot(km_linspace,temp_bottom_linspace[:,m,mm,mmm], linestyle='-', lw=1.25,color=plot_col[n_color])
            n_color = n_color + 1

plt.ylim([0.0,120.0])
plt.xlabel('distance from inflow [km]',fontsize=9)
plt.ylabel('temp [C]',fontsize=9)
plt.title('temp_bottom_linspace')


plt.savefig(gen_path+'fpmm_trial.png',bbox_inches='tight')
