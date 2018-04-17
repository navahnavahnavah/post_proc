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

plot_col = ['#801515', '#c90d0d', '#d26618', '#EDB92E', '#cdeb14', '#7d9d10', '#1ff675', '#139a72', '#48b4d1', '#075fd2', '#7f05d4', '#b100de', '#ff8ac2']

plot_col_bold = ['#801515', '#c90d0d', '#ec9c14', '#1ff675', '#1473ee', '#8c06e9', '#b100de', '#ff8ac2']

#todo: parameters
dx_blocks = 20


# sub_dir = "s= " + print_s + ", h= " + print_h + ", q= " + print_q
# print sub_dir
# linear_dir_path = "../output/revival/local_fp_output/par_s_" + print_s + "_h_" + print_h +"/par_q_" + print_q + "/"
# outpath = linear_dir_path

param_s_nums = np.array([200.0])
param_s_strings = ['200']

param_h_nums = np.array([200.0, 400.0, 600.0])
param_h_strings = ['200', '400', '600']

# param_h_nums = np.array([200.0])
# param_h_strings = ['200']

# param_q_nums = np.array([1.0, 3.0, 5.0, 10.0, 30.0, 50.0])
# param_q_strings = ['1.0', '3.0', '5.0', '10.0', '30.0', '50.0']

param_q_nums = np.array([1.0, 3.0, 5.0, 10.0, 30.0])
param_q_strings = ['1.0', '3.0', '5.0', '10.0', '30.0']

#hack: big arrays
temp_top_linspace = np.zeros([dx_blocks,len(param_s_nums),len(param_h_nums),len(param_q_nums)])
temp_bottom_linspace = np.zeros([dx_blocks,len(param_s_nums),len(param_h_nums),len(param_q_nums)])

#hack: gen_path
gen_path = '../output/revival/local_fp_output/'
unique_string = 'k_10_s_' + param_s_strings[0]

#hack: site data
x_sd_temps = np.array([22.443978220690354, 25.50896648184225, 33.32254358359559, 39.22503621559518, 44.528597832059546, 54.624706528797645, 74.32349268195217, 100.7522853289375, 102.99635346420898, 100.74349368100305])
x_sd_temps_km = np.array([22.443978220690354, 25.50896648184225, 33.32254358359559, 39.22503621559518, 44.528597832059546, 54.624706528797645, 74.32349268195217, 100.7522853289375, 102.99635346420898])
age_sd_temps = np.array([0.86, 0.97, 1.257, 1.434, 1.615, 1.952, 2.621, 3.511, 3.586])
for j in range(len(x_sd_temps)):
    x_sd_temps[j] = (x_sd_temps[j]-20.0)*1000.0
for j in range(len(x_sd_temps_km)):
    x_sd_temps_km[j] = x_sd_temps_km[j] - 20.0
y_sd_temps = np.array([15.256706129177289, 22.631899695289484, 38.471851740846205, 39.824366851491085, 50.20180828213198, 58.10639892102503, 56.69024426794546, 60.72611019531446, 62.36115690094412, 62.91363204955294])
y_sd_temps_km = np.array([15.256706129177289, 22.631899695289484, 38.471851740846205, 39.824366851491085, 50.20180828213198, 58.10639892102503, 56.69024426794546, 60.72611019531446, 62.36115690094412])

for m in range(len(param_s_nums)):
    for mm in range(len(param_h_nums)):
        for mmm in range(len(param_q_nums)):

            txt_path = "../output/revival/local_fp_output/par_k_10_s_" + param_s_strings[m] + "_h_" + param_h_strings[mm] +"/par_q_" + param_q_strings[mmm] + "/"

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

plt.plot(x_sd_temps_km,y_sd_temps_km,'k^',label="data")

plt.ylim([0.0,120.0])
plt.xlabel('distance from inflow [km]',fontsize=9)
plt.ylabel('temp [C]',fontsize=9)
plt.title('param_h = 200')

plt.legend(fontsize=8,bbox_to_anchor=(-0.15, 0.7))



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






dashes = [6, 3]
thick_line = 1.25
thin_line = 0.75
dashes0 = [2, 3]

ax=fig.add_subplot(2, 3, 4, frameon=True)


for mm in range(len(param_h_nums)):
    n_color = 0
    for m in range(len(param_s_nums)):
        for mmm in range(len(param_q_nums)):
            the_line, = plt.plot(km_linspace,temp_top_linspace[:,m,mm,mmm],label="s: " + param_s_strings[m] + ", h: " + param_h_strings[mm] +", q: " + param_q_strings[mmm], lw=thin_line, color=plot_col_bold[n_color])
            if mm == 1:
                the_line.set_dashes(dashes)
                the_line.set_linewidth(thin_line)
            if mm == 2:
                the_line.set_dashes(dashes0)
                the_line.set_linewidth(thick_line)
            n_color = n_color + 1

plt.plot(x_sd_temps_km,y_sd_temps_km,markersize=6,marker='^',label="data",lw=0,markerfacecolor='none',markeredgecolor='k',markeredgewidth=1.0)

plt.xlim([-5.0,100.0])
plt.ylim([0.0,120.0])
plt.xlabel('distance from inflow [km]',fontsize=9)
plt.ylabel('temp [C]',fontsize=9)
plt.title('temp_top_linspace')
plt.legend(fontsize=8,bbox_to_anchor=(-0.15, 0.7))



ax=fig.add_subplot(2, 3, 5, frameon=True)


for mm in range(len(param_h_nums)):
    n_color = 0
    for m in range(len(param_s_nums)):
        for mmm in range(len(param_q_nums)):
            the_line, = plt.plot(km_linspace,temp_bottom_linspace[:,m,mm,mmm], linestyle='-', lw=thin_line, color=plot_col_bold[n_color])
            if mm == 1:
                the_line.set_dashes(dashes)
                the_line.set_linewidth(thin_line)
            if mm == 2:
                the_line.set_dashes(dashes0)
                the_line.set_linewidth(thick_line)
            n_color = n_color + 1

plt.plot(x_sd_temps_km,y_sd_temps_km,markersize=6,marker='^',label="data",lw=0,markerfacecolor='none',markeredgecolor='k',markeredgewidth=1.0)

plt.xlim([-5.0,100.0])
plt.ylim([0.0,120.0])
plt.xlabel('distance from inflow [km]',fontsize=9)
plt.ylabel('temp [C]',fontsize=9)
plt.title('temp_bottom_linspace')




ax=fig.add_subplot(2, 3, 6, frameon=True)


for mm in range(len(param_h_nums)):
    n_color = 0
    for m in range(len(param_s_nums)):
        for mmm in range(len(param_q_nums)):
            the_line, = plt.plot(age_linspace,temp_bottom_linspace[:,m,mm,mmm], linestyle='-', lw=thin_line, color=plot_col_bold[n_color])
            if mm == 1:
                the_line.set_dashes(dashes)
                the_line.set_linewidth(thin_line)
            if mm == 2:
                the_line.set_dashes(dashes0)
                the_line.set_linewidth(thick_line)
            n_color = n_color + 1

# plt.plot(x_sd_temps_km,y_sd_temps_km,markersize=6,marker='^',label="data",lw=0,markerfacecolor='none',markeredgecolor='k',markeredgewidth=1.0)

plt.xlim([0.5,4.0])
plt.ylim([0.0,120.0])
plt.xlabel('age_linspace [Myr]',fontsize=9)
plt.ylabel('temp [C]',fontsize=9)
plt.title('temp_bottom_linspace, f(age_linspace)')



plt.subplots_adjust(hspace=0.4)

plt.savefig(gen_path+'fpmm_trial_'+unique_string+'.png',bbox_inches='tight')
plt.savefig(gen_path+'fpmm_trial_'+unique_string+'.eps',bbox_inches='tight')




# #todo: iso_fig
# fig=plt.figure(figsize=(14.0,8.0))
#
# ax=fig.add_subplot(2, 3, 1, frameon=True)
#
#
#
# plt.savefig(gen_path+'fpmm_trial.png',bbox_inches='tight')
# plt.savefig(gen_path+'fpmm_trial.eps',bbox_inches='tight')
