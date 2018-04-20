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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rcParams['axes.titlesize'] = 10



plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

plot_col = ['#801515', '#c90d0d', '#d26618', '#EDB92E', '#cdeb14', '#7d9d10', '#1ff675', '#139a72', '#48b4d1', '#075fd2', '#7f05d4', '#b100de', '#ff8ac2']

plot_col_bold = ['#801515', '#c90d0d', '#ec9c14', '#1ff675', '#1473ee', '#8c06e9', '#b100de', '#ff8ac2', '#838383']

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
unique_string = 'k_13_s_' + param_s_strings[0]

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

            txt_path = "../output/revival/local_fp_output/par_k_13_s_" + param_s_strings[m] + "_h_" + param_h_strings[mm] +"/par_q_" + param_q_strings[mmm] + "/"

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





#todo: 3D plot goes here

fig = plt.figure(figsize=(20.0,11.0))

#dashes collection
the_dash0 = [6, 3]
the_dash1 = [2, 4]
the_dash2 = [2, 2]

## NEW INTERP STUFF GOES HERE ##

plot_iter = np.arange(len(param_q_nums))
plot_iter_back = plot_iter[::-1]
plot_iter_back_negative = -1.0*plot_iter_back
print "len(plot_iter_back)" , len(plot_iter_back)

short_length = len(km_linspace)
start_point = 3

xs = km_linspace[start_point:short_length]



print "temp_top_linspace[start_point:short_length,0,0,:].shape" , temp_top_linspace[start_point:short_length,0,0,:].shape
print "xs.shape", xs.shape
print "param_q_nums.shape" , param_q_nums.shape

the_f_3d_top = interpolate.interp2d(param_q_nums, xs, temp_top_linspace[start_point:short_length,0,0,:], kind="linear")
the_f_3d_bottom = interpolate.interp2d(param_q_nums, xs, temp_bottom_linspace[start_point:short_length,0,0,:], kind="linear")


#hack: q paths go here, len(xs) long
n_q_paths = 8
path_of_q = np.zeros([len(xs), n_q_paths])
path_of_q_plot = np.zeros([len(xs),n_q_paths])

path_of_q_80 = np.zeros([len(xs), n_q_paths])
path_of_q_80_plot = np.zeros([len(xs),n_q_paths])

path_of_q_60 = np.zeros([len(xs), n_q_paths])
path_of_q_60_plot = np.zeros([len(xs),n_q_paths])

path_of_q_40 = np.zeros([len(xs), n_q_paths])
path_of_q_40_plot = np.zeros([len(xs),n_q_paths])

path_of_q_20 = np.zeros([len(xs), n_q_paths])
path_of_q_20_plot = np.zeros([len(xs),n_q_paths])



model_interp_3d_top = np.zeros([len(xs),n_q_paths])
model_interp_3d_bottom = np.zeros([len(xs),n_q_paths])

model_interp_3d_top_80 = np.zeros([len(xs),n_q_paths])
model_interp_3d_bottom_80 = np.zeros([len(xs),n_q_paths])

model_interp_3d_top_60 = np.zeros([len(xs),n_q_paths])
model_interp_3d_bottom_60 = np.zeros([len(xs),n_q_paths])

model_interp_3d_top_40 = np.zeros([len(xs),n_q_paths])
model_interp_3d_bottom_40 = np.zeros([len(xs),n_q_paths])

model_interp_3d_top_20 = np.zeros([len(xs),n_q_paths])
model_interp_3d_bottom_20 = np.zeros([len(xs),n_q_paths])

interp_shifts_3d = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

path_of_q_labels = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

# 0th path_of_q
# path_of_q[:len(xs)/2,0] = 3.0
# path_of_q[len(xs)/2:,0] = 5.0
path_of_q[:,0] = np.linspace(3.0,1.0,len(xs))
path_of_q_80[:,0] = np.linspace(3.0,1.0,len(xs))
path_of_q_60[:,0] = np.linspace(3.0,1.0,len(xs))
path_of_q_40[:,0] = np.linspace(3.0,1.0,len(xs))
path_of_q_20[:,0] = np.linspace(3.0,1.0,len(xs))

path_of_q_labels[0] = '3 to 1'

# 1st path_of_q
path_of_q[:,1] = np.linspace(5.0,1.0,len(xs))
path_of_q_80[:,1] = np.linspace(5.0,1.0,len(xs))
path_of_q_60[:,1] = np.linspace(5.0,1.0,len(xs))
path_of_q_40[:,1] = np.linspace(5.0,1.0,len(xs))
path_of_q_20[:,1] = np.linspace(5.0,1.0,len(xs))

path_of_q_labels[1] = '5 to 1'

# 2nd path_of_q
path_of_q[:,2] = np.linspace(10.0,1.0,len(xs))
path_of_q_80[:,2] = np.linspace(10.0,1.0,len(xs))
path_of_q_60[:,2] = np.linspace(10.0,1.0,len(xs))
path_of_q_40[:,2] = np.linspace(10.0,1.0,len(xs))
path_of_q_20[:,2] = np.linspace(10.0,1.0,len(xs))

path_of_q_labels[2] = '10 to 1'

# 3rd path_of_q
path_of_q[:,3] = np.linspace(30.0,1.0,len(xs))
path_of_q_80[:,3] = np.linspace(30.0,1.0,len(xs))
path_of_q_60[:,3] = np.linspace(30.0,1.0,len(xs))
path_of_q_40[:,3] = np.linspace(30.0,1.0,len(xs))
path_of_q_20[:,3] = np.linspace(30.0,1.0,len(xs))

path_of_q_labels[3] = '30 to 1'

# 4th path_of_q
path_of_q[:,4] = np.linspace(1.0,1.0,len(xs))
path_of_q_80[:,4] = np.linspace(1.0,1.0,len(xs))
path_of_q_60[:,4] = np.linspace(1.0,1.0,len(xs))
path_of_q_40[:,4] = np.linspace(1.0,1.0,len(xs))
path_of_q_20[:,4] = np.linspace(1.0,1.0,len(xs))

path_of_q_labels[4] = '1 to 1'



# turn path_of_q into plottable paths
interp_to_y = interpolate.interp1d(param_q_nums, plot_iter)

for n in range(n_q_paths):
    if np.max(path_of_q[:,n]) > 0.0:
        path_of_q_plot[:,n] = -1.0*interp_to_y(path_of_q[:,n])
    if np.max(path_of_q_80[:,n]) > 0.0:
        path_of_q_80_plot[:,n] = -1.0*interp_to_y(path_of_q_80[:,n])
    if np.max(path_of_q_60[:,n]) > 0.0:
        path_of_q_60_plot[:,n] = -1.0*interp_to_y(path_of_q_60[:,n])
    if np.max(path_of_q_40[:,n]) > 0.0:
        path_of_q_40_plot[:,n] = -1.0*interp_to_y(path_of_q_40[:,n])
    if np.max(path_of_q_20[:,n]) > 0.0:
        path_of_q_20_plot[:,n] = -1.0*interp_to_y(path_of_q_20[:,n])




for n in range(n_q_paths):
    for j in range(len(xs)):
        if np.max(path_of_q[:,n]) > 0.0:
            model_interp_3d_top[j,n] = the_f_3d_top(path_of_q[j,n], xs[j])
            model_interp_3d_bottom[j,n] = the_f_3d_bottom(path_of_q[j,n], xs[j])

        if np.max(path_of_q_80[:,n]) > 0.0:
            model_interp_3d_top_80[j,n] = the_f_3d_top(path_of_q_80[j,n], xs[j]-20.0)
            model_interp_3d_bottom_80[j,n] = the_f_3d_bottom(path_of_q_80[j,n], xs[j]-20.0)

        if np.max(path_of_q_60[:,n]) > 0.0:
            model_interp_3d_top_60[j,n] = the_f_3d_top(path_of_q_60[j,n], xs[j]-40.0)
            model_interp_3d_bottom_60[j,n] = the_f_3d_bottom(path_of_q_60[j,n], xs[j]-40.0)

        if np.max(path_of_q_40[:,n]) > 0.0:
            model_interp_3d_top_40[j,n] = the_f_3d_top(path_of_q_40[j,n], xs[j]-60.0)
            model_interp_3d_bottom_40[j,n] = the_f_3d_bottom(path_of_q_40[j,n], xs[j]-60.0)

        if np.max(path_of_q_20[:,n]) > 0.0:
            model_interp_3d_top_20[j,n] = the_f_3d_top(path_of_q_20[j,n], xs[j]-80.0)
            model_interp_3d_bottom_20[j,n] = the_f_3d_bottom(path_of_q_20[j,n], xs[j]-80.0)



angle1 = 10
angle2 = -115

angle3 = 4
angle4 = -115

#hack: 3d subplots
ax=fig.add_subplot(2, 3, 1, projection='3d')


#for mmm in range(len(param_q_nums)):
for mmm in plot_iter[::-1]:

    # short_length = len(km_linspace)
    # start_point = 4
    # xs = km_linspace[start_point:short_length]
    verts = []
    zs = [-1.0*float(mmm)]

    mmm_fix = mmm
    ys = temp_bottom_linspace[start_point:short_length,0,0,mmm_fix]
    ys2 = temp_top_linspace[start_point:short_length,0,0,mmm_fix]

    ys[0] = 0.0
    ys2[0] = 0.0

    ys3 = np.zeros(len(xs)*2)
    ys3[:len(xs)] = ys
    ys3[len(xs):] = ys2[::-1]

    xs3 = np.zeros(len(xs)*2)
    xs3[:len(xs)] = xs
    xs3[len(xs):] = xs[::-1]

    verts.append(list(zip(xs3, ys3)))

    poly = PolyCollection(verts, facecolors = [plot_col_bold[mmm]], edgecolors=['none'])
    poly.set_alpha(0.7)
    print "zs" , zs
    ax.add_collection3d(poly, zs=zs, zdir='y')

project_x = np.linspace(0.0,100.0,len(xs))
project_x_3d = np.linspace(0.0,100.0,len(xs))
project_y = np.zeros(len(xs))#np.linspace(-4.0,0.0,len(km_linspace))
project_z1 = np.linspace(0.0,100.0,len(xs))
project_z2 = np.linspace(0.0,80.0,len(xs))
project_z3 = np.linspace(0.0,60.0,len(xs))
project_z = np.zeros(len(xs))

# cset = ax.plot(project_x_3d, project_z1, project_y, 'k.-', zdir='y')
# cset = ax.plot(project_x_3d, project_z2, project_y, 'k.-', zdir='y')
# cset = ax.plot(project_x_3d, project_z3, project_y, 'k.-', zdir='y')

cset0, = ax.plot(project_x_3d, model_interp_3d_top[:,0], path_of_q_plot[:,0], 'k-', zdir='y')
cset0, = ax.plot(project_x_3d, model_interp_3d_bottom[:,0], path_of_q_plot[:,0], 'k-', zdir='y')

cset0, = ax.plot(project_x_3d-20.0, model_interp_3d_top_80[:,0], path_of_q_80_plot[:,0], 'k-', zdir='y')
cset0, = ax.plot(project_x_3d-20.0, model_interp_3d_bottom_80[:,0], path_of_q_80_plot[:,0], 'k-', zdir='y')

cset0, = ax.plot(project_x_3d-40.0, model_interp_3d_top_60[:,0], path_of_q_60_plot[:,0], 'k-', zdir='y')
cset0, = ax.plot(project_x_3d-40.0, model_interp_3d_bottom_60[:,0], path_of_q_60_plot[:,0], 'k-', zdir='y')

cset0, = ax.plot(project_x_3d-60.0, model_interp_3d_top_40[:,0], path_of_q_40_plot[:,0], 'k-', zdir='y')
cset0, = ax.plot(project_x_3d-60.0, model_interp_3d_bottom_40[:,0], path_of_q_40_plot[:,0], 'k-', zdir='y')

cset0, = ax.plot(project_x_3d-80.0, model_interp_3d_top_20[:,0], path_of_q_20_plot[:,0], 'k-', zdir='y')
cset0, = ax.plot(project_x_3d-80.0, model_interp_3d_bottom_20[:,0], path_of_q_20_plot[:,0], 'k-', zdir='y')


cset1, = ax.plot(project_x_3d, model_interp_3d_top[:,1], path_of_q_plot[:,1], 'k', zdir='y')
cset1.set_dashes(the_dash1)
cset1, = ax.plot(project_x_3d, model_interp_3d_bottom[:,1], path_of_q_plot[:,1], 'k', zdir='y')
cset1.set_dashes(the_dash1)

cset2, = ax.plot(project_x_3d, model_interp_3d_top[:,2], path_of_q_plot[:,2], 'k', zdir='y')
cset2.set_dashes(the_dash2)
cset2, = ax.plot(project_x_3d, model_interp_3d_bottom[:,2], path_of_q_plot[:,2], 'k', zdir='y')
cset2.set_dashes(the_dash2)


cset3, = ax.plot(project_x_3d, model_interp_3d_top[:,3], path_of_q_plot[:,3], 'k', zdir='y')
cset3.set_dashes(the_dash2)
cset3, = ax.plot(project_x_3d, model_interp_3d_bottom[:,3], path_of_q_plot[:,3], 'k', zdir='y')
cset3.set_dashes(the_dash2)

fs_label = 9
ax.set_xlabel('x distance from inflow region [km]',fontsize=fs_label)
ax.set_xlim3d(-5, 100)
ax.set_ylabel('y lateral fluid velocity',fontsize=fs_label)
ax.set_ylim3d(-4, 0)
plt.yticks([-4.0, -3.0, -2.0, -1.0, 0.0])
ax.set_zlabel('z temperature [C]',fontsize=fs_label)
ax.set_zlim3d(0, 140.0)

ax.view_init(angle1, angle2)









ax=fig.add_subplot(2, 3, 2, projection='3d')

for n in range(n_q_paths):
    cset = ax.plot(project_x, model_interp_3d_top[:,n], project_y, color=plot_col_bold[n], zdir='y')
    cset = ax.plot(project_x, model_interp_3d_bottom[:,n], project_y, color=plot_col_bold[n], zdir='y')
    cset = ax.plot(project_x, project_z, path_of_q_plot[:,n], color=plot_col_bold[n], zdir='y')

    cset = ax.plot(project_x-20.0, model_interp_3d_top_80[:,n], project_y, color=plot_col_bold[n], zdir='y')
    cset = ax.plot(project_x-20.0, model_interp_3d_bottom_80[:,n], project_y, color=plot_col_bold[n], zdir='y')
    cset = ax.plot(project_x-20.0, project_z, path_of_q_80_plot[:,n], color=plot_col_bold[n], zdir='y')

    cset = ax.plot(project_x-40.0, model_interp_3d_top_60[:,n], project_y, color=plot_col_bold[n], zdir='y')
    cset = ax.plot(project_x-40.0, model_interp_3d_bottom_60[:,n], project_y, color=plot_col_bold[n], zdir='y')
    cset = ax.plot(project_x-40.0, project_z, path_of_q_60_plot[:,n], color=plot_col_bold[n], zdir='y')

    cset = ax.plot(project_x-60.0, model_interp_3d_top_40[:,n], project_y, color=plot_col_bold[n], zdir='y')
    cset = ax.plot(project_x-60.0, model_interp_3d_bottom_40[:,n], project_y, color=plot_col_bold[n], zdir='y')
    cset = ax.plot(project_x-60.0, project_z, path_of_q_40_plot[:,n], color=plot_col_bold[n], zdir='y')

    cset = ax.plot(project_x-80.0, model_interp_3d_top_20[:,n], project_y, color=plot_col_bold[n], zdir='y')
    cset = ax.plot(project_x-80.0, model_interp_3d_bottom_20[:,n], project_y, color=plot_col_bold[n], zdir='y')
    cset = ax.plot(project_x-80.0, project_z, path_of_q_20_plot[:,n], color=plot_col_bold[n], zdir='y')




fs_label = 9
ax.set_xlabel('x distance from inflow region [km]',fontsize=fs_label)
ax.set_xlim3d(-5, 100)
ax.set_ylabel('y lateral fluid velocity',fontsize=fs_label)
ax.set_ylim3d(-4, 0)
plt.yticks([-4.0, -3.0, -2.0, -1.0, 0.0])
ax.set_zlabel('z temperature [C]',fontsize=fs_label)
ax.set_zlim3d(0, 140.0)

ax.view_init(angle1, angle2)





ax=fig.add_subplot(2, 3, 3, projection='3d')
# ax.auto_scale_xyz([the_xmin, the_xmax], [the_ymin, the_ymax], [the_zmin, the_zmax*2.0])



plot_iter = range(len(param_q_nums))
#for mmm in range(len(param_q_nums)):
for mmm in plot_iter[::-1]:


    mmm_fix = mmm

    # [0,0,mmm]
    verts = []
    zs = [-1.0*float(mmm)]
    ys = temp_bottom_linspace[start_point:short_length,0,0,mmm_fix]
    ys2 = temp_top_linspace[start_point:short_length,0,0,mmm_fix]

    ys[0] = 0.0
    ys2[0] = 0.0

    ys3 = np.zeros(len(xs)*2)
    ys3[:len(xs)] = ys
    ys3[len(xs):] = ys2[::-1]

    xs3 = np.zeros(len(xs)*2)
    xs3[:len(xs)] = xs
    xs3[len(xs):] = xs[::-1]

    verts.append(list(zip(xs3, ys3)))

    poly = PolyCollection(verts, facecolors = [plot_col_bold[mmm]], edgecolors=['k'])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')




fs_label = 9

angle3 = 4
angle4 = -115

the_xmin = -5
the_xmax = 100

the_ymin = -4
the_ymax = 0.0

the_zmin = 0.0
the_zmax = 140.0


ax.set_xlabel('distance from inflow region [km]',fontsize=fs_label)
ax.set_xlim3d(the_xmin, the_xmax)
ax.set_ylabel('lateral fluid velocity',fontsize=fs_label)
ax.set_ylim3d(the_ymin, the_ymax)
plt.yticks([-4.0, -3.0, -2.0, -1.0, 0.0])
ax.set_zlabel('temperature [C]',fontsize=fs_label)
ax.set_zlim3d(the_zmin, the_zmax)



ax.view_init(angle3, angle4)


plt.savefig(gen_path+'fpmm_3d_'+unique_string+'.png',bbox_inches='tight')
plt.savefig(gen_path+'fpmm_3d_'+unique_string+'.eps',bbox_inches='tight')








fig=plt.figure(figsize=(16.0,10.0))


#hack: 2D bunches
ax=fig.add_subplot(2, 3, 1)

for j in range(n_q_paths):
    cset = ax.plot(project_x, model_interp_3d_top[:,j], color=plot_col_bold[j], label=path_of_q_labels[j])
    cset = ax.plot(project_x-20.0, model_interp_3d_top_80[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-40.0, model_interp_3d_top_60[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-60.0, model_interp_3d_top_40[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-80.0, model_interp_3d_top_20[:,j], color=plot_col_bold[j])

plt.xlim([0.0,110.0])
plt.ylim([0.0,140.0])
plt.title('top temperatures (20, 40, 60, 80, 100), all paths')
plt.xlabel('distance from inflow region [km]')
plt.ylabel('temperature [C]')


ax=fig.add_subplot(2, 3, 2)

for j in range(n_q_paths):
    cset = ax.plot(project_x, model_interp_3d_bottom[:,j], color=plot_col_bold[j], label=path_of_q_labels[j])
    cset = ax.plot(project_x-20.0, model_interp_3d_bottom_80[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-40.0, model_interp_3d_bottom_60[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-60.0, model_interp_3d_bottom_40[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-80.0, model_interp_3d_bottom_20[:,j], color=plot_col_bold[j])

plt.xlim([0.0,110.0])
plt.ylim([0.0,140.0])
plt.title('bottom temperatures (20, 40, 60, 80, 100), all paths')
plt.xlabel('distance from inflow region [km]')
plt.ylabel('temperature [C]')
plt.legend(fontsize=10,bbox_to_anchor=(1.3, 1.0))





selected_paths = [0, 2, 3, 4]

ax=fig.add_subplot(2, 3, 4)

for j in selected_paths:
    cset = ax.plot(project_x, model_interp_3d_top[:,j], color=plot_col_bold[j], label=path_of_q_labels[j])
    cset = ax.plot(project_x-20.0, model_interp_3d_top_80[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-40.0, model_interp_3d_top_60[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-60.0, model_interp_3d_top_40[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-80.0, model_interp_3d_top_20[:,j], color=plot_col_bold[j])

plt.xlim([0.0,110.0])
plt.ylim([0.0,140.0])
plt.title('top temperatures (20, 40, 60, 80, 100), sel. paths')
plt.xlabel('distance from inflow region [km]')
plt.ylabel('temperature [C]')


ax=fig.add_subplot(2, 3, 5)

for j in selected_paths:
    cset = ax.plot(project_x, model_interp_3d_bottom[:,j], color=plot_col_bold[j], label=path_of_q_labels[j])
    cset = ax.plot(project_x-20.0, model_interp_3d_bottom_80[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-40.0, model_interp_3d_bottom_60[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-60.0, model_interp_3d_bottom_40[:,j], color=plot_col_bold[j])
    cset = ax.plot(project_x-80.0, model_interp_3d_bottom_20[:,j], color=plot_col_bold[j])

plt.xlim([0.0,110.0])
plt.ylim([0.0,140.0])
plt.title('bottom temperatures (20, 40, 60, 80, 100), sel. paths')
plt.xlabel('distance from inflow region [km]')
plt.ylabel('temperature [C]')
plt.legend(fontsize=10,bbox_to_anchor=(1.3, 1.0))

plt.savefig(gen_path+'fpmm_bunches_'+unique_string+'.png',bbox_inches='tight')
