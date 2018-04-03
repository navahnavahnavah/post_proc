# flow_piece_multi.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import streamplot as sp
import multiplot_data as mpd
import heapq
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams['axes.titlesize'] = 12

plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

plot_col = ['#801515', '#c90d0d', '#d26618', '#dfa524', '#cdeb14', '#7d9d10', '#139a55', '#359ab5', '#075fd2', '#3c33a3', '#7f05d4', '#b100de', '#ff8ac2']

#todo: parameters

steps = 10
max_steps = 9
minNum = 57
ison=10000
trace = 0
chem = 1
iso = 0
cell = 5
x_num = 2001

# param_age_nums = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
# param_age_strings = ['0.50', '0.75', '1.00', '1.25', '1.50', '1.75', '2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50']

# param_age_nums = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
# param_age_strings = ['0.50', '0.75', '1.00', '1.25', '1.50', '1.75', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50']

param_age_nums = [0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
param_age_strings = ['0.75', '1.00', '1.25', '1.50', '1.75', '2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50']

param_sed_nums = param_age_nums[:]
for i in range(len(param_age_nums)):
    param_sed_nums[i] = param_age_nums[i] * 100.0

print "param_age_nums" , param_age_nums
print " "
print "param_sed_nums" , param_sed_nums

#todo: big arrays
t_col_mean = np.zeros([x_num,len(param_age_nums)])
t_col_bottom = np.zeros([x_num,len(param_age_nums)])
t_col_top = np.zeros([x_num,len(param_age_nums)])


linear_dir_path = "../output/revival/local_fp_output/ao_q_1.0/"
outpath = linear_dir_path

for ii in range(len(param_age_nums)):

    ii_path = linear_dir_path + 'ao_' + param_age_strings[ii] + '/'
    print 'ii_path', ii_path
    print ' '

    #todo: sample import
    if ii == 0:
        x0 = np.loadtxt(ii_path + 'x.txt',delimiter='\n')
        y0 = np.loadtxt(ii_path + 'y.txt',delimiter='\n')

        x=x0
        y=y0

        bitsx = len(x)
        bitsy = len(y)
        print "bitsx" , bitsx

        dx = float(np.max(x))/float(bitsx)
        dy = np.abs(float(np.max(np.abs(y)))/float(bitsy))

        xg, yg = np.meshgrid(x[:],y[:])

    #todo: regular import

    mask = np.loadtxt(ii_path + 'mask.txt')
    maskP = np.loadtxt(ii_path + 'maskP.txt')
    psi0 = np.loadtxt(ii_path + 'psiMat.txt')
    perm = np.loadtxt(ii_path + 'permeability.txt')

    perm = np.log10(perm)

    temp0 = np.loadtxt(ii_path + 'hMat.txt')
    temp0 = temp0 - 273.0
    u0 = np.loadtxt(ii_path + 'uMat.txt')
    v0 = np.loadtxt(ii_path + 'vMat.txt')


    #hack: start time loop
    for i in range(0,steps,1):

        psi = psi0[:,i*len(x):((i)*len(x)+len(x))]
        temp = temp0[:,i*len(x):((i)*len(x)+len(x))]
        u = u0[:,i*len(x):((i)*len(x)+len(x))]
        v = v0[:,i*len(x):((i)*len(x)+len(x))]

        # if i == steps-1:
        if i > -1:


            for j in range(len(x)):
            #for j in range(20):
                temp_temp = temp[:,j]
                temp_perm = perm[:,j]
                temp_maskP = maskP[:,j]
                t_col_mean[j,ii] = np.mean(temp_temp[(temp_perm>=-13.0) & (temp_maskP>0.0)])
                t_col_bottom[j,ii] = np.max(temp_temp[(temp_perm>=-13.0) & (temp_maskP>0.0)])
                t_col_top[j,ii] = np.min(temp_temp[(temp_perm>=-13.0) & (temp_maskP>0.0)])


#todo: jdf_t_mean.png

if i == max_steps:
    fig=plt.figure(figsize=(12.0,12.0))
    ax=fig.add_subplot(2, 2, 1, frameon=True)

    for ii in range(len(param_age_nums)):
        plt.plot(x/1000.0,t_col_mean[:,ii], label=param_age_strings[ii], c=plot_col[ii],lw=2)
        plt.plot(x/1000.0,t_col_bottom[:,ii], linestyle='--', c=plot_col[ii],lw=2)
        plt.plot(x/1000.0,t_col_top[:,ii], linestyle=':', c=plot_col[ii],lw=2)

    plt.xlabel('x distance along transect [km]')
    plt.ylabel('temperature')

    plt.legend(fontsize=8,bbox_to_anchor=(1.2, 0.7))
    plt.savefig(outpath+'jdf_t_mean_all'+'_'+str(i)+'.png',bbox_inches='tight')

t_col_mean_list = np.zeros(len(param_age_nums))
t_col_bottom_list = np.zeros(len(param_age_nums))
t_col_top_list = np.zeros(len(param_age_nums))

for ii in range(len(param_age_nums)):
    t_col_mean_list[ii] = np.mean(t_col_mean[:,ii])
    t_col_bottom_list[ii] = np.mean(t_col_bottom[:,ii])
    t_col_top_list[ii] = np.mean(t_col_top[:,ii])


#todo: scatter means
fig=plt.figure(figsize=(12.0,12.0))


ax=fig.add_subplot(2, 2, 1, frameon=True)
plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50],t_col_mean_list,'ro-',label='t_col_mean_list')
plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50],t_col_bottom_list,'go-',label='t_col_bottom_list')
plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50],t_col_top_list,'bo-',label='t_col_top_list')

plt.legend(fontsize=8, loc='best')
plt.xlabel('age of crust [Myr]')
plt.ylabel('mean lateral fluid temperature along 100km of crust')

plt.savefig(outpath+'jdf_t_scatter.png',bbox_inches='tight')
