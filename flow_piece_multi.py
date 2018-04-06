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
y_num = 51

# param_age_nums = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
# param_age_strings = ['0.50', '0.75', '1.00', '1.25', '1.50', '1.75', '2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50']

# param_age_nums = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
# param_age_strings = ['0.50', '0.75', '1.00', '1.25', '1.50', '1.75', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50']

# param_age_nums = [0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
# param_age_strings = ['0.75', '1.00', '1.25', '1.50', '1.75', '2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50']

param_age_nums = [1.00, 1.50, 2.00, 2.50, 3.00, 3.50]
param_age_strings = ['1.00', '1.50', '2.00', '2.50', '3.00', '3.50']

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

temp_all = np.zeros([y_num,x_num,len(param_age_nums)])


#todo: geotherm data
end_temp = np.zeros([y_num,x_num,len(param_age_nums)])
geotherm_left = np.zeros([y_num,len(param_age_nums)])
geotherm_right = np.zeros([y_num,len(param_age_nums)])
geotherm_mean = np.zeros([y_num,len(param_age_nums)])

index_left = 100
index_right = -100

first_flow_vec = np.zeros(len(param_age_nums))
cells_flow_vec = np.zeros(len(param_age_nums))
cells_sed_vec = np.zeros(len(param_age_nums))
cells_above_vec = np.zeros(len(param_age_nums))

#hack: path
sub_dir = "ao_q_5.0"
linear_dir_path = "../output/revival/local_fp_output/"+sub_dir+"/"
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



    first_flow = np.argmax(perm[:,-1]>-16.0)
    cells_flow = np.argmax(perm[first_flow:,-1]<-16.0)
    print "cells flow count: " , cells_flow
    cells_sed = np.argmax(perm[first_flow+cells_flow:,-1]>-16.0)
    print "cells sed count: " , cells_sed
    cells_above = len(y) - cells_sed - cells_flow - first_flow
    print "cells above count: " , cells_above

    first_flow_vec[ii] = first_flow
    cells_flow_vec[ii] = cells_flow
    cells_sed_vec[ii] = cells_sed
    cells_above_vec[ii] = cells_above


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

        #todo: fill geotherm arrays
        geotherm_base = 25
        if i == max_steps:
            geotherm_left[geotherm_base:,ii] = temp[geotherm_base:,index_left]
            geotherm_right[geotherm_base:,ii] = temp[geotherm_base:,index_right]
            geotherm_mean[geotherm_base:,ii] = temp[geotherm_base:,bitsx/2]

            end_temp[:,:,ii] = temp




#todo: jdf_t_mean.png

if i == max_steps:
    fig=plt.figure(figsize=(10.0,10.0))
    ax=fig.add_subplot(2, 2, 1, frameon=True)

    for ii in range(len(param_age_nums)):
        plt.plot(x/1000.0,t_col_mean[:,ii], label=param_age_strings[ii], c=plot_col[ii],lw=2)
        plt.plot(x/1000.0,t_col_bottom[:,ii], linestyle='--', c=plot_col[ii],lw=2)
        plt.plot(x/1000.0,t_col_top[:,ii], linestyle=':', c=plot_col[ii],lw=2)

    plt.xlabel('x distance along transect [km]')
    plt.ylabel('temperature')
    plt.ylim([0.0,80.0])
    plt.title('sub_dir = ' + sub_dir)

    plt.legend(fontsize=8,bbox_to_anchor=(1.2, 0.7))
    plt.savefig(outpath+'jdf_t_mean_all'+'_'+str(i)+'.png',bbox_inches='tight')


    #todo: t_lat.png
    fig=plt.figure(figsize=(10.0,10.0))
    ax=fig.add_subplot(2, 2, 1, frameon=True)

    #for j in range(cells_flow):

    for ii in range(len(param_age_nums)):
        print x.shape
        print end_temp[bitsy-cells_sed_vec[ii]-cells_flow_vec[ii]-cells_above_vec[ii],:,ii].shape
        plt.plot(x,end_temp[bitsy-cells_sed_vec[ii]-cells_flow_vec[ii]-cells_above_vec[ii],:,ii], label=param_age_strings[ii],c=plot_col[ii],lw=2)
        plt.plot(x,end_temp[bitsy-cells_sed_vec[ii]-cells_above_vec[ii]-1,:,ii],c=plot_col[ii],lw=1, linestyle=':')
    plt.title(sub_dir)

    plt.legend(fontsize=8,loc='best')







    ax=fig.add_subplot(2, 2, 2, frameon=True)


    for ii in range(len(param_age_nums)):
        plt.plot(x,-1.2*(end_temp[bitsy-cells_above_vec[ii]-2,:,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:,ii])/25.0, label=param_age_strings[ii],c=plot_col[ii],lw=2,linestyle='--')
        plt.plot(x,(-1.2*(end_temp[bitsy-cells_above_vec[ii]-2,:,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:,ii])/25.0)/((510.0*(param_age_nums[ii]**(-0.5)))/1000.0), label=param_age_strings[ii],c=plot_col[ii],lw=2,linestyle='-')
    plt.ylim([0.0,0.5])
    plt.title("Q_out [W/m^2]")









    ax=fig.add_subplot(2, 2, 3, frameon=True)


    for ii in range(len(param_age_nums)):
        plt.plot(x,-1.2*(end_temp[bitsy-cells_above_vec[ii]-2,:,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:,ii])/25.0, label=param_age_strings[ii],c=plot_col[ii],lw=2,linestyle='--')
    plt.ylim([0.0,0.5])
    plt.title("Q_out [W/m^2]")





    ax=fig.add_subplot(2, 2, 4, frameon=True)


    for ii in range(len(param_age_nums)):
        print " "
        q_lith_temp = (510.0*(param_age_nums[ii]**(-0.5)))/1000.0
        print " Q_lith" , q_lith_temp
        plt.plot(x,(-1.2*(end_temp[bitsy-cells_above_vec[ii]-2,:,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:,ii])/25.0)/((510.0*(param_age_nums[ii]**(-0.5)))/1000.0), label=param_age_strings[ii],c=plot_col[ii],lw=2,linestyle='-')
        plt.plot(x,(-1.2*(end_temp[bitsy-cells_above_vec[ii]-1,:,ii]-end_temp[bitsy-cells_above_vec[ii]-2,:,ii])/25.0)/((510.0*(param_age_nums[ii]**(-0.5)))/1000.0), label=param_age_strings[ii],c=plot_col[ii],lw=2,linestyle='--')
        plt.plot(x,(-1.2*(end_temp[bitsy-cells_above_vec[ii]-0,:,ii]-end_temp[bitsy-cells_above_vec[ii]-1,:,ii])/25.0)/((510.0*(param_age_nums[ii]**(-0.5)))/1000.0), label=param_age_strings[ii],c=plot_col[ii],lw=2,linestyle=':')
        print "sub_dir: " , sub_dir

        print "age = " , param_age_nums[ii]
        # print "mean 1: " , np.mean((-1.2*(end_temp[bitsy-cells_above_vec[ii]-2,:,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:,ii])/25.0)/((510.0*(param_age_nums[ii]**(-0.5)))/1000.0))
        # print "mean 2: " , np.mean((-1.2*(end_temp[bitsy-cells_above_vec[ii]-2,:-50,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:-50,ii])/25.0)/((510.0*(param_age_nums[ii]**(-0.5)))/1000.0))
        # print "np.max(x): " , np.max(x)
        # print "sum 1: " , (np.sum((-1.2*50.0*(end_temp[bitsy-cells_above_vec[ii]-2,:-50,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:-50,ii])/25.0)/((510.0*(param_age_nums[ii]**(-0.5)))/1000.0)))/np.max(x)
        print "np.max(x): " , np.max(x)
        print "sum 2: " , (np.sum(-1.2*(end_temp[bitsy-cells_above_vec[ii]-2,:400,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:400,ii])/25.0)/q_lith_temp)/400.0
    plt.ylim([0.0,1.0])
    plt.title("Q_out / Q_in?")

    plt.savefig(outpath+'jdf_t_lat_all'+'_'+str(i)+'.png',bbox_inches='tight')



t_col_mean_list = np.zeros(len(param_age_nums))
t_col_bottom_list = np.zeros(len(param_age_nums))
t_col_top_list = np.zeros(len(param_age_nums))

t_col_east_mean_list = np.zeros(len(param_age_nums))
t_col_east_bottom_list = np.zeros(len(param_age_nums))
t_col_east_top_list = np.zeros(len(param_age_nums))

for ii in range(len(param_age_nums)):
    t_col_mean_list[ii] = np.mean(t_col_mean[:,ii])
    t_col_bottom_list[ii] = np.mean(t_col_bottom[:,ii])
    t_col_top_list[ii] = np.mean(t_col_top[:,ii])

    t_col_east_mean_list[ii] = np.max(t_col_mean[:-100,ii])
    t_col_east_bottom_list[ii] = np.max(t_col_bottom[:-100,ii])
    t_col_east_top_list[ii] = np.max(t_col_top[:-100,ii])







#todo: geothermal gradients figure
fig=plt.figure(figsize=(12.0,12.0))

ax=fig.add_subplot(2, 2, 1, frameon=True)

for ii in range(len(param_age_nums)):
    if (ii+1)%2 == 0:
        plt.plot(geotherm_left[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],label=param_age_strings[ii])
        plt.plot(geotherm_right[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],linestyle='--')
        #plt.plot(geotherm_mean[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],linestyle='-')
        #ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.xlabel('temperature [C]')
plt.ylabel('depth below seafloor [m]')
plt.title('sub_dir = ' + sub_dir)
#plt.legend(fontsize=8,bbox_to_anchor=(1.2, 0.7))


ax=fig.add_subplot(2, 2, 2, frameon=True)

for ii in range(len(param_age_nums)):
    # if (ii+1)%2 == 0:
    plt.plot(geotherm_left[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],label=param_age_strings[ii])
    plt.plot(geotherm_right[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],linestyle='-')
    #plt.plot(geotherm_mean[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],linestyle='-')

plt.xlim([0.0,100.0])
plt.ylim([-600.0,-400.0])
plt.xlabel('temperature [C]')
plt.ylabel('depth below seafloor [m]')
plt.title('sub_dir = ' + sub_dir)
plt.legend(fontsize=8,bbox_to_anchor=(1.2, 0.7))


ax=fig.add_subplot(2, 2, 3, frameon=True)
#cmap1 = LinearSegmentedColormap.from_list("my_colormap", ((0.64, 0.1, 0.53), (0.78, 0.61, 0.02)), N=15, gamma=1.0)
cmap1 = cm.jet
diff_colors = [ cmap1(xc) for xc in np.linspace(0.0, 1.0, 40) ]

ii = 3.0
for j in range(len(x[::100])):
    plt.plot(end_temp[geotherm_base:,j*(99),ii],y[geotherm_base:],color=diff_colors[j])



plt.savefig(outpath+'jdf_geotherm'+'_'+str(i)+'.png',bbox_inches='tight')








#todo: scatter means
fig=plt.figure(figsize=(10.0,10.0))


ax=fig.add_subplot(2, 2, 1, frameon=True)
# plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50],t_col_mean_list,'ro-',label='t_col_mean_list')
# plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50],t_col_bottom_list,'go-',label='t_col_bottom_list')
# plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50],t_col_top_list,'bo-',label='t_col_top_list')
#
# plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50],t_col_east_mean_list,'r^-',markersize=0)
# plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50],t_col_east_bottom_list,'g^-',markersize=0)
# plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50],t_col_east_top_list,'b^-',markersize=0)

plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_mean_list,'ro-',label='t_col_mean_list')
plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_bottom_list,'go-',label='t_col_bottom_list')
plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_top_list,'bo-',label='t_col_top_list')

plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_east_mean_list,'r^-',markersize=0)
plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_east_bottom_list,'g^-',markersize=0)
plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_east_top_list,'b^-',markersize=0)

plt.legend(fontsize=8, loc='best')
plt.xlabel('age of crust [Myr]')
plt.ylabel('mean lateral fluid temperature along 100km of crust')

plt.ylim([0.0,80.0])
plt.title('sub_dir = ' + sub_dir)

plt.savefig(outpath+'jdf_t_scatter.png',bbox_inches='tight')