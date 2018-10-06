# flow_piece_multi.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
# import streamplot as sp
# import multiplot_data as mpd
import heapq
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rcParams['axes.titlesize'] = 9

plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

plot_col = ['#801515', '#c90d0d', '#d26618', '#dfa524', '#cdeb14', '#7d9d10', '#1ff675', '#139a72', '#359ab5', '#075fd2', '#151fa4', '#3c33a3', '#7f05d4', '#b100de', '#ff8ac2']

#todo: parameters

steps = 10
max_steps = 9
minNum = 57
ison=10000
trace = 0
chem = 1
iso = 0
cell = 5
x_num = 4001
dx_blocks = (x_num-1)/100
y_num = 51
write_txt = 1

# param_age_nums = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
# param_age_strings = ['0.50', '0.75', '1.00', '1.25', '1.50', '1.75', '2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50']

# param_age_nums = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
# param_age_strings = ['0.50', '0.75', '1.00', '1.25', '1.50', '1.75', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50']

# param_age_nums = np.array([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50])
# param_age_strings = ['0.75', '1.00', '1.25', '1.50', '1.75', '2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50']

param_age_nums = np.array([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00])
param_age_strings = ['0.75', '1.00', '1.25', '1.50', '1.75', '2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50', '3.75', '4.00']

# param_age_nums = [1.00, 1.50, 2.00, 2.50, 3.00, 3.50]
# param_age_strings = ['1.00', '1.50', '2.00', '2.50', '3.00', '3.50']

param_sed_nums = np.array([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00])
for i in range(len(param_age_nums)):
    param_sed_nums[i] = param_age_nums[i] * 200.0

print "param_age_nums" , param_age_nums
print " "
print "param_sed_nums" , param_sed_nums

#todo: big arrays
t_col_mean = np.zeros([x_num,len(param_age_nums)])
t_col_bottom = np.zeros([x_num,len(param_age_nums)])
t_col_top = np.zeros([x_num,len(param_age_nums)])

temp_all = np.zeros([y_num,x_num,len(param_age_nums)])

temp_age_dx_top = np.zeros([len(param_age_nums), dx_blocks])
temp_age_dx_bottom = np.zeros([len(param_age_nums), dx_blocks])

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
print_s = "100"
print_h = "200"
print_q = "50.0"
sub_dir = "s= " + print_s + ", h= " + print_h + ", q= " + print_q
print sub_dir
linear_dir_path = "../output/revival/local_fp_output/oc_output/oc_k_13_s_" + print_s + "_h_" + print_h +"_ts/par_q_" + print_q + "/"
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

    # mask = np.loadtxt(ii_path + 'mask.txt')
    maskP = np.loadtxt(ii_path + 'maskP.txt')
    # psi0 = np.loadtxt(ii_path + 'psiMat.txt')
    perm = np.loadtxt(ii_path + 'permeability.txt')

    perm = np.log10(perm)

    temp0 = np.loadtxt(ii_path + 'hMat.txt')
    temp0 = temp0 - 273.0
    # u0 = np.loadtxt(ii_path + 'uMat.txt')
    # v0 = np.loadtxt(ii_path + 'vMat.txt')



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

        # psi = psi0[:,i*len(x):((i)*len(x)+len(x))]
        temp = temp0[:,i*len(x):((i)*len(x)+len(x))]
        # u = u0[:,i*len(x):((i)*len(x)+len(x))]
        # v = v0[:,i*len(x):((i)*len(x)+len(x))]

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
print "what's i vs max_steps? " , i , max_steps
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
    plt.savefig(outpath+'zps_t_mean_all'+'_'+str(i)+'.eps',bbox_inches='tight')




    #todo: t_lat.png
    fig=plt.figure(figsize=(12.0,9.0))
    ax=fig.add_subplot(2, 3, 1, frameon=True)

    #for j in range(cells_flow):

    for ii in range(len(param_age_nums)):
        print x.shape
        print end_temp[bitsy-cells_sed_vec[ii]-cells_flow_vec[ii]-cells_above_vec[ii],:,ii].shape
        plt.plot(x,end_temp[bitsy-cells_sed_vec[ii]-cells_flow_vec[ii]-cells_above_vec[ii],:,ii], label=param_age_strings[ii],c=plot_col[ii],lw=2)
        plt.plot(x,end_temp[bitsy-cells_sed_vec[ii]-cells_above_vec[ii]-1,:,ii],c=plot_col[ii],lw=1, linestyle=':')

    x_sd_temps = np.array([22.443978220690354, 25.50896648184225, 33.32254358359559, 39.22503621559518, 44.528597832059546, 54.624706528797645, 74.32349268195217, 100.7522853289375, 102.99635346420898, 100.74349368100305])
    # x_sd_temps = x_sd_temps - np.min(x_sd_temps)
    x_sd_temps_km = np.array([22.443978220690354, 25.50896648184225, 33.32254358359559, 39.22503621559518, 44.528597832059546, 54.624706528797645, 74.32349268195217, 100.7522853289375, 102.99635346420898])
    # x_sd_temps_km = x_sd_temps_km - np.min(x_sd_temps_km)
    age_sd_temps = np.array([0.86, 0.97, 1.257, 1.434, 1.615, 1.952, 2.621, 3.511, 3.586])
    for j in range(len(x_sd_temps)):
        x_sd_temps[j] = (x_sd_temps[j]-20.0)*1000.0
    for j in range(len(x_sd_temps_km)):
        x_sd_temps_km[j] = x_sd_temps_km[j] - 20.0
    y_sd_temps = np.array([15.256706129177289, 22.631899695289484, 38.471851740846205, 39.824366851491085, 50.20180828213198, 58.10639892102503, 56.69024426794546, 60.72611019531446, 62.36115690094412, 62.91363204955294])
    y_sd_temps_km = np.array([15.256706129177289, 22.631899695289484, 38.471851740846205, 39.824366851491085, 50.20180828213198, 58.10639892102503, 56.69024426794546, 60.72611019531446, 62.36115690094412])
    plt.scatter(x_sd_temps,y_sd_temps,zorder=4,s=60,facecolor='none')
    # 17.716269543933265, 1.5965832459163778
    plt.xlim([0.0,100000.0])
    plt.ylim([0.0,100.0])

    plt.title(sub_dir)

    plt.legend(fontsize=8,bbox_to_anchor=(3.7, 0.7))






    ax=fig.add_subplot(2, 3, 2, frameon=True)

    for ii in range(len(param_age_nums)):
        plt.plot(x,-1.2*(end_temp[bitsy-cells_above_vec[ii]-2,:,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:,ii])/25.0, label=param_age_strings[ii],c=plot_col[ii],lw=2,linestyle='--')
        plt.plot(x,(-1.2*(end_temp[bitsy-cells_above_vec[ii]-2,:,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:,ii])/25.0)/((510.0*(param_age_nums[ii]**(-0.5)))/1000.0), label=param_age_strings[ii],c=plot_col[ii],lw=2,linestyle='-')
    plt.ylim([0.0,0.5])
    plt.title("solid Q_in, dashed Q_out ?")







    # ax=fig.add_subplot(2, 2, 3, frameon=True)
    #
    # for ii in range(len(param_age_nums)):
    #     plt.plot(x,-1.2*(end_temp[bitsy-cells_above_vec[ii]-2,:,ii]-end_temp[bitsy-cells_above_vec[ii]-3,:,ii])/25.0, label=param_age_strings[ii],c=plot_col[ii],lw=2,linestyle='--')
    # plt.ylim([0.0,0.5])
    # plt.title("Q_out [W/m^2]")




    ax=fig.add_subplot(2, 3, 3, frameon=True)

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
    plt.title("Q_out / Q_in" + sub_dir)

    plt.savefig(outpath+'jdf_t_lat_all'+'_'+str(i)+'.png',bbox_inches='tight')
    plt.savefig(outpath+'zps_t_lat_all'+'_'+str(i)+'.eps',bbox_inches='tight')






    #hack: 2d_contour_trial DATA
    for ii in range(len(param_age_nums)):
        for iii in range(dx_blocks):
            temp_age_dx_top[ii,iii] = t_col_top[iii*100.0,ii]
            temp_age_dx_bottom[ii,iii] = t_col_bottom[iii*100.0,ii]
    dx_blocks_array = np.linspace(0.0,100.0,dx_blocks)
    print "dx_blocks_array.shape" , dx_blocks_array.shape
    print "param_age_nums.shape" , param_age_nums.shape
    print "temp_age_dx_top.shape" , temp_age_dx_top.shape



    #hack: interp2d DATA MAKING
    model_interp_top = np.zeros([len(x_sd_temps_km),8])
    model_interp_bottom = np.zeros([len(x_sd_temps_km),8])
    interp_shifts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


    the_f_top = interpolate.interp2d(dx_blocks_array, param_age_nums, temp_age_dx_top, kind="linear")
    the_f_bottom = interpolate.interp2d(dx_blocks_array, param_age_nums, temp_age_dx_bottom, kind="linear")
    for j in range(len(x_sd_temps_km)):
        model_interp_top[j,0] = the_f_top(x_sd_temps_km[j], age_sd_temps[j])
        model_interp_bottom[j,0] = the_f_bottom(x_sd_temps_km[j], age_sd_temps[j])

        model_interp_top[j,1] = the_f_top(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[1])
        model_interp_bottom[j,1] = the_f_bottom(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[1])

        model_interp_top[j,2] = the_f_top(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[2])
        model_interp_bottom[j,2] = the_f_bottom(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[2])

        model_interp_top[j,3] = the_f_top(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[3])
        model_interp_bottom[j,3] = the_f_bottom(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[3])

        model_interp_top[j,4] = the_f_top(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[4])
        model_interp_bottom[j,4] = the_f_bottom(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[4])

        model_interp_top[j,5] = the_f_top(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[5])
        model_interp_bottom[j,5] = the_f_bottom(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[5])

        model_interp_top[j,6] = the_f_top(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[6])
        model_interp_bottom[j,6] = the_f_bottom(x_sd_temps_km[j], age_sd_temps[j]-interp_shifts[6])



    #hack: write interp line to text
    interp_slope = (np.max(x_sd_temps_km) - np.min(x_sd_temps_km)) / (np.max(age_sd_temps) - np.min(age_sd_temps))
    print "interp_slope " , interp_slope
    age_linspace = np.linspace(0.0, 4.0, 20)
    km_linspace = (interp_slope*(age_linspace-np.min(age_sd_temps)))
    temp_top_linspace = np.zeros(len(age_linspace))
    temp_bottom_linspace = np.zeros(len(age_linspace))
    print "age_linspace" , age_linspace
    print "km_linspace" , km_linspace

    for j in range(len(age_linspace)):
        temp_top_linspace[j] = the_f_top(km_linspace[j], age_linspace[j])
        temp_bottom_linspace[j] = the_f_bottom(km_linspace[j], age_linspace[j])

    #hack: MAYBE WRITE TO FILE?
    if write_txt == 1:
        np.savetxt(outpath+'z_age_linspace.txt', age_linspace)
        np.savetxt(outpath+'z_km_linspace.txt', km_linspace)
        np.savetxt(outpath+'z_temp_top_linspace.txt', temp_top_linspace)
        np.savetxt(outpath+'z_temp_bottom_linspace.txt', temp_bottom_linspace)



    #todo: 2d_contour_trial
    fig=plt.figure(figsize=(12.0,9.0))
    # plt.subplots_adjust(hspace=0.1)

    ax=fig.add_subplot(2, 3, 1, frameon=True)
    plt.title (sub_dir, fontsize=9)

    plt.plot(x_sd_temps_km,y_sd_temps_km,'mo',label="data")

    plt.plot(x_sd_temps_km,model_interp_top[:,0],'bo-',label="model_interp_top")
    plt.plot(x_sd_temps_km,model_interp_bottom[:,0],'ro-',label="model_interp_bottom")

    plt.plot(x_sd_temps_km,model_interp_top[:,1],'b.-',alpha=0.9)
    plt.plot(x_sd_temps_km,model_interp_bottom[:,1],'r.-',alpha=0.9)

    plt.plot(x_sd_temps_km,model_interp_top[:,2],'b.-',alpha=0.8)
    plt.plot(x_sd_temps_km,model_interp_bottom[:,2],'r.-',alpha=0.8)

    plt.plot(x_sd_temps_km,model_interp_top[:,3],'b.-',alpha=0.7)
    plt.plot(x_sd_temps_km,model_interp_bottom[:,3],'r.-',alpha=0.7)

    plt.plot(x_sd_temps_km,model_interp_top[:,4],'b.-',alpha=0.6)
    plt.plot(x_sd_temps_km,model_interp_bottom[:,4],'r.-',alpha=0.6)

    plt.plot(x_sd_temps_km,model_interp_top[:,5],'b.-',alpha=0.5)
    plt.plot(x_sd_temps_km,model_interp_bottom[:,5],'r.-',alpha=0.5)

    plt.plot(x_sd_temps_km,model_interp_top[:,6],'b.-',alpha=0.4)
    plt.plot(x_sd_temps_km,model_interp_bottom[:,6],'r.-',alpha=0.4)

    plt.plot(km_linspace,temp_top_linspace,'y^-')
    plt.plot(km_linspace,temp_bottom_linspace,'yv-')


    plt.legend(loc='best',fontsize=8)

    # plt.xlim([0.0,100.0])
    # plt.ylim([0.0,120.0])
    plt.xlabel('distance from inflow [km]',fontsize=9)
    plt.ylabel('temp [C]',fontsize=9)






    v_min_all = np.min(temp_age_dx_top)
    if np.min(temp_age_dx_bottom) < v_min_all:
        v_min_all = np.min(temp_age_dx_bottom)

    v_max_all = np.min(temp_age_dx_top)
    if np.max(temp_age_dx_bottom) > v_max_all:
        v_max_all = np.max(temp_age_dx_bottom)

    ax=fig.add_subplot(2, 3, 2, frameon=True)
    the_pcol = plt.pcolor(dx_blocks_array, param_age_nums, temp_age_dx_top, vmin=v_min_all, vmax=v_max_all,cmap=cm.rainbow)
    plt.title (sub_dir + " temp_age_dx_top", fontsize=9)

    plt.plot(x_sd_temps_km,age_sd_temps,color='k',lw=1)
    plt.scatter(x_sd_temps_km,age_sd_temps,s=40,edgecolor='k',facecolor='none',lw=1.0,linestyle='-')

    plt.scatter(x_sd_temps_km,age_sd_temps-interp_shifts[1],s=10,edgecolor='k',facecolor='k')
    plt.scatter(x_sd_temps_km,age_sd_temps-interp_shifts[2],s=10,edgecolor='k',facecolor='k')
    plt.scatter(x_sd_temps_km,age_sd_temps-interp_shifts[3],s=10,edgecolor='k',facecolor='k')
    plt.scatter(x_sd_temps_km,age_sd_temps-interp_shifts[4],s=10,edgecolor='k',facecolor='k')
    plt.scatter(x_sd_temps_km,age_sd_temps-interp_shifts[5],s=10,edgecolor='k',facecolor='k')
    plt.scatter(x_sd_temps_km,age_sd_temps-interp_shifts[6],s=10,edgecolor='k',facecolor='k')

    plt.plot(x_sd_temps_km,age_sd_temps-interp_shifts[1],color='k',lw=1)
    plt.plot(x_sd_temps_km,age_sd_temps-interp_shifts[2],color='k',lw=1)
    plt.plot(x_sd_temps_km,age_sd_temps-interp_shifts[3],color='k',lw=1)
    plt.plot(x_sd_temps_km,age_sd_temps-interp_shifts[4],color='k',lw=1)
    plt.plot(x_sd_temps_km,age_sd_temps-interp_shifts[5],color='k',lw=1)
    plt.plot(x_sd_temps_km,age_sd_temps-interp_shifts[6],color='k',lw=1)

    plt.scatter(km_linspace,age_linspace,s=30,edgecolor='k',facecolor='yellow')

    # plt.xlim([np.min(dx_blocks_array),np.max(dx_blocks_array)])
    # plt.ylim([np.min(param_age_nums),np.max(param_age_nums)+0.1])
    plt.colorbar(the_pcol, orientation='horizontal')
    plt.ylabel('age of crust [Myr]',fontsize=9)
    plt.xlabel('distance from inflow [km]',fontsize=9)




    ax=fig.add_subplot(2, 3, 3, frameon=True)
    the_pcol = plt.pcolor(dx_blocks_array, param_age_nums, temp_age_dx_bottom, vmin=v_min_all, vmax=v_max_all,cmap=cm.rainbow)
    plt.title (sub_dir + " temp_age_dx_bottom", fontsize=9)
    plt.scatter(x_sd_temps_km,age_sd_temps,s=40,edgecolor='k',facecolor='none')

    plt.scatter(km_linspace,age_linspace,s=30,edgecolor='k',facecolor='yellow')

    # plt.xlim([np.min(dx_blocks_array),np.max(dx_blocks_array)])
    # plt.ylim([np.min(param_age_nums),np.max(param_age_nums)+0.1])
    plt.colorbar(the_pcol, orientation='horizontal')
    plt.ylabel('age of crust [Myr]',fontsize=9)
    plt.xlabel('distance from inflow [km]',fontsize=9)













    plt.savefig(outpath+'jdf_cont_trial'+'_'+str(i)+'.png',bbox_inches='tight')
    plt.savefig(outpath+'zps_cont_trial'+'_'+str(i)+'.eps',bbox_inches='tight')






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
fig=plt.figure(figsize=(12.0,9.0))

ax=fig.add_subplot(2, 3, 1, frameon=True)

for ii in range(len(param_age_nums)):
    if (ii+1)%2 == 0:
        plt.plot(geotherm_left[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],label=param_age_strings[ii])
        plt.plot(geotherm_right[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],linestyle='--')
        #plt.plot(geotherm_mean[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],linestyle='-')
        #ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.xlabel('temperature [C]',fontsize=9)
plt.ylabel('depth below seafloor [m]',fontsize=9)
plt.title('sub_dir = ' + sub_dir)
plt.xlim([0.0,160.0])
plt.legend(fontsize=8,bbox_to_anchor=(1.2, 0.7))


# ax=fig.add_subplot(2, 3, 2, frameon=True)
#
# for ii in range(len(param_age_nums)):
#     # if (ii+1)%2 == 0:
#     plt.plot(geotherm_left[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],label=param_age_strings[ii])
#     plt.plot(geotherm_right[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],linestyle='-')
#     #plt.plot(geotherm_mean[geotherm_base:,ii],y[geotherm_base:],c=plot_col[ii],linestyle='-')
#
# plt.xlim([0.0,100.0])
# plt.ylim([-600.0,-400.0])
# plt.xlabel('temperature [C]',fontsize=9)
# plt.ylabel('depth below seafloor [m]',fontsize=9)
# plt.title('sub_dir = ' + sub_dir)
# plt.legend(fontsize=8,bbox_to_anchor=(1.2, 0.7))


# ax=fig.add_subplot(2, 3, 3, frameon=True)
# #cmap1 = LinearSegmentedColormap.from_list("my_colormap", ((0.64, 0.1, 0.53), (0.78, 0.61, 0.02)), N=15, gamma=1.0)
# cmap1 = cm.jet
# diff_colors = [ cmap1(xc) for xc in np.linspace(0.0, 1.0, 30) ]
#
# ii = 3.0
# for j in range(len(x[::100])):
#     plt.plot(end_temp[geotherm_base:,j*(99),ii],y[geotherm_base:],color=diff_colors[j])
# plt.xlim([0.0,100.0])
# plt.xlabel('temperature [C]',fontsize=9)
# plt.ylabel('depth below seafloor [m]',fontsize=9)


plt.savefig(outpath+'jdf_geotherm'+'_'+print_q+'.png',bbox_inches='tight')
plt.savefig(outpath+'zps_geotherm'+'_'+print_q+'.eps',bbox_inches='tight')







#todo: scatter means
fig=plt.figure(figsize=(10.0,10.0))


ax=fig.add_subplot(2, 2, 1, frameon=True)
plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00],t_col_mean_list,'ro-',label='t_col_mean_list')
plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00],t_col_bottom_list,'go-',label='t_col_bottom_list')
plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00],t_col_top_list,'bo-',label='t_col_top_list')

plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00],t_col_east_mean_list,'r^-',markersize=0)
plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00],t_col_east_bottom_list,'g^-',markersize=0)
plt.plot([0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00],t_col_east_top_list,'b^-',markersize=0)

# plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_mean_list,'ro-',label='t_col_mean_list')
# plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_bottom_list,'go-',label='t_col_bottom_list')
# plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_top_list,'bo-',label='t_col_top_list')
#
# plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_east_mean_list,'r^-',markersize=0)
# plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_east_bottom_list,'g^-',markersize=0)
# plt.plot([1.00, 1.50, 2.00, 2.50, 3.00, 3.50],t_col_east_top_list,'b^-',markersize=0)

plt.legend(fontsize=8, loc='best')
plt.xlabel('age of crust [Myr]')
plt.ylabel('mean lateral fluid temperature along 100km of crust')

plt.ylim([0.0,160.0])
plt.title('sub_dir = ' + sub_dir)

plt.savefig(outpath+'jdf_t_scatter_'+print_q+'.png',bbox_inches='tight')
plt.savefig(outpath+'jdf_t_scatter_'+print_q+'.eps',bbox_inches='tight')
