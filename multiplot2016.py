# revived_JDF.py

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import math
import multiplot_data as mpd
from numpy.ma import masked_array
from mpl_toolkits.axes_grid1 import AxesGrid
import copy
import matplotlib as mpl # in python
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
# plt.rc('title', fontsize=8)
# plt.rc('ylabel', labelsize=8)
# plt.rc('xlabel', labelsize=8)






# plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"
plt.rcParams['axes.color_cycle'] = "#942200, #f50002, #fc7c00, #ffd100, #4fff00, #00e8c3, #4500ff, #7280c4, #9d27a8, #d62463"



# hot_cm = mpl.colors.ListedColormap(c_mat/255.0)
# cool_cm = mpl.colors.ListedColormap(c_mat_cool/255.0)
# dawn_cm = mpl.colors.ListedColormap(c_dawn/255.0)

color_string = ["#942200", "#f50002", "#fc7c00", "#ffd100", "#4fff00", "#00e8c3", "#4500ff", "#7280c4", "#9d27a8", "#d62463"]

sed = np.abs(mpd.interp_s)
sed1 = np.abs(mpd.interp_b)
sed2 = np.abs(mpd.interp_s - mpd.interp_b)

sed = np.append(sed,sed[-1])
sed1 = np.append(sed1,sed1[-1])
sed2 = np.append(sed2,sed2[-1])

##############
# INITIALIZE #
##############

cell = 10
#steps = 400
steps = 10
minNum = 58

param_o_rhs = 600.0
param_o_rhs_string = '600'

tsw = 3040
# path = "output/revival/may16/31_h100_wrl500/"
# path_ex = "output/revival/may16/jdf_h100_orl200/w400wrhs450/"

# path = "output/revival/june16/crunch_klhs_g-1/"
# path_ex = "output/revival/june16/crunch_klhs_g-1/w200ch600/"

# path = "output/revival/summer16/10_01_h200_closed_porf/"
# path_ex = "output/revival/summer16/10_01_h200_closed_porf/por1e-3b-0.8/"

path = "output/revival/summer16/10_25_halfstep_h300_split/"
path_ex = "output/revival/summer16/10_25_halfstep_h300_split/por1e-3b-1.3/"
# path = "output/revival/summer16/06_group_trial_11/"
# path_ex = "output/revival/summer16/06_group_trial_11/por1e-3b-1.0/"
param_h = 300.0
cell_thing = 12.0


# o300orhs500w1200wrhs1200h075

# load output
x = np.loadtxt(path_ex + 'x.txt',delimiter='\n')
y = np.loadtxt(path_ex + 'y.txt',delimiter='\n')
xn = len(x)
asp = np.abs(np.max(x)/np.min(y))/4.0
xg, yg = np.meshgrid(x[:],y[:])

bitsx = len(x)
bitsy = len(y)

ripTrans = np.transpose(mpd.ripSort)
hf_interp = np.zeros(200)
hf_interp = np.interp(x,ripTrans[0,:],ripTrans[1,:])


xg, yg = np.meshgrid(x[:],y[:])


param_w = np.array([200.0, 400.0 ,600.0, 800.0, 1000.0, 1200.0, 1400.0])
param_w_string = ['200', '400', '600', '800', '1000', '1200', '1400']

# param_f_dx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
# param_f_dx_string = ['0.005', '0.01', '0.02', '0.04', '0.06', '0.08', '0.10']
# param_f_dx = np.array([1.0, 2.0, 3.0])
# param_f_dx_string = ['0.005', '0.01', '0.02']

#param_f_dx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
# param_f_dx_string = ['1e-3', '5e-3', '1e-2', '2e-2', '4e-2', '6e-2', '8e-2', '1e-1']
# param_f_dx_actual = [1e-3, 5e-3, 1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 1e-1]
# param_f_dx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
# param_f_dx_string = ['1e-3', '5e-3', '1e-2', '2e-2', '3e-2', '4e-2', '5e-2', '6e-2', '7e-2', '8e-2', '9e-2', '1e-1']
# param_f_dx_actual = [1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]

# param_f_dx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
# param_f_dx_string = ['-2.4', '-2.2', '-2.0', '-1.8', '-1.6', '-1.4', '-1.2', '-1.0', '-0.8', '-0.6']
# param_f_dx_actual = [-2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6]

# param_f_dx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
# param_f_dx_string = ['-2.2', '-2.0', '-1.8', '-1.6', '-1.4', '-1.2', '-1.0', '-0.8']
# param_f_dx_actual = [-2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8]
#
# param_f_por = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
# param_f_por_string = ['1e-4', '2e-4', '3e-4', '4e-4', '5e-4', '6e-4', '7e-4', '8e-4', '9e-4', '1e-3']
# param_f_por_actual = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3])
# param_f_por_string2 = ['0.01%', '0.02%', '0.03%', '0.04%', '0.05%', '0.06%', '0.07%', '0.08%', '0.09%', '0.10%']


# # ULTIMATE HALF-STEPS
# param_f_dx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
# param_f_dx_string = ['-2.2', '-2.1', '-2.0', '-1.9', '-1.8', '-1.7', '-1.6', '-1.5', '-1.4', '-1.3', '-1.2', '-1.1', '-1.0', '-0.9', '-0.8']
# param_f_dx_actual = np.array([-2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8])
# param_f_dx_actual_squared = np.zeros(param_f_dx_actual.shape)
# param_f_dx_actual_cubed = np.zeros(param_f_dx_actual.shape)
# for i in range(len(param_f_dx_actual)):
#     param_f_dx_actual_squared[i] = (10.0**param_f_dx_actual[i])*(10.0**param_f_dx_actual[i])
#     param_f_dx_actual_cubed[i] = (10.0**param_f_dx_actual[i])*(10.0**param_f_dx_actual[i])*(10.0**param_f_dx_actual[i])
# print "param_f_dx_actual_squared"
# print param_f_dx_actual_squared
# #
# param_f_por = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
# param_f_por_string = ['1e-4', '2e-4', '3e-4', '4e-4', '5e-4', '6e-4', '7e-4', '8e-4', '9e-4', '1e-3']
# param_f_por_actual = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3])
# param_f_por_string2 = ['0.01%', '0.02%', '0.03%', '0.04%', '0.05%', '0.06%', '0.07%', '0.08%', '0.09%', '0.10%']


# -2.2 - -1.0 HALF-STEPS
param_f_dx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
param_f_dx_string = ['-2.2', '-2.1', '-2.0', '-1.9', '-1.8', '-1.7', '-1.6', '-1.5', '-1.4', '-1.3', '-1.2', '-1.1', '-1.0']
param_f_dx_actual = np.array([-2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0])
param_f_dx_actual_squared = np.zeros(param_f_dx_actual.shape)
param_f_dx_actual_cubed = np.zeros(param_f_dx_actual.shape)
for i in range(len(param_f_dx_actual)):
    param_f_dx_actual_squared[i] = (10.0**param_f_dx_actual[i])*(10.0**param_f_dx_actual[i])
    param_f_dx_actual_cubed[i] = (10.0**param_f_dx_actual[i])*(10.0**param_f_dx_actual[i])*(10.0**param_f_dx_actual[i])
print "param_f_dx_actual_squared"
print param_f_dx_actual_squared
#
param_f_por = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
param_f_por_string = ['1e-4', '2e-4', '3e-4', '4e-4', '5e-4', '6e-4', '7e-4', '8e-4', '9e-4', '1e-3']
param_f_por_actual = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3])
param_f_por_string2 = ['0.01%', '0.02%', '0.03%', '0.04%', '0.05%', '0.06%', '0.07%', '0.08%', '0.09%', '0.10%']

#
# # LIMITED HALF-STEPS
# param_f_dx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
# param_f_dx_string = ['-2.2', '-2.1', '-2.0', '-1.9', '-1.8', '-1.7', '-1.6', '-1.5', '-1.4', '-1.3']
# param_f_dx_actual = np.array([-2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3])
# param_f_dx_actual_squared = np.zeros(param_f_dx_actual.shape)
# param_f_dx_actual_cubed = np.zeros(param_f_dx_actual.shape)
# for i in range(len(param_f_dx_actual)):
#     param_f_dx_actual_squared[i] = (10.0**param_f_dx_actual[i])*(10.0**param_f_dx_actual[i])
#     param_f_dx_actual_cubed[i] = (10.0**param_f_dx_actual[i])*(10.0**param_f_dx_actual[i])*(10.0**param_f_dx_actual[i])
# print "param_f_dx_actual_squared"
# print param_f_dx_actual_squared
# #
# param_f_por = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0])
# param_f_por_string = ['1e-4', '2e-4', '3e-4', '4e-4', '5e-4', '6e-4', '7e-4',  '9e-4', '1e-3']
# param_f_por_actual = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4,  9e-4, 1e-3])
# param_f_por_string2 = ['0.01%', '0.02%', '0.03%', '0.04%', '0.05%', '0.06%', '0.07%', '0.09%', '0.10%']


# param_f_dx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
# param_f_dx_string = ['-2.0', '-1.8', '-1.6', '-1.4', '-1.2', '-1.0']
# param_f_dx_actual = [-2.0, -1.8, -1.6, -1.4, -1.2, -1.0]
#
# param_f_por = np.array([1.0, 2.0, 3.0])
# param_f_por_string = ['2e-4', '6e-4', '1e-3']
# param_f_por_actual = np.array([2e-4, 6e-4, 1e-3])
# param_f_por_string2 = ['0.02%', '0.06%', '0.10%']


param1 = param_f_por
param1_string = param_f_por_string
param2 = param_f_dx
param2_string = param_f_dx_string

u_mean = np.zeros((len(param1)+1,len(param2)+1))
u_1d = np.zeros((len(param1)+1,len(param2)+1))
u_ts = np.zeros((len(param1)+1,len(param2)+1,10))
u_mean_ts = np.zeros((len(param1)+1,len(param2)+1))
u_max = np.zeros((len(param1)+1,len(param2)+1))
hf_range = np.zeros((len(param1)+1,len(param2)+1))
dom = np.zeros((len(param1)+1,len(param2)+1))
f_temp = np.zeros((len(param1)+1,len(param2)+1))
product = np.zeros((len(param1)+1,len(param2)+1))
nu = np.zeros((len(param1)+1,len(param2)+1))
nu_apparent = np.zeros((len(param1)+1,len(param2)+1))
nu_apparent2 = np.zeros((len(param1)+1,len(param2)+1))
q_adv = np.zeros((len(param1)+1,len(param2)+1))
q_adv2 = np.zeros((len(param1)+1,len(param2)+1))
q_adv3 = np.zeros((len(param1)+1,len(param2)+1))
por_mat = np.zeros((len(param1)+1,len(param2)+1))
conv_to_lat = np.zeros((len(param1)+1,len(param2)+1))
conv_to_lat2 = np.zeros((len(param1)+1,len(param2)+1))
temp_mean = np.zeros((len(param1)+1,len(param2)+1))

top_temp_min = np.zeros((len(param1)+1,len(param2)+1))
top_temp_max = np.zeros((len(param1)+1,len(param2)+1))

temp_1023 = np.zeros((len(param1)+1,len(param2)+1))
temp_1024 = np.zeros((len(param1)+1,len(param2)+1))
temp_1025 = np.zeros((len(param1)+1,len(param2)+1))

dtdx_top = np.zeros((len(param1)+1,len(param2)+1))

h_metric1 = np.zeros((len(param1)+1,len(param2)+1))
h_metric2 = np.zeros((len(param1)+1,len(param2)+1))
h_metric3 = np.zeros((len(param1)+1,len(param2)+1))
h_metric4 = np.zeros((len(param1)+1,len(param2)+1))
h_metric5 = np.zeros((len(param1)+1,len(param2)+1))

step1 = 1
step2 = 11

for i in range(len(param1)):
    for j in range(len(param2)):
        print " "
        print param1[i]
        print param2[j]
        
        por_mat[i,j] = param_f_por_actual[i]
        
        u_1d[i,j] = (((10.0**param_f_dx_actual[j]) * (10.0**param_f_dx_actual[j]) * (10.0**param_f_dx_actual[j]) * param_f_por_actual[i] * 9.8) / (12.0 * .001)) * (977.0-999.2)
        u_1d[i,j] = -0.1*u_1d[i,j]*(3.14e7)/param_h
        product[i,j] = np.log10((10.0**param_f_dx_actual[j])*(10.0**param_f_dx_actual[j])*(10.0**param_f_dx_actual[j])*param_f_por_actual[i])
        
        #if param2[j] > param1[i]:
        if 2.0 > 1.0:
            u_last = np.zeros([len(y),len(x)])
            u_last_sum = np.zeros([len(y),len(x)])
            sim_name = 'por' + param1_string[i] + 'b' + param2_string[j]
            path_sim = path + sim_name + "/"

            print sim_name
    
            # load stuff
            uMat0 = np.loadtxt(path_sim + 'uMat.txt')*(3.14e7)#*10.0
            vMat0 = np.loadtxt(path_sim + 'vMat.txt')*(3.14e7)#*10.0
            psiMat0 = np.loadtxt(path_sim + 'psiMat.txt')
            hMat0 = np.loadtxt(path_sim + 'hMat.txt')
            #permMat0 = np.log10(np.loadtxt(path_sim + 'permMat.txt'))
            perm_last = np.log10(np.loadtxt(path_sim + 'permeability.txt'))
            maskP = np.loadtxt(path_sim + 'maskP.txt')
            lambdaMat = np.loadtxt(path_sim + 'lambdaMat.txt')
            temp6 = np.loadtxt(path_sim + 'temp6.txt')
            step = 10
            v_last = vMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
            psi_last = psiMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
            if np.max(np.isnan(psi_last)>0):
                psi_last = np.zeros(psi_last.shape)
                psi_last[:len(y)/2,:] = -1.0e-8
            h_last = hMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
            h_last_mask = np.zeros(h_last.shape)
            #perm_last = permMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
            #u_last[perm_last < -13.0] = 0.0
        
            cap1 = int((300.0/50.0))+4# + 12
            cap2 = int((10))+4# + 12
            # cap1 = int((500.0/50.0)) + 4
            # cap2 = int((500.0/50.0)) + 4
            cap = 19
            # capy = (np.max(param_o[k],param_orhs[l])/(y[1]-y[0])) - 5
            capy = 2#((param_o_rhs)/(y[1]-y[0])) + 5
            
            f_temp[i,j] = temp6[1,-15]
            print "f_temp", f_temp[i,j]


            for stepk in range(step1,step2):
                u_last = uMat0[:,(stepk-1.0)*len(x):(((stepk-1.0))*len(x)+len(x))]
                h_last = hMat0[:,(stepk-1.0)*len(x):(((stepk-1.0))*len(x)+len(x))]
                u_last_sum = u_last_sum + uMat0[:,(stepk-1.0)*len(x):(((stepk-1.0))*len(x)+len(x))]

                colAq = np.zeros(len(x))
                colTemp = np.zeros(len(x))
                colAq_count = np.zeros(len(x))
                aq_count = 0.0
                for n in range(cap1,len(x)-cap2):
                    # ALT MEAN VELOCITY CALC
                    aq_count = aq_count + 1.0
                    for m in range(len(y)):
                        if int(perm_last[m,n]) == -12 and maskP[m,n] != 0.0:
                            colAq_count[n] = colAq_count[n] + 1.0
                            colAq[n] = colAq[n] + u_last[m,n]
                            colTemp[n] = colTemp[n] + h_last[m,n]
                    colAq[n] = colAq[n]/colAq_count[n]
                    colTemp[n] = colTemp[n]/colAq_count[n]
                            
                u_ts[i,j,stepk-1] = np.sum(colAq)/aq_count
                temp_mean[i,j] = np.sum(colTemp)/aq_count
                
            print "temp_mean" , temp_mean[i,j]
                
                # #SORT OF ARBITRARY CONVECTION CUT OFF
                # for n in range(cap1,len(x)-cap2):
                #     for m in range(len(y)):
                #         if int(perm_last[m,n]) == -12 and maskP[m,n] != 0.0:
                #             if u_last[m,n] < 0.0:
                #                 u_ts[i,j,stepk-1] = -1.0
            
        
            
            u_mean_ts[i,j] = np.sum(u_ts[i,j,stepk-1])/1.0
            # ALT VELOCITY
            #u_mean_ts[i,j] = np.sum(colAq)/aq_count
            if np.max(psi_last) == 0.0:
                u_mean_ts[i,j] = 999.0
            print "u_mean_ts[i,j]" , u_mean_ts[i,j]
            
            
            ##########################
            ##    PLOT U MEAN TS    ##
            ##########################
            fig=plt.figure()
            ax1=fig.add_subplot(1,1,1)

            plt.plot(u_ts[i,j,:])

            #plt.ylim([-0.2,0.2])

            plt.savefig(path+'u_mean_ts_'+sim_name+'_.png')


            # ##############################
            # ##    PLOT LAST TIMESTEP    ##
            # ##############################
            # fig=plt.figure()
            #
            # varMat = hMat0
            # varStep = h_last[:,:]
            #
            # contours = np.linspace(np.min(varStep),np.max(varStep),12)
            # ax1=fig.add_subplot(1,1,1, aspect=asp/2.0,frameon=False)
            # pGlass = plt.contourf(x, y[:], varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            # cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
            # p = plt.contour(xg,yg,perm_last,[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
            # CS = plt.contour(xg, yg, psi_last, 10, colors='black',linewidths=np.array([0.5]))
            # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            # plt.ylim([y[0],0.0])
            #
            # # varMat = vMat0
            # # varStep = v_last
            # #
            # # contours = np.linspace(np.min(varMat),np.max(varMat),10)
            # # ax1=fig.add_subplot(2,2,2, aspect=asp,frameon=False)
            # # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            # # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            # # cbar.ax.set_xlabel('v [m/yr]')
            # # p = plt.contour(xg,yg,perm_last,[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
            # # #CS = plt.contour(xg, yg, psi_last, 10, colors='black',linewidths=np.array([0.5]))
            # # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            # #
            # #
            # # varMat = uMat0
            # # varStep = u_last
            # #
            # # contours = np.linspace(np.min(varMat),np.max(varMat),10)
            # # ax1=fig.add_subplot(2,2,1, aspect=asp,frameon=False)
            # # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            # # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            # # cbar.ax.set_xlabel('u [m/yr]')
            # # p = plt.contour(xg,yg,perm_last,[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
            # # #CS = plt.contour(xg, yg, psi_last, 10, colors='black',linewidths=np.array([0.5]))
            # # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            #
            # plt.savefig(path+'jdf_'+sim_name+'_.eps')


            ################
            # AQUIFER PLOT #
            ################

            fig=plt.figure()

            # varMat = v_last[capy:-capy,cap1:-cap2]
            # varStep = v_last[capy:-capy,cap1:-cap2]
            #
            # contours = np.linspace(np.min(varMat),np.max(varMat),20)
            # ax1=fig.add_subplot(2,1,2, aspect=asp,frameon=False)
            # pGlass = plt.contourf(x[cap1:-cap2], y[capy:-capy], varStep, contours,
            #                      cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            # cbar.ax.set_xlabel('v [m/yr]')
            # cbar.solids.set_rasterized(True)
            # cbar.solids.set_edgecolor("face")
            # p = plt.contour(xg[capy:-capy,cap1:-cap2],yg[capy:-capy,cap1:-cap2],perm_last[capy:-capy,cap1:-cap2],
            #                     [-12.0,-13.5],colors='black',linewidths=np.array([1.0]))
            # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            # for c in pGlass.collections:
            #     c.set_edgecolor("face")
            # #plt.xlim([np.max(x)/5.0,np.max(x)*4.0/5.0])
            # #plt.ylim([y[0],0.0])
            #
            varMat = u_last[capy:-capy,cap1:-cap2]
            varStep = u_last[capy:-capy,cap1:-cap2]

            contours = np.linspace(np.min(varMat),np.max(varMat),20)
            ax1=fig.add_subplot(2,1,1, aspect=asp,frameon=False)
            pGlass = plt.contourf(x[cap1:-cap2], y[capy:-capy], varStep, contours,
                                 cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            cbar.ax.set_xlabel('u [m/yr]')
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")
            p = plt.contour(xg[capy:-capy,cap1:-cap2],yg[capy:-capy,cap1:-cap2],perm_last[capy:-capy,cap1:-cap2],
                                [-12.0,-13.5],colors='black',linewidths=np.array([1.0]))
            cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            for c in pGlass.collections:
                c.set_edgecolor("face")
            #plt.xlim([np.max(x)/5.0,np.max(x)*4.0/5.0])
            #plt.ylim([y[0],0.0])


            varMat = h_last[capy:-capy,cap1:-cap2]
            varStep = h_last[capy:-capy,cap1:-cap2]

            contours = np.arange(275.0,525.0,25.0)
            ax1=fig.add_subplot(2,1,2, aspect=asp,frameon=False)
            pGlass = plt.contourf(x[cap1:-cap2], y[capy:-capy], varStep, contours,
                                 cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            cbar.ax.set_xlabel('v [m/yr]')
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")
            p = plt.contour(xg[capy:-capy,cap1:-cap2],yg[capy:-capy,cap1:-cap2],perm_last[capy:-capy,cap1:-cap2],
                                [-12.0,-13.5],colors='black',linewidths=np.array([1.0]))
            cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            for c in pGlass.collections:
                c.set_edgecolor("face")
            #plt.xlim([np.max(x)/5.0,np.max(x)*4.0/5.0])
            #plt.ylim([y[0],0.0])


            plt.savefig(path+'aq_'+sim_name+'.eps')
            #
            #
            #
            ######################################
            #        ZOOM OUTCROP PSI PLOT       #
            ######################################

            lim_a = 0.0
            lim_b = 2000.0
            lim_a0 = int(lim_a/(x[1]-x[0]))
            lim_b0 = int(lim_b/(x[1]-x[0]))
            fset = 5

            lim_u = 0
            lim_o = len(y)

            fig=plt.figure()

            varMat = h_last#[lim_u:lim_o,lim_a0:lim_b0]
            varStep = h_last[lim_u:lim_o,lim_a0:lim_b0]
            # contours = np.linspace(np.min(varMat),np.max(varMat),20)
            contours = np.arange(275.0,525.0,25.0)

            ax1=fig.add_subplot(2,2,1, aspect=asp/8.0,frameon=False)
            pGlass = plt.contourf(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep, contours, cmap=cm.rainbow)
            CS = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0], yg[lim_u:lim_o,lim_a0:lim_b0], psi_last[lim_u:lim_o,lim_a0:lim_b0], 8, colors='black',linewidths=np.array([0.5]))
            p = plt.contour(xg,yg,perm_last,[-15.9],colors='black',linewidths=np.array([1.5]))
            cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
            for c in pGlass.collections:
                c.set_edgecolor("face")

            plt.xlim([lim_a,lim_b])
            plt.ylim([np.min(y),0.])
            cbar = plt.colorbar(pGlass, orientation='horizontal')
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")




            varMat = h_last#[lim_u:lim_o,xn-lim_b0-fset:xn-lim_a0-fset]
            varStep = h_last[lim_u:lim_o,xn-lim_b0-fset:xn-lim_a0-fset]
            # contours = np.linspace(np.min(varMat),np.max(varMat),20)
            contours = np.arange(275.0,525.0,25.0)

            ax1=fig.add_subplot(2,2,2, aspect=asp/8.0,frameon=False)
            pGlass = plt.contourf(x[xn-lim_b0-fset:xn-lim_a0-fset], y[lim_u:lim_o], varStep, contours, cmap=cm.rainbow)
            CS = plt.contour(xg[lim_u:lim_o,xn-lim_b0-fset:xn-lim_a0-fset], yg[lim_u:lim_o,xn-lim_b0-fset:xn-lim_a0-fset], psi_last[lim_u:lim_o,xn-lim_b0-fset:xn-lim_a0-fset], 8, colors='black',linewidths=np.array([0.5]))
            p = plt.contour(xg,yg,perm_last,[-15.9],colors='black',linewidths=np.array([1.5]))
            cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
            for c in pGlass.collections:
                c.set_edgecolor("face")

            plt.xlim([np.max(x)-lim_b-250.0,np.max(x)-lim_a-250.0])
            plt.ylim([np.min(y),0.])
            cbar = plt.colorbar(pGlass, orientation='horizontal')
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")




            varMat = psi_last#[lim_u:lim_o,lim_a0:lim_b0]
            varStep = psi_last[lim_u:lim_o,lim_a0:lim_b0]
            contours = np.linspace(np.min(varMat),np.max(varMat),20)

            ax1=fig.add_subplot(2,2,3, aspect=asp/8.0,frameon=False)
            pGlass = plt.pcolor(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep, cmap=cm.rainbow)
            pGlass.set_edgecolor('face')


            plt.xlim([lim_a,lim_b])
            plt.ylim([np.min(y),0.])
            cbar = plt.colorbar(pGlass, orientation='horizontal')
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")



            varMat = psi_last#[lim_u:lim_o,xn-lim_b0-fset:xn-lim_a0-fset]
            varStep = psi_last[lim_u:lim_o,xn-lim_b0-fset:xn-lim_a0-fset]
            contours = np.linspace(np.min(varMat),np.max(varMat),20)

            ax1=fig.add_subplot(2,2,4, aspect=asp/8.0,frameon=False)
            pGlass = plt.pcolor(x[xn-lim_b0-fset:xn-lim_a0-fset], y[lim_u:lim_o], varStep, cmap=cm.rainbow)
            pGlass.set_edgecolor('face')


            plt.xlim([np.max(x)-lim_b-250.0,np.max(x)-lim_a-250.0])
            plt.ylim([np.min(y),0.])
            cbar = plt.colorbar(pGlass, orientation='horizontal')
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")



            plt.savefig(path+'zoomVEL_'+sim_name+'.eps')



    

            ##########################
            ##    PLOT HEAT FLOW    ##
            ##########################

            fig=plt.figure()
            ax1=fig.add_subplot(2,1,1)

            heatVec = np.zeros(len(h_last[-1,:]))
            heatSed = np.zeros(len(h_last[-1,:]))
            heatBottom = np.zeros(len(h_last[-1,:]))
            heatLeft = np.zeros(len(h_last[:,-1]))
            heatRight = np.zeros(len(h_last[:,-1]))
            h_conductive = np.zeros(len(h_last[-1,:]))
            
            h_top = np.zeros(len(h_last[-1,:]))
            h_base = np.zeros(len(h_last[-1,:]))
            h_low = np.zeros(len(h_last[-1,:]))
            hc_top = np.zeros(len(h_last[-1,:]))
            hc_base = np.zeros(len(h_last[-1,:]))
            nusselt = np.zeros(len(h_last[-1,:]))
            hc_base2 = np.zeros(len(h_last[-1,:]))
            hc_top2 = np.zeros(len(h_last[-1,:]))
            
            hc_x_adv = np.zeros(len(h_last[-1,:]))
            hc_y_adv = np.zeros(len(h_last[-1,:]))
            hc_x_cond = np.zeros(len(h_last[-1,:]))
            hc_y_cond = np.zeros(len(h_last[-1,:]))
            
            top_temp = np.zeros(len(h_last[-1,:]))
            
            rho_h = 2080.0
            cp_h = 4200.0
            
            v_last_mask = np.zeros(v_last.shape)
            
            tt = 0
            for m in range(len(x)-1):
                for n in range(1,len(y-1)):
                    if int(perm_last[n-1,m]) == -12 and int(perm_last[n,m]) != -12 and np.any(maskP[:,m] == 50.0):
                         v_last_mask[n-9:n-1,m] = v_last[n-9:n-1,m]
            #             #print h_last_mask[n-9:n-1,20]
                        
            h_last_mask = h_last
            # print h_last_mask[n-9:n-1,20]
            
            # (10.0**param_f_dx_actual[j])
            
            bsi = np.zeros(len(h_last[-1,:]))
            
            for m in range(10,len(x)-20):
                for n in range(1,len(y-1)):
                    if int(perm_last[n-1,m]) == -12 and int(perm_last[n,m]) != -12 and np.any(maskP[:,m] == 50.0):
                        top_temp[m] = h_last[n-1,m] - 273.0
                        bsi[m] = n
            for m in range(10,len(x)-20):
                for n in range(1,len(y-1)):
                    if int(perm_last[n-1,m]) == -12 and int(perm_last[n,m]) != -12 and np.any(maskP[:,m] == 50.0):
                        if x[m] + 300.0 == 3000.0:
                            temp_1023[i,j] = np.sum(top_temp[m-10:m+10])/20.0-273.0#h_last[n-1,m] - 273.15
                        if x[m] + 300.0 == 6000.0:
                            temp_1024[i,j] = np.sum(top_temp[m-10:m+10])/20.0-273.0#h_last[n-1,m] - 273.15
                        if x[m] + 300.0 == 14500.0:
                            temp_1025[i,j] = np.sum(top_temp[m-10:m+10])/20.0-273.0#h_last[n-1,m] - 273.15
                            

            print "temp 1023" , temp_1023[i,j]
            print "temp 1024" , temp_1024[i,j]
            print "temp 1025" , temp_1025[i,j]
            dtdx_top[i,j] = (temp_1025[i,j] - temp_1023[i,j])/ 12.5
            print "dtdx top" , dtdx_top[i,j]
                    
            for m in range(10,len(x)-20):
                heatBottom[m] = -1000.0*1.8*(h_last[2,m] - h_last[1,m])/(y[1]-y[0])
                for n in range(1,len(y-1)):
                    #if int(perm_last[n-1,m]) == -12 and int(perm_last[n,m]) != -12 and np.any(maskP[:,m] == 50.0):
                    if int(perm_last[n,m]) == -12 and maskP[n,m] == 1 and np.any(maskP[:,m] == 50.0):
                        #print n, m, "n, m"

                        hc_top[m] = hc_top[m]# + mpd.rip_lith_y[m]*1000.0#(-1000.0*1.8*(h_last[n,m] - h_last[n-1,m])/(y[1]-y[0]))
                        
                        # GOOD CONDUCTION
                        hc_x_cond[m] = hc_x_cond[m] - 1000.0*1.8*(h_last[n,m+2] - h_last[n,m])/(2.0*(x[1]-x[0]))
                        hc_x_cond[m] = hc_x_cond[m] - 1000.0*1.8*(h_last[n,m] - h_last[n,m-2])/(2.0*(x[1]-x[0]))
                        
                        # hc_y_cond[m] = hc_y_cond[m] - 1000.0*1.8*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])/8.0
                        # hc_y_cond[m] = hc_y_cond[m] - 1000.0*1.8*(h_last[n,m] - h_last[n-1,m])/(y[1]-y[0])/8.0
                        
                        # if np.abs(8.0*1000.0*1.8*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])) > np.abs(hc_y_cond[m]):
                        #     hc_y_cond[m] = -8.0*1000.0*1.8*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])
                        # if np.abs(8.0*1000.0*1.8*(h_last[n,m] - h_last[n-1,m])/(y[1]-y[0])) > np.abs(hc_y_cond[m]):
                        #     hc_y_cond[m] = -8.0*1000.0*1.8*(h_last[n,m] - h_last[n-1,m])/(y[1]-y[0])
                        
                        # if int(perm_last[n,m]) == -12 and int(perm_last[n+1,m]) != -12 and np.any(maskP[:,m] == 50.0):
                        #     hc_y_cond[m] = hc_y_cond[m] - 1000.0*1.8*(h_last[n+1,m] - h_last[n-3,m])/((y[1]-y[0]))
                        if int(perm_last[n,m]) == -12 and int(perm_last[n+1,m]) != -12 and np.any(maskP[:,m] == 50.0):
                            hc_y_cond[m] = 1000.0*mpd.rip_lith_y[m]#- 1000.0*1.8*(h_last[n,m] - h_last[n-1,m])/(y[1]-y[0])
                        #     hc_y_cond[m] = hc_y_cond[m] - 1000.0*1.8*(h_last[n+3,m] - h_last[n-1,m])/((y[1]-y[0]))
                        
                        # if u_last[n,m] > 0.0 and bsi[m] == bsi[m-1] and bsi[m] == bsi[m+1] and bsi[m] == bsi[m-2] and bsi[m] == bsi[m+2]:
                        #     hc_x_adv[m] = hc_x_adv[m] + u_last[n,m]*rho_h*cp_h*(h_last[n,m] - h_last[n,m-1])/10000.0
                        # if u_last[n,m] < 0.0 and bsi[m] == bsi[m-1] and bsi[m] == bsi[m+1] == bsi[m-2] and bsi[m] == bsi[m+2]:
                        #     hc_x_adv[m] = hc_x_adv[m] + u_last[n,m]*rho_h*cp_h*(h_last[n,m+1] - h_last[n,m])/10000.0
                        # if v_last[n,m] > 0.0:
                        #     hc_y_adv[m] = hc_y_adv[m] + v_last[n,m]*rho_h*cp_h*(h_last[n,m] - h_last[n-1,m])/80000.0
                        # if v_last[n,m] < 0.0:
                        #     hc_y_adv[m] = hc_y_adv[m] + v_last[n,m]*rho_h*cp_h*(h_last[n+1,m] - h_last[n,m])/80000.0
                        
                        if u_last[n,m] > 0.0:
                            hc_x_adv[m] = hc_x_adv[m] + 1000.0*(u_last[n,m]/3.14e7)*rho_h*cp_h*(h_last[n,m] - h_last[n,m-1])/10.0
                            #hc_x_cond[m] = hc_x_cond[m] - 1000.0*1.8*(h_last[n,m+1] - h_last[n,m])/(1.0*(x[1]-x[0]))
                        if u_last[n,m] < 0.0:
                            hc_x_adv[m] = hc_x_adv[m] + 1000.0*(u_last[n,m]/3.14e7)*rho_h*cp_h*(h_last[n,m+1] - h_last[n,m])/10.0
                            #hc_x_cond[m] = hc_x_cond[m] - 1000.0*1.8*(h_last[n,m] - h_last[n,m-1])/((x[1]-x[0]))
                        if v_last[n,m] > 0.0:
                            hc_y_adv[m] = hc_y_adv[m] + 1000.0*(v_last[n,m]/3.14e7)*rho_h*cp_h*(h_last[n,m] - h_last[n-1,m])/(10.0*cell_thing)
                        if v_last[n,m] < 0.0:
                            hc_y_adv[m] = hc_y_adv[m] + 1000.0*(v_last[n,m]/3.14e7)*rho_h*cp_h*(h_last[n+1,m] - h_last[n,m])/(10.0*cell_thing)

                        # if u_last[n,m] > 0.0 and int(perm_last[n,m+1]) == -12 and int(perm_last[n,m-1]) == -12:
#                             hc_x_adv[m] = hc_x_adv[m] + 0.5*(u_last[n,m]+u_last[n,m-1])*rho_h*cp_h*(h_last[n,m] - h_last[n,m-1])/80000.0
#                         if u_last[n,m] < 0.0 and int(perm_last[n,m-1]) == -12 and int(perm_last[n,m+1]) == -12:
#                             hc_x_adv[m] = hc_x_adv[m] + 0.5*(u_last[n,m]+u_last[n,m+1])*rho_h*cp_h*(h_last[n,m+1] - h_last[n,m])/80000.0

                        # if u_last[n,m] > 0.0 and np.all(perm_last[n-9:n+9,m+1] == perm_last[n-9:n+9,m]) and np.all(perm_last[n-9:n+9,m-1] == perm_last[n-9:n+9,m]):
                        #     hc_x_adv[m] = hc_x_adv[m] + 0.5*(u_last[n,m]+u_last[n,m-1])*rho_h*cp_h*(h_last[n,m] - h_last[n,m-1])/80000.0
                        # if u_last[n,m] < 0.0 and np.all(perm_last[n-9:n+9,m+1] == perm_last[n-9:n+9,m]) and np.all(perm_last[n-9:n+9,m-1] == perm_last[n-9:n+9,m]):
                        #     hc_x_adv[m] = hc_x_adv[m] + 0.5*(u_last[n,m]+u_last[n,m+1])*rho_h*cp_h*(h_last[n,m+1] - h_last[n,m])/80000.0
                        #
                            
                            
                        #
                        #
                        #
                        # if v_last[n,m] > 0.0:
                        #     hc_y_adv[m] = hc_y_adv[m] + 0.5*(v_last[n,m]+v_last[n-1,m])*rho_h*cp_h*(h_last[n,m] - h_last[n-1,m])/80000.0
                        # if v_last[n,m] < 0.0:
                        #     hc_y_adv[m] = hc_y_adv[m] + 0.5*(v_last[n+1,m]+v_last[n,m])*rho_h*cp_h*(h_last[n+1,m] - h_last[n,m])/80000.0
                            
                        hc_base[m] = (-1000.0*1.8*(h_last[n-1,m] - h_last[n-2,m])/(1.0*(y[1]-y[0])))
                        
                
                        
                    if (maskP[n,m] == 25.0) or (maskP[n,m] == 50.0):
                        heatVec[m] = -1000.0*1.2*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])
                        heatSed[m] = -1000.0*1.2*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])
                for n in range(len(y)):
                    if maskP[n,m] == 5.0:
                        heatLeft[n] = -1000.0*lambdaMat[n,m+1]*(h_last[n,m+1] - h_last[n,m])/(x[1]-x[0])
                    if maskP[n,m] == 10.0:
                        heatRight[n] = -1000.0*lambdaMat[n,m-1]*(h_last[n,m-1] - h_last[n,m])/(x[1]-x[0])
                        
            nusselt = mpd.rip_lith_y/1.8
            #hf = -1000.0*lambdaMat[-1,:]*(h_last[-1,:] - h_last[-2,:])/(y[1]-y[0])
            #hc_base = hc_y_cond
            hc_base[-12:] = 0.0
            hc_y_cond = hc_y_cond
            hc_base = hc_y_cond
            
            if np.mean(np.abs(hc_x_cond)) < np.mean(np.abs(hc_x_adv)):
                hc_x_cond = -1.0*hc_x_adv
                
            # for m in range(10,len(x)-20):
            #     if np.abs(hc_x_cond[m] < np.abs(hc_x_adv[m]):
            #         hc_x_cond[m] = -1.0*hc_x_adv[m]
                    
            hc_top = hc_y_cond + hc_x_cond + hc_x_adv + 0.0*hc_y_adv
            hc_top[-12:] = 0.0
            #hc_top = hc_top/8.0
            
            


            heatVec = heatVec#*np.mean(sed1)/sed1#np.mean(sed2)/(sed2)
            sh = 35
            sh2 = 35#40
            sh3 = 35#sh - (sh2 - sh)
            # shc = int(param_w[i]/130.0)

            plt.ylim([-1500.0,3000.0])
            plt.xlim([np.min(x),np.max(x)])
            # plt.plot(mpd.steinX1[::2],mpd.steinY1[::2], 'k', linestyle="-.", lw=1, label='Stein 2004 7.9 m/yr')
            # plt.plot(mpd.steinX2[::2],mpd.steinY2[::2], 'k', linestyle="--", lw=1, label='Stein 2004 4.9 m/yr')
            #p = plt.plot(x,hf_interp,'k-',linewidth=1.0)

            heat_actual4 = heatVec[sh2-tt:-sh3-tt] + np.abs(((np.mean(sed1[sh2:-sh3])-(sed1[sh2:-sh3]))/((sed1[sh2:-sh3]-sed[sh2:-sh3])))*heatVec[sh2-tt:-sh3-tt])
            heat_actual3 = heatVec[sh2-tt:-sh3-tt] + ((sed[sh2:-sh3]-sed2[sh2:-sh3])/(np.mean(sed1[sh2:-sh3])-sed[sh2:-sh3]))*heatVec[sh2-tt:-sh3-tt]
            heat_actual2 = heatVec[sh2-tt:-sh3-tt] + ((sed[sh2:-sh3]-np.mean(sed2[sh2:-sh3]))/(sed1[sh2:-sh3]-sed[sh2:-sh3]))*heatVec[sh2-tt:-sh3-tt]
            heat_actual = heatVec[sh2-tt:-sh3-tt] + ((sed[sh2:-sh3]-sed2[sh2:-sh3])/(np.mean(150.0)))*heatVec[sh2-tt:-sh3-tt]

            # p = plt.plot(x[sh:-sh]-750.0,heat_actual,'c',linewidth=1.5,label='Model adjusted heat flow')
            # p = plt.plot(x[sh:-sh]-750.0,heat_actual2,'m',linewidth=1.5,label='Model adjusted heat flow')
            # p = plt.plot(x[sh:-sh]-750.0,heat_actual3,'gold',linewidth=1.5,label='Model adjusted heat flow')
            #p = plt.plot(x[sh:-sh]-700.0,heat_actual4,'g',linewidth=1.5,label='Model adjusted heat flow')
            # -750


            #plt.scatter(mpd.ripXdata,mpd.ripQdata,s=15,c='k')
            #p = plt.plot(x,heatSed,'b',linewidth=2.0,label='sediment only')
            #p = plt.plot(x,heatBottom,'gold',linewidth=2.0,label='Conduction')
            p = plt.plot(x,mpd.rip_lith_y*1000.0,'gold',linewidth=2.0)
            #p = plt.plot(x,h_conductive,'c',linewidth=2.0,label='h_conductive')
            
            p = plt.plot(x,hc_base,'r',linewidth=1.0,label='hc_base')
            p = plt.plot(x,hc_x_cond,'g',linewidth=1.0,label='hc_x_cond')
            p = plt.plot(x,hc_y_cond,'purple',linewidth=1.0,label='hc_y_cond (base?)')
            p = plt.plot(x,hc_x_adv,'orange',linewidth=1.0,label='hc_x_adv')
            p = plt.plot(x,hc_y_adv,'c',linewidth=1.0,label='hc_y_adv')
            p = plt.plot(x,hc_top,'k',linewidth=1.5,label='hc_top')

            #ax1.set_xticks(x,minor=True)
            #ax1.set_xticks(x[::5])
            #ax1.grid(which='minor', alpha=0.5)
            ax1.grid(which='major', alpha=1.0)
            
            hf_interp_adjusted = hf_interp[sh-14:-sh-14]

            #hf_range[i,j] =  np.sum(np.abs(hf_interp_adjusted[heat_actual4>0] - heat_actual4[heat_actual4>0]))/np.sum(np.abs(hf_interp_adjusted[heat_actual4>0]))
            #hf_range[i,j] =  np.mean(np.abs(hf_interp_adjusted[heat_actual4>0] - heat_actual4[heat_actual4>0])/np.abs(hf_interp_adjusted[heat_actual4>0]))
            hf_range[i,j] = np.abs(np.sum(hf_interp_adjusted[heat_actual4>0] - heat_actual4[heat_actual4>0])/np.sum(np.abs(hf_interp_adjusted[heat_actual4>0])))
            print  "heat flow error" , hf_range[i,j]
            print " "
            
            q_adv2[i,j] = np.mean(mpd.rip_lith_y[heatSed>0.0]) - np.mean(heatSed[heatSed>0.0])/1000.0
            print "q_adv2"
            print q_adv2[i,j]
            
            q_adv3[i,j] = np.mean(hc_base[heatSed>0.0]) - np.mean(heatSed[heatSed>0.0])/1000.0
            print "q_adv3"
            print q_adv3[i,j]
            

            
            h_metric1[i,j] = np.abs(np.mean(np.abs(1000.0*mpd.rip_lith_y[heat_actual4>0] - heat_actual4[heat_actual4>0]))/np.mean(1000.0*mpd.rip_lith_y[heat_actual4>0] - heat_actual4[heat_actual4>0]))
            h_metric2[i,j] = np.abs(np.mean((1000.0*mpd.rip_lith_y[heatSed>0] - heatSed[heatSed>0])/heatSed[heatSed>0]))# - 1.0
            print "h_metric1"
            print h_metric1[i,j]
            print "h_metric2"
            print h_metric2[i,j]
            
            plt.ylabel('heat flow [mW/m^2]')
            plt.xlabel('x direction [m]')
            plt.legend(fontsize=8,loc='best',ncol=4)
            
            
            print " "
            print " "

            
            ax1=fig.add_subplot(2,1,2)
            heatSed[-12:] = 0.0
            
            # p = plt.plot(x,hc_top/hc_base,'r',linewidth=2.0,label='h_base')
            p = plt.plot(x,(np.abs(hc_top))/(hc_base),'b',linewidth=2.0,label='h_base')
            
            
            plt.ylim([-20.0,20.0])
            
            nu[i,j] = np.mean(1.0 + np.abs((hc_top[hc_base>0]-hc_base[hc_base>0]))/(hc_base[hc_base>0]))
            # if u_mean_ts[i,j] > 10.0:
            #     nu[i,j] = 1.0
            
            # nu_apparent[i,j] = np.mean(1.0 + np.abs((1000.0*mpd.rip_lith_y[heat_actual4>1.0]-heat_actual4[heat_actual4>1.0]))/(heat_actual4[heat_actual4>1.0]))
            #
            # nu_apparent2[i,j] = np.mean(1.0 + np.abs((1000.0*mpd.rip_lith_y[heatSed>1.0]-heatSed[heatSed>1.0]))/(heatSed[heatSed>1.0]))
            
            nu_apparent[i,j] = np.sum(1000.0*mpd.rip_lith_y[heat_actual4>0.0])/np.sum(heat_actual4[heat_actual4>0.0])
            nu_apparent2[i,j] = np.sum(1000.0*mpd.rip_lith_y[heatSed>0.0])/np.sum(heatSed[heatSed>0.0])
            print "nu??" , nu[i,j]

            plt.savefig(path+'hf_'+sim_name+'.png')
            
            
            

            # fig=plt.figure()
            #
            # #
            # ax1=fig.add_subplot(1,2,1)
            #
            # plt.scatter(mpd.ripXdata,mpd.ripQdata,s=15,c='k')
            # p = plt.plot(x,1000.0*mpd.rip_lith_y)
            # p = plt.plot(x,heatSed,'b',linewidth=2.0,label='sediment only')
            # p = plt.plot(x[sh:-sh],heat_actual4,'g',linewidth=1.5,label='Model adjusted heat flow')
            # p = plt.plot(x[sh:-sh],hf_interp_adjusted)
            #
            # plt.xlim([np.min(x),np.max(x)])
            # plt.xticks([0.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0])
            # plt.ylim([-100.0,700.0])
            # plt.yticks([0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
            # ax1.grid(which='major', alpha=1.0)
            # plt.legend(fontsize=6,loc='best',ncol=1)
            #
            # x0,x1 = ax1.get_xlim()
            # y0,y1 = ax1.get_ylim()
            # ax1.set_aspect(abs(x1-x0)/abs(y1-y0))
            #
            #
            #
            # ax1=fig.add_subplot(1,2,2)
            #
            # plt.scatter([3000.0, 6000.0, 14500], [15.5, 22.4, 38.2],s=15,c='k')
            # p = plt.plot(x,top_temp,lw=2)
            #
            # plt.xlim([np.min(x),np.max(x)])
            # plt.xticks([0.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0])
            # plt.ylim([0.0,100.0])
            # plt.yticks([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
            # ax1.grid(which='major', alpha=1.0)
            #
            # x0,x1 = ax1.get_xlim()
            # y0,y1 = ax1.get_ylim()
            # ax1.set_aspect(abs(x1-x0)/abs(y1-y0))
            #
            #
            # plt.savefig(path+'mini_'+sim_name+'.eps')
            #
            
            # print "heatactual4 > 0"
            # print heat_actual4[heat_actual4>0]
            #
            # print "heatsed > 0"
            # print heatSed[heatSed>0]
            
            
            # #############################
            # ##    PLOT HEAT FLOW ALT   ##
            # #############################
            #
            # fig=plt.figure()
            # ax1=fig.add_subplot(1,1,1)
            #
            #
            # p = plt.plot(x,mpd.rip_lith_y*1000.0,'gold',linewidth=2.0)
            # p = plt.plot(x[sh:-sh]-700.0,heat_actual4,'g',linewidth=1.5,label='heat_actual4')
            # plt.scatter(mpd.ripXdata,mpd.ripQdata,s=15,c='k')
            # p = plt.plot(x,heatSed,'b',linewidth=2.0,label='heatSed')
            # p = plt.plot(x,hc_base,'r',linewidth=1.0,label='hc_base')
            #
            # plt.ylabel('heat flow [mW/m^2]')
            # plt.xlabel('x direction [m]')
            # plt.legend(fontsize=8,loc='best',ncol=4)
            #
            #
            # plt.savefig(path+'hf_alt_'+sim_name+'.eps')


            
            
          


######################
##    FIRST PLOT    ##
######################

fig=plt.figure()

x_param = param2
y_param = param1

x_shift = x_param[1]-x_param[0]
y_shift = y_param[1]-y_param[0]
print 'umean'
#u_mean[:,:,:,2] = 0.0
print u_mean_ts
u_mean_raw = copy.deepcopy(u_mean_ts)


x_param = np.append(x_param, x_param[-1]+x_shift)
y_param = np.append(y_param, y_param[-1]+y_shift)
print x_param
print y_param


asp_multi = (float(x_param[-1])-float(x_param[0]))/(float(y_param[-1])-float(y_param[0]))
print "asp_multi" , asp_multi
print x_param.shape
print y_param.shape
print u_mean.shape

u_mean_ts[-1,:] = u_mean_ts[-2,:]
u_mean_ts[:,-1] = u_mean_ts[:,-2]

# ax1=fig.add_subplot(1,2,1, aspect=asp_multi)
ax1=fig.add_subplot(2,3,1, aspect=asp_multi)

#u_mean_ts[u_mean_ts<0.0] = None
u_mean_ts_mask = ma.array(u_mean_ts,mask=np.isnan(u_mean_ts))
#u_mean_ts[u_mean_ts>=10.0] = 10.0

pCol = plt.pcolor(x_param, y_param, u_mean_ts_mask, cmap=cm.rainbow, vmin=0.0, vmax=10.0)
pCol.set_edgecolor('face')
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('fracture width b [m]', fontsize=10)
plt.ylabel('fracture spacing [m]', fontsize=10)
ax1.set_xticklabels(param_f_dx_string)
ax1.set_yticklabels(param_f_por_string)
plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
cbar = plt.colorbar(pCol, orientation='horizontal')
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")





f_temp[-1,:] = f_temp[-2,:]
f_temp[:,-1] = f_temp[:,-2]

# ax1=fig.add_subplot(1,2,1, aspect=asp_multi)
ax1=fig.add_subplot(2,3,2, aspect=asp_multi)

f_temp[f_temp<1.0] = None
f_temp_mask = ma.array(f_temp,mask=np.isnan(f_temp))


pCol = plt.pcolor(x_param, y_param, f_temp_mask, cmap=cm.rainbow)
pCol.set_edgecolor('face')
#pCol = plt.pcolor(x_param, y_param, product, cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('fracture width b [m]', fontsize=10)
plt.ylabel('fracture spacing [m]', fontsize=10)
ax1.set_xticklabels(param_f_dx_string)
ax1.set_yticklabels(param_f_por_string)
plt.title('fluid temp in fracture [K]', fontsize=10)
#plt.title('product', fontsize=10)
cbar = plt.colorbar(pCol, orientation='horizontal')
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")



u_mean_ts[-1,:] = u_mean_ts[-2,:]
u_mean_ts[:,-1] = u_mean_ts[:,-2]



# ax1=fig.add_subplot(1,2,1, aspect=asp_multi)
ax1=fig.add_subplot(2,3,3, aspect=asp_multi)

# print "u_mean_ts"
# print u_mean_ts

u_mean_ts0 = copy.deepcopy(u_mean_ts)

u_mean_ts0[u_mean_ts0<0.0] = 0.0
u_mean_ts0[u_mean_ts0>0.0] = 1.0

print "u_mean_ts0"
print u_mean_ts0



u_mean_ts1 = copy.deepcopy(u_mean_ts)

u_mean_ts1[u_mean_ts1<2.0] = 0.0
u_mean_ts1[u_mean_ts1>=2.0] = 1.0

print "u_mean_ts1"
print u_mean_ts1


u_mean_ts2 = copy.deepcopy(u_mean_ts)

u_mean_ts2[u_mean_ts2<10.0] = 0.0
u_mean_ts2[u_mean_ts2>=10.0] = 1.0

print "u_mean_ts2"
print u_mean_ts2


pCol = plt.pcolor(x_param, y_param, u_mean_ts0 + u_mean_ts1 + u_mean_ts2, cmap=cm.rainbow)
pCol.set_edgecolor('face')
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('fracture width b [m]', fontsize=10)
plt.ylabel('fracture spacing [m]', fontsize=10)
ax1.set_xticklabels(param_f_dx_string)
ax1.set_yticklabels(param_f_por_string)
#plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
cbar = plt.colorbar(pCol, orientation='horizontal')
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")








ax1=fig.add_subplot(2,3,4, aspect=asp_multi)

print ""
print "hf range before"
print hf_range
print " "
hf_range[np.isnan(hf_range)] = 1.0
print "hf range after"
print hf_range
print " "


pCol = plt.pcolor(x_param, y_param, hf_range, cmap=cm.Greens)
pCol.set_edgecolor('face')
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('fracture width b [m]', fontsize=10)
plt.ylabel('fracture spacing [m]', fontsize=10)
ax1.set_xticklabels(param_f_dx_string)
ax1.set_yticklabels(param_f_por_string)
#plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
cbar = plt.colorbar(pCol, orientation='horizontal')
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")





ax1=fig.add_subplot(2,3,5, aspect=asp_multi)

print ""
print "nu"
print nu

nu[nu<1.0] = 1.0


pCol = plt.pcolor(x_param, y_param, nu, cmap=cm.rainbow)
pCol.set_edgecolor('face')
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('fracture width b [m]', fontsize=10)
plt.ylabel('fracture spacing [m]', fontsize=10)
ax1.set_xticklabels(param_f_dx_string)
ax1.set_yticklabels(param_f_por_string)
#plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
cbar = plt.colorbar(pCol, orientation='horizontal')
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")

fig.set_tight_layout(True)
plt.subplots_adjust(hspace=0.3)


# st = fig.suptitle(str(param_h))
plt.savefig(path+'pCol.eps')













#######################
##    SECOND PLOT    ##
#######################

print "u_mean_raw"
print u_mean_raw


print "u_1d"
print u_1d

fig=plt.figure()

ax1=fig.add_subplot(2,2,1)

for i in range(len(param_f_por)):
    plt.plot(param_f_dx_actual,nu[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=6.0,zorder=3,lw=1.0,ls='-')
plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
plt.ylim([0.0,7.0])
plt.yticks(np.arange(0.0,7.0))
legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
plt.setp(legend.get_title(),fontsize=6)
ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$(b)', fontsize=8)
plt.ylabel('Nu actual', fontsize=8)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))




ax1=fig.add_subplot(2,2,2)

for i in range(len(param_f_por)):
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu_apparent2[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
#plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
#legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
# plt.setp(legend.get_title(),fontsize=6)
plt.ylim([0.0,7.0])
plt.yticks(np.arange(0.0,7.0))
ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$(b)', fontsize=8)
plt.ylabel('Nu actual', fontsize=8)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))






ax1=fig.add_subplot(2,2,3)

for i in range(len(param_f_por)):
    plt.plot(param_f_dx_actual,nu_apparent2[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=6.0,zorder=3,lw=1.0,ls='-')
plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
plt.ylim([0.0,7.0])
plt.yticks(np.arange(0.0,7.0))
legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
plt.setp(legend.get_title(),fontsize=6)
ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$(b)', fontsize=8)
plt.ylabel('Nu apparent', fontsize=8)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))




ax1=fig.add_subplot(2,2,4)

for i in range(len(param_f_por)):
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu_apparent2[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
#legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
# plt.setp(legend.get_title(),fontsize=6)
plt.ylim([0.0,7.0])
plt.yticks(np.arange(0.0,7.0))
ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$(b)', fontsize=8)
plt.ylabel('Nu apparent', fontsize=8)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))


plt.subplots_adjust(wspace=0.1,hspace=0.1)

plt.savefig(path+'pLot.eps')








conv_to_lat = np.abs((nu - 1.0)/(nu - (1.0/nu_apparent)))
conv_to_lat2 = np.abs((nu - 1.0)/(nu - (1.0/nu_apparent2)))



fig=plt.figure()

ax1=fig.add_subplot(1,2,1)

# for i in range(len(param_f_por)):
for i in [0,len(param_f_por)-1]:
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu[i,:-1],color_string[0],label='nu_actual',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu_apparent[i,:-1],color_string[1],label='nu_apparent',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),conv_to_lat[i,:-1],color_string[2],label='nu_apparent',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
#plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
plt.setp(legend.get_title(),fontsize=6)
plt.ylim([-0.5,5.5])
plt.yticks(np.arange(0.0,6.0))
ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$($\phi_f$b$^{3}$)', fontsize=12)
plt.ylabel('Nu', fontsize=12)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))



ax1=fig.add_subplot(1,2,2)

for i in range(len(param_f_por)):
    plt.plot(param_f_dx_actual,nu[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(param_f_dx_actual,nu_apparent2[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(param_f_dx_actual,(nu[i,:-1])/nu_apparent2[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(param_f_dx_actual,(nu_apparent2[i,:-1])/nu[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(param_f_dx_actual,conv_to_lat2[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
#legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
# plt.setp(legend.get_title(),fontsize=6)
plt.ylim([0.0,7.0])
plt.yticks(np.arange(0.0,7.0))
ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$($\phi_f$b$^{3}$)', fontsize=12)
plt.ylabel('Nu', fontsize=12)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))






# ax1=fig.add_subplot(1,2,2)
#
# for i in range(len(param_f_por)):
#     plt.plot(param_f_dx_actual,nu[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
#     plt.plot(param_f_dx_actual,nu_apparent[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
#     plt.plot(param_f_dx_actual,(nu[i,:-1])/nu_apparent[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
#     plt.plot(param_f_dx_actual,(nu_apparent[i,:-1])/nu[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
#     plt.plot(param_f_dx_actual,conv_to_lat[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
# #plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
# #legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
# # plt.setp(legend.get_title(),fontsize=6)
# plt.ylim([0.0,7.0])
# plt.yticks(np.arange(0.0,7.0))
# ax1.grid(True,zorder=0)
# plt.xlabel('log$_{10}$($\phi_f$b$^{2}$)', fontsize=12)
# plt.ylabel('Nu', fontsize=12)
#
# x0,x1 = ax1.get_xlim()
# y0,y1 = ax1.get_ylim()
# ax1.set_aspect(abs(x1-x0)/abs(y1-y0))


plt.savefig(path+'pLot_apparent_alt.eps')






fig=plt.figure()

ax1=fig.add_subplot(1,2,1)

#for i in range(len(param_f_por)):
for i in [0]:
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu[i,:-1],color_string[0],label='nu_act',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu_apparent[i,:-1]-1.0,color_string[1],label='nu_app - 1',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    # plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),(nu[i,:-1])/nu_apparent[i,:-1],color_string[2],label='nu_act/nu_app',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    # plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),(nu_apparent[i,:-1])/nu[i,:-1],color_string[3],label='nu_app/nu_act',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    # plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),conv_to_lat[i,:-1],color_string[4],label='conv_to_lat',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
for i in [len(param_f_por)-1]:
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu[i,:-1],color_string[0],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu_apparent[i,:-1]-1.0,color_string[1],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    # plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),(nu[i,:-1])/nu_apparent[i,:-1],color_string[2],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    # plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),(nu_apparent[i,:-1])/nu[i,:-1],color_string[3],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    # plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),conv_to_lat[i,:-1],color_string[4],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
#plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
legend = plt.legend(fontsize=6,loc='best',ncol=2)
plt.setp(legend.get_title(),fontsize=3)
plt.ylim([0.5,6.0])
plt.yticks(np.arange(1.0,6.0))
plt.xlim([-11.5,-5.5])
plt.xticks(np.arange(-11.0,-5.0))
ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$($\phi_f$b$^{3}$)', fontsize=12)
plt.ylabel('Nu', fontsize=12)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))


ax1=fig.add_subplot(1,2,2)

# for i in range(len(param_f_por)):
for i in [0,len(param_f_por)-1]:
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),np.log10(u_mean_raw[i,:-1]),color_string[0],label=param_f_por_string2[0],marker='o',mec='None',ms=4.0,zorder=3,lw=0.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),np.log10(u_1d[i,:-1]),color_string[0],marker='o',mec=color_string[0],ms=0.0,zorder=2,lw=2.0)
#plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
plt.setp(legend.get_title(),fontsize=6)
plt.ylim([-2.5,3.0])
plt.yticks(np.arange(-2.0,4.0))
plt.xlim([-11.5,-5.5])
plt.xticks(np.arange(-11.0,-5.0))
ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$($\phi_f$b$^{3}$)', fontsize=12)
plt.ylabel('u_lateral', fontsize=12)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))


plt.savefig(path+'pLot_apparent.eps')









fig=plt.figure()

ax1=fig.add_subplot(1,2,1)

#for i in range(len(param_f_por)):
for i in [0]:
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu[i,:-1],color_string[0],label='nu_act',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu_apparent2[i,:-1],color_string[1],label='nu_app2',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),(nu[i,:-1])/nu_apparent2[i,:-1],color_string[2],label='nu_act/nu_app2',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),(nu_apparent2[i,:-1])/nu[i,:-1],color_string[3],label='nu_app/nu_act2',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),conv_to_lat2[i,:-1],color_string[4],label='conv_to_lat2',marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
for i in [len(param_f_por)-1]:
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu[i,:-1],color_string[0],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),nu_apparent2[i,:-1],color_string[1],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),(nu[i,:-1])/nu_apparent2[i,:-1],color_string[2],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),(nu_apparent2[i,:-1])/nu[i,:-1],color_string[3],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),conv_to_lat2[i,:-1],color_string[4],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
#plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
legend = plt.legend(fontsize=6,loc='best',ncol=3)
plt.setp(legend.get_title(),fontsize=6)
plt.ylim([0.0,11.0])
plt.yticks(np.arange(0.0,12.0))
plt.xlim([-11.5,-6.5])
plt.xticks(np.arange(-11.0,-5.0))
ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$($\phi_f$b$^{3}$)', fontsize=12)
plt.ylabel('Nu', fontsize=12)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))


ax1=fig.add_subplot(1,2,2)

# for i in range(len(param_f_por)):
for i in [0,len(param_f_por)-1]:
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),np.log10(u_mean_raw[i,:-1]),color_string[0],label=param_f_por_string2[0],marker='o',mec='None',ms=4.0,zorder=3,lw=0.0,ls='-')
    plt.plot(np.log10(param_f_dx_actual_cubed*(param_f_por_actual[i])),np.log10(u_1d[i,:-1]),color_string[0],marker='o',mec=color_string[0],ms=0.0,zorder=2,lw=2.0)
#plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
plt.setp(legend.get_title(),fontsize=6)
plt.ylim([-2.5,2.5])
plt.yticks(np.arange(-2.0,3.0))
ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$($\phi_f$b$^{3}$)', fontsize=12)
plt.ylabel('u_lateral', fontsize=12)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))


plt.savefig(path+'pLot_apparent2.eps')

#
# fig=plt.figure()
#
# ax1=fig.add_subplot(1,2,1)
#
# for i in range(len(param_f_por)):
#     plt.plot(param_f_dx_actual,nu_apparent[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=6.0,zorder=3,lw=1.0,ls='-')
# plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
# legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
# plt.setp(legend.get_title(),fontsize=6)
# ax1.grid(True,zorder=0)
# plt.xlabel('log$_{10}$(b)', fontsize=8)
# plt.ylabel('Nu', fontsize=8)
#
# x0,x1 = ax1.get_xlim()
# y0,y1 = ax1.get_ylim()
# ax1.set_aspect(abs(x1-x0)/abs(y1-y0))
#
#
#
#
# ax1=fig.add_subplot(1,2,2)
#
# for i in range(len(param_f_por)):
#     plt.plot(np.log10(param_f_dx_actual_squared*param_f_por_actual[i]),nu_apparent[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=6.0,zorder=3,lw=1.0,ls='-')
# #plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
# #legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
# # plt.setp(legend.get_title(),fontsize=6)
# ax1.grid(True,zorder=0)
# plt.xlabel('log$_{10}$(b)', fontsize=8)
# plt.ylabel('Nu', fontsize=8)
#
# x0,x1 = ax1.get_xlim()
# y0,y1 = ax1.get_ylim()
# ax1.set_aspect(abs(x1-x0)/abs(y1-y0))
#
#
# plt.subplots_adjust(wspace=0.3,hspace=0.0)
#
# plt.savefig(path+'pLot_apparent.eps')
#
#
#
#
#
#
#
#
# fig=plt.figure()
#
# ax1=fig.add_subplot(1,2,1)
#
# for i in range(len(param_f_por)):
#     plt.plot(param_f_dx_actual,nu_apparent2[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=6.0,zorder=3,lw=1.0,ls='-')
# plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
# legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
# plt.setp(legend.get_title(),fontsize=6)
# ax1.grid(True,zorder=0)
# plt.xlabel('log$_{10}$(b)', fontsize=8)
# plt.ylabel('Nu', fontsize=8)
#
# x0,x1 = ax1.get_xlim()
# y0,y1 = ax1.get_ylim()
# ax1.set_aspect(abs(x1-x0)/abs(y1-y0))
#
#
#
#
# ax1=fig.add_subplot(1,2,2)
#
# for i in range(len(param_f_por)):
#     plt.plot(np.log10(param_f_dx_actual_squared*param_f_por_actual[i]),nu_apparent2[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=6.0,zorder=3,lw=1.0,ls='-')
# #plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
# #legend = plt.legend(fontsize=6,loc='best',ncol=2,title='fracture volume fraction')
# # plt.setp(legend.get_title(),fontsize=6)
# ax1.grid(True,zorder=0)
# plt.xlabel('log$_{10}$(b)', fontsize=8)
# plt.ylabel('Nu', fontsize=8)
#
# x0,x1 = ax1.get_xlim()
# y0,y1 = ax1.get_ylim()
# ax1.set_aspect(abs(x1-x0)/abs(y1-y0))
#
#
# plt.subplots_adjust(wspace=0.3,hspace=0.0)
#
# plt.savefig(path+'pLot_apparent2.eps')


######################
##    THIRD PLOT    ##
######################



fig=plt.figure()

ax1=fig.add_subplot(1,1,1)

for i in range(0,len(param_f_por),1):
    plt.plot(param_f_dx_actual[:],np.log10(u_mean_raw[i,:-1]),color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=6.0,zorder=3,lw=0.0,ls='-')
    plt.plot(param_f_dx_actual[:],np.log10(u_1d[i,:-1]),color_string[i],marker='o',mec=color_string[i],ms=0.0,zorder=2,lw=2.0)
plt.xlim([param_f_dx_actual[0],param_f_dx_actual[-1]])
#plt.ylim(-2.0,3.0)

ax1.grid(True,zorder=0)
plt.xlabel('log$_{10}$(b)', fontsize=8)
plt.ylabel('log$_{10}$(u$_{mean}$)', fontsize=8)
plt.title(param_f_por_string2[i])
plt.legend(loc='best',fontsize=7,ncol=2)

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect(abs(x1-x0)/abs(y1-y0))


# ax1=fig.add_subplot(2,2,2)
#
# i = 0
# plt.plot(param_f_dx_actual[:],u_mean_raw[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
# plt.plot(param_f_dx_actual[:],u_1d[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=0.0,zorder=2,lw=2.0)
# plt.xlim([param_f_dx_actual[1],param_f_dx_actual[-1]])
#
# ax1.grid(True,zorder=0)
# plt.xlabel('log$_{10}$(b)', fontsize=8)
# plt.ylabel('u$_{mean}$', fontsize=8)
# plt.title(param_f_por_string2[i])
#
# x0,x1 = ax1.get_xlim()
# y0,y1 = ax1.get_ylim()
# ax1.set_aspect(abs(x1-x0)/abs(y1-y0))



#
# ax1=fig.add_subplot(2,2,3)
#
# i = 4
# plt.plot(param_f_dx_actual[:],np.log10(u_mean_raw[i,:-1]),color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=0.0,ls='-')
# plt.plot(param_f_dx_actual[:],np.log10(u_1d[i,:-1]),'w',label=param_f_por_string2[i],marker='o',mec=color_string[i],ms=10.0,zorder=2,lw=0.0)
# plt.xlim([param_f_dx_actual[1],param_f_dx_actual[-1]])
#
# ax1.grid(True,zorder=0)
# plt.xlabel('log$_{10}$(b)', fontsize=8)
# plt.ylabel('log$_{10}$(u$_{mean}$)', fontsize=8)
# plt.title(param_f_por_string2[i])
#
# x0,x1 = ax1.get_xlim()
# y0,y1 = ax1.get_ylim()
# ax1.set_aspect(abs(x1-x0)/abs(y1-y0))


# ax1=fig.add_subplot(2,2,4)
#
# i = 4
# plt.plot(param_f_dx_actual[:],u_mean_raw[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=4.0,zorder=3,lw=1.0,ls='-')
# plt.plot(param_f_dx_actual[:],u_1d[i,:-1],color_string[i],label=param_f_por_string2[i],marker='o',mec='None',ms=0.0,zorder=2,lw=2.0)
# plt.xlim([param_f_dx_actual[1],param_f_dx_actual[-1]])
#
# ax1.grid(True,zorder=0)
# plt.xlabel('log$_{10}$(b)', fontsize=8)
# plt.ylabel('u$_{mean}$', fontsize=8)
# plt.title(param_f_por_string2[i])
#
# x0,x1 = ax1.get_xlim()
# y0,y1 = ax1.get_ylim()
# ax1.set_aspect(abs(x1-x0)/abs(y1-y0))


plt.subplots_adjust(wspace=0.2,hspace=0.3)

plt.savefig(path+'pIndiv.eps')






########################
##    CONTOUR PLOT    ##
########################

asp_multi_c = (float(x_param[-2])-float(x_param[0]))/(float(y_param[-2])-float(y_param[0]))
xx, yy = np.meshgrid(x_param[:-1], y_param[:-1])

fig=plt.figure()



asp_multi = (float(x_param[-1])-float(x_param[0]))/(float(y_param[-1])-float(y_param[0]))

u_mean_ts_mask_contour = copy.deepcopy(u_mean_ts_mask)



# ax1=fig.add_subplot(1,2,1, aspect=asp_multi)
ax1=fig.add_subplot(1,3,1, aspect=asp_multi_c)


vv = np.linspace(-1.8, 2.5, 20, endpoint=True)
pColc = plt.contourf(xx, yy, np.log10(u_mean_ts_mask_contour[:-1,:-1]), vv, cmap=cm.YlOrBr)
for c in pColc.collections:
    c.set_edgecolor("face")
pCol = plt.contour(xx, yy, u_mean_ts_mask_contour[:-1,:-1], [1.0, 10.0, 40.0], colors='w',linestyles=':')
pCol = plt.contour(xx, yy, dtdx_top[:-1,:-1], [1.816,], colors='w',linestyles='dashed')
pCol = plt.contour(xx, yy, 100.0*hf_range[:-1,:-1], [20.0], colors='w')
#plt.clabel(pCol, fontsize=5, fmt = '%1.0f', inline=1)
plt.xticks(x_param[::2])
plt.yticks(y_param[1::2])
plt.xlim([np.min(x_param), np.max(x_param[:-1])])
plt.ylim([np.min(y_param), np.max(y_param[:-1])])
# plt.xlabel('fracture width b [m]', fontsize=10)
# plt.ylabel('fracture spacing [m]', fontsize=10)
ax1.set_xticklabels(param_f_dx_string[::2])
ax1.set_yticklabels(param_f_por_string[1::2])
# plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
cbar = plt.colorbar(pColc, orientation='vertical',fraction=0.046, pad=0.04, ticks = [-1.5, -1.0, -0.5, -0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
# ticks = [-2.0, -1.5, -1.0, -0.5, -0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax1=fig.add_subplot(1,3,2, aspect=asp_multi_c)


hf_range_contour = copy.deepcopy(hf_range)
# hf_range_contour[hf_range_contour>80.0] = 80.0

# hf_range_contour[1:,1:] = hf_range[:-1,:-1]
# hf_range_contour[0,1:] = hf_range[0,:-1]
# hf_range_contour[1:,0] = hf_range[:-1,0]
print "hf_range contour"
print hf_range_contour
#
# hf_range[:,-1] = hf_range[:,-2]
# hf_range[-1,:] = hf_range[-2,:]
#


q_adv = rho_h*cp_h*(f_temp-275.0)*u_mean_ts_mask*por_mat

#dtdx_top[dtdx_top<0.0] = 0.0

vv = np.linspace(0, 2.75, 20, endpoint=True)
pColc = plt.contourf(xx, yy, dtdx_top[:-1,:-1], 20, cmap=cm.YlGn)
for c in pColc.collections:
    c.set_edgecolor("face")
pCol = plt.contour(xx, yy, u_mean_ts_mask_contour[:-1,:-1], [1.0, 10.0, 40.0], colors='w',linestyles=':')
pCol = plt.contour(xx, yy, dtdx_top[:-1,:-1], [1.816,], colors='w',linestyles='dashed')
pCol = plt.contour(xx, yy, 100.0*hf_range[:-1,:-1], [20.0], colors='w')
# plt.clabel(pCol, fontsize=5, fmt = '%1.2f', inline=1)
#pCol = plt.contour(xx, yy, 100.0*hf_range[:-1,:-1], [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0], colors='k')
# plt.clabel(pCol, fontsize=5, fmt = '%1.0f', inline=1)
plt.xticks(x_param[::2])
plt.yticks(y_param[1::2])
plt.xlim([np.min(x_param), np.max(x_param[:-1])])
plt.ylim([np.min(y_param), np.max(y_param[:-1])])
ax1.set_xticklabels(param_f_dx_string[::2])
ax1.set_yticklabels(param_f_por_string[1::2])
cbar = plt.colorbar(pColc, orientation='vertical',fraction=0.046, pad=0.04)
#, ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax1=fig.add_subplot(1,3,3, aspect=asp_multi_c)


vv = np.linspace(0.0, 100.0, 20, endpoint=True)
pColc = plt.contourf(xx, yy, 100.0*hf_range[:-1,:-1], vv, cmap=cm.YlGnBu)
for c in pColc.collections:
    c.set_edgecolor("face")
pCol = plt.contour(xx, yy, u_mean_ts_mask_contour[:-1,:-1], [1.0, 10.0, 40.0], colors='w',linestyles=':')
pCol = plt.contour(xx, yy, dtdx_top[:-1,:-1], [1.816,], colors='w',linestyles='dashed')


pCol = plt.contour(xx, yy, 100.0*hf_range[:-1,:-1], [20.0], colors='w')
# plt.clabel(pCol, fontsize=5, fmt = '%1.0f', inline=1)
plt.xticks(x_param[::2])
plt.yticks(y_param[1::2])
plt.xlim([np.min(x_param), np.max(x_param[:-1])])
plt.ylim([np.min(y_param), np.max(y_param[:-1])])
ax1.set_xticklabels(param_f_dx_string[::2])
ax1.set_yticklabels(param_f_por_string[1::2])
cbar = plt.colorbar(pColc, orientation='vertical',fraction=0.046, pad=0.04, ticks=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")



# ax1=fig.add_subplot(1,3,3, aspect=asp_multi_c)
#
# # nu_contour = np.zeros([nu.shape])
# nu_contour = copy.deepcopy(nu)
#
# print "nu contour"
# print nu_contour
#
# print "h_metric2"
# print h_metric2
#
# vv = np.linspace(-1.0, 4.5, 100, endpoint=True)
# pColc = plt.contourf(xx, yy, np.log10(h_metric2[:-1,:-1]), vv, cmap=cm.hot_r)
# for c in pColc.collections:
#     c.set_edgecolor("face")
# pCol = plt.contour(xx, yy, hf_range[:-1,:-1], [0.2, 0.3, 0.4, 0.5], colors='w')
# plt.clabel(pCol, fontsize=6, inline=1)
# plt.xticks(x_param[::2])
# plt.yticks(y_param[::2])
# plt.xlim([np.min(x_param), np.max(x_param[:-1])])
# plt.ylim([np.min(y_param), np.max(y_param[:-1])])
# plt.xlabel('fracture width b [m]', fontsize=10)
# plt.ylabel('fracture spacing [m]', fontsize=10)
# ax1.set_xticklabels(param_f_dx_string[::2])
# ax1.set_yticklabels(param_f_por_string[::2])
# plt.title('heat metric 2', fontsize=10)
# cbar = plt.colorbar(pColc, orientation='vertical',fraction=0.046, pad=0.04, ticks = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
# cbar.solids.set_rasterized(True)
# cbar.solids.set_edgecolor("face")

fig.set_tight_layout(True)
plt.subplots_adjust(hspace=0.3)



#
# ax1=fig.add_subplot(2,2,3, aspect=asp_multi_c)
#
#
# pColc = plt.contourf(xx, yy, q_adv2[:-1,:-1], 40, cmap=cm.rainbow)
# for c in pColc.collections:
#     c.set_edgecolor("face")
# pCol = plt.contour(xx, yy, hf_range[:-1,:-1], [0.2, 0.3, 0.4, 0.5], colors='k')
# plt.clabel(pCol, fontsize=5, inline=1)
# plt.xticks(x_param[::2])
# plt.yticks(y_param[::2])
# plt.xlim([np.min(x_param), np.max(x_param[:-1])])
# plt.ylim([np.min(y_param), np.max(y_param[:-1])])
# plt.xlabel('fracture width b [m]', fontsize=10)
# plt.ylabel('fracture spacing [m]', fontsize=10)
# ax1.set_xticklabels(param_f_dx_string[::2])
# ax1.set_yticklabels(param_f_por_string[::2])
# plt.title('advective heat flow 2 [W/m^2]', fontsize=10)
# cbar = plt.colorbar(pColc, orientation='vertical',fraction=0.046, pad=0.04)
# cbar.solids.set_rasterized(True)
# cbar.solids.set_edgecolor("face")
#






# st = fig.suptitle(str(param_h))
plt.savefig(path+'pCont.eps')

