#coarse_multi.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import streamplot as sp
import multiplot_data as mpd
import heapq
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import ternary
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rcParams['axes.titlesize'] = 10
#plt.rcParams['hatch.linewidth'] = 0.1

plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #31aa22, #04776b"
#hack: colors
col = ['#880000', '#ff0000', '#ff7411', '#bddb00', '#159600', '#00ffc2', '#0000ff', '#2f3699','#8f00ff', '#ec52ff', '#6e6e6e', '#000000', '#c6813a', '#7d4e22', '#ffff00', '#df9a00', '#812700', '#6b3f67', '#0f9995', '#4d4d4d', '#00530d', '#d9d9d9', '#e9acff']

plot_col = ['#940000', '#cf5448', '#fc8800', '#2ab407', '#6aabf7', '#bb43e6']

secondary = np.array(['', 'kaolinite', 'saponite_mg', 'celadonite', 'clinoptilolite', 'pyrite', 'mont_na', 'goethite',
'smectite', 'calcite', 'kspar', 'saponite_na', 'nont_na', 'nont_mg', 'fe_celad', 'nont_ca',
'mesolite', 'hematite', 'mont_ca', 'verm_ca', 'analcime', 'philipsite', 'mont_mg', 'gismondine',
'verm_mg', 'natrolite', 'talc', 'smectite_low', 'prehnite', 'chlorite', 'scolecite', 'clinochlorte14a',
'clinochlore7a', 'saponite_ca', 'verm_na', 'pyrrhotite', 'fe_saponite_ca', 'fe_saponite_mg', 'daphnite7a', 'daphnite14a', 'epidote'])

primary = np.array(['', '', 'plagioclase', 'pyroxene', 'olivine', 'basaltic glass'])

density = np.array([0.0, 2.65, 2.3, 3.05, 2.17, 5.01, 2.5, 3.8,
2.7, 2.71, 2.56, 2.3, 2.28, 2.28, 3.05, 2.28,
2.25, 5.3, 2.5, 2.55, 2.27, 2.2, 2.5, 2.26,
2.55, 2.25, 2.75, 2.7, 2.87, 2.9, 2.275, 2.8,
2.8, 2.3, 2.55, 4.61, 2.3, 2.3, 3.2, 3.2, 3.45])

molar = np.array([0.0, 258.156, 480.19, 429.02, 2742.13, 119.98, 549.07, 88.851,
549.07, 100.0869, 287.327, 480.19, 495.90, 495.90, 429.02, 495.90,
380.22, 159.6882, 549.07, 504.19, 220.15, 649.86, 549.07, 649.86,
504.19, 380.22, 379.259, 549.07, 395.38, 64.448, 392.34, 64.448,
64.448, 480.19, 504.19, 85.12, 480.19, 480.19, 664.0, 664.0, 519.0])

molar_pri = np.array([277.0, 153.0, 158.81, 110.0])

density_pri = np.array([2.7, 3.0, 3.0, 3.0])

#hack: path stuff
prefix_string = "s_75A_25B_"
suffix_string = "/"
batch_path = "../output/revival/summer_coarse_grid/"
batch_path_ex = "../output/revival/summer_coarse_grid/"+prefix_string+"8e10/"


#hack: param_t_diff listed here
# param_t_diff = np.array([10e10, 5e10, 1e10])
# param_t_diff_string = ['10e10' , '5e10' , '1e10']


# param_t_diff = np.array([8e10, 6e10, 4e10, 2e10, 1e10])
# param_t_diff_string = ['8e10' , '6e10' , '4e10', '2e10', '1e10']
#
# plot_strings = ['8e10', '6e10', '4e10', '2e10', '1e10', 'solo']

# param_t_diff = np.array([10e10, 8e10, 6e10, 4e10, 2e10])
# param_t_diff_string = ['10e10', '8e10' , '6e10' , '4e10', '2e10']
# plot_strings = ['10e10 (least mixing)', '8e10', '6e10', '4e10', '2e10 (most mixing)', 'solo']

param_t_diff = np.array([8e10, 6e10, 4e10, 2e10])
param_t_diff_string = ['8e10' , '6e10' , '4e10', '2e10']
plot_strings = ['8e10 (least mixing ish)', '6e10', '4e10', '2e10 (most mixing)', 'solo']



x0 = np.loadtxt(batch_path_ex + 'x.txt',delimiter='\n')
y0 = np.loadtxt(batch_path_ex + 'y.txt',delimiter='\n')

#hack: params here
cellx = 90
celly = 1
steps = 50
minNum = 41
# even number
max_step = 49
final_index = 4
restart=1

xCell = x0[1::cellx]
yCell = y0[0::celly]

xgCell, ygCell = np.meshgrid(xCell[:],yCell[:])


def cut_chem(geo0,index):
    geo_cut_chem = geo0[:,(index*len(xCell)):(index*len(xCell)+len(xCell))]
    return geo_cut_chem

def chemplot(varMat, varStep, sp1, sp2, sp3, contour_interval, cp_title, xtix=1, ytix=1, cb=1, cb_title='', cb_min=-10.0, cb_max=10.0):
    varStep[np.isinf(varStep)] = 2.0
    varStep[np.isnan(varStep)] = 2.0
    varMat[np.isinf(varMat)] = 2.0
    varMat[np.isnan(varMat)] = 2.0
    varMat[varMat>200.0] = 1.15
    varMat[varMat<-200.0] = -1.15
    varStep[varStep>200.0] = 1.15
    varStep[varStep<-200.0] = -1.15
    if np.any(varMat) == 2.0:
        cb_max=2.0
        cb_min=-2.0
    if np.abs(np.abs(np.max(varStep)) - np.abs(np.min(varStep))) <= 0.0:
        if cb_min==-10.0 and cb_max==10.0:
            contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),5)
        if cb_max!=10.0:
            contours = np.linspace(cb_min,cb_max,5)

        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4,frameon=False)
        pGlass = plt.pcolor(xCell,yCell,np.zeros(varStep.shape),cmap=cm.rainbow,vmin=contours[0], vmax=contours[-1])
        plt.yticks([])
        if ytix==1:
            plt.yticks([-450, -400, -350])
        if xtix==0:
            plt.xticks([])
        plt.ylim([np.min(yCell),0.])
        plt.title(cp_title,fontsize=8)
        plt.ylim([-510.0,-320.0])
        if cb==1:
            bbox = ax1.get_position()
            cax = fig.add_axes([bbox.xmin+bbox.width/10.0, bbox.ymin-0.28, bbox.width*0.8, bbox.height*0.13])
            cbar = plt.colorbar(pGlass, cax = cax,orientation='horizontal',ticks=contours[::contour_interval])
            plt.title(cb_title,fontsize=10)
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")

    if np.abs(np.abs(np.max(varStep)) - np.abs(np.min(varStep))) > 0.0:
        if cb_min==-10.0 and cb_max==10.0:
            contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),5)
        if cb_max!=10.0:
            contours = np.linspace(cb_min,cb_max,5)
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4,frameon=False)

        xCell_t = []
        xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
        varStep_t = np.zeros([len(yCell),len(xCell)+1])
        varStep_t[:,:len(xCell)] = varStep


        pGlass = plt.pcolor(xCell_t,yCell,varStep_t,cmap=cm.rainbow,vmin=contours[0], vmax=contours[-1])

        plt.yticks([])
        if ytix==1:
            plt.yticks([-450, -400, -350])
        if xtix==0:
            plt.xticks([])
        plt.ylim([np.min(yCell),0.])
        plt.title(cp_title,fontsize=8)
        plt.ylim([-510.0,-320.0])

        pGlass.set_edgecolor("face")
        if cb==1:
            bbox = ax1.get_position()
            cax = fig.add_axes([bbox.xmin+bbox.width/10.0, bbox.ymin-0.28, bbox.width*0.8, bbox.height*0.13])
            cbar = plt.colorbar(pGlass, cax = cax,orientation='horizontal',ticks=contours[::contour_interval])
            plt.title(cb_title,fontsize=10)
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")
    return chemplot





alt_vol_curves = np.zeros([len(xCell),steps,len(param_t_diff_string) + 1])



#hack: init big arrays
secMat = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string)])
secMat_a = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string)])
secMat_b = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string)])
secMat_d = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string)])

secStep = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])
secStep_a = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])
secStep_b = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])
secStep_d = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])

secStep_last = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])
secStep_last_a = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])
secStep_last_b = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])
secStep_last_d = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])

dsecMat = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string)])
dsecMat_a = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string)])
dsecMat_b = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string)])
dsecMat_d = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string)])

dsecStep = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])
dsecStep_a = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])
dsecStep_b = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])
dsecStep_d = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string)])

secStep_ts = np.zeros([steps,minNum+1,len(param_t_diff_string)])
secStep_ts_a = np.zeros([steps,minNum+1,len(param_t_diff_string)])
secStep_ts_b = np.zeros([steps,minNum+1,len(param_t_diff_string)])
secStep_ts_d = np.zeros([steps,minNum+1,len(param_t_diff_string)])

dsecStep_ts = np.zeros([steps,minNum+1,len(param_t_diff_string) ])
dsecStep_ts_a = np.zeros([steps,minNum+1,len(param_t_diff_string)])
dsecStep_ts_b = np.zeros([steps,minNum+1,len(param_t_diff_string)])
dsecStep_ts_d = np.zeros([steps,minNum+1,len(param_t_diff_string)])

x_secStep_ts = np.zeros([steps,minNum+1,len(param_t_diff_string)])
x_secStep_ts_a = np.zeros([steps,minNum+1,len(param_t_diff_string)])
x_secStep_ts_b = np.zeros([steps,minNum+1,len(param_t_diff_string)])
x_secStep_ts_d = np.zeros([steps,minNum+1,len(param_t_diff_string)])

x_dsecStep_ts = np.zeros([steps,minNum+1,len(param_t_diff_string)])
x_dsecStep_ts_a = np.zeros([steps,minNum+1,len(param_t_diff_string)])
x_dsecStep_ts_b = np.zeros([steps,minNum+1,len(param_t_diff_string)])
x_dsecStep_ts_d = np.zeros([steps,minNum+1,len(param_t_diff_string)])

sec_binary = np.zeros([len(xCell),steps,minNum+1,len(param_t_diff_string)])
sec_binary_a = np.zeros([len(xCell),steps,minNum+1,len(param_t_diff_string)])
sec_binary_b = np.zeros([len(xCell),steps,minNum+1,len(param_t_diff_string)])
sec_binary_d = np.zeros([len(xCell),steps,minNum+1,len(param_t_diff_string)])
sec_binary_any = np.zeros([len(xCell),steps,minNum+1,len(param_t_diff_string)])

x_d = -5
xd_move = 0
moves = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40, 42])


# new 12/27/17

priMat = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])
priMat_a = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])
priMat_b = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])
priMat_d = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])

priStep = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
priStep_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
priStep_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
priStep_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])

dpriMat = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])
dpriMat_a = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])
dpriMat_b = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])
dpriMat_d = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])

dpriStep = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
dpriStep_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
dpriStep_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
dpriStep_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])

x_priStep_ts = np.zeros([steps,len(param_t_diff_string)])
x_priStep_ts_a = np.zeros([steps,len(param_t_diff_string)])
x_priStep_ts_b = np.zeros([steps,len(param_t_diff_string)])
x_priStep_ts_d = np.zeros([steps,len(param_t_diff_string)])

x_dpriStep_ts = np.zeros([steps,len(param_t_diff_string)])
x_dpriStep_ts_a = np.zeros([steps,len(param_t_diff_string)])
x_dpriStep_ts_b = np.zeros([steps,len(param_t_diff_string)])
x_dpriStep_ts_d = np.zeros([steps,len(param_t_diff_string)])

#end new



# changed last argument from 6

priStep_ts = np.zeros([steps,len(param_t_diff_string)])
priStep_ts_a = np.zeros([steps,len(param_t_diff_string)])
priStep_ts_b = np.zeros([steps,len(param_t_diff_string)])
priStep_ts_d = np.zeros([steps,len(param_t_diff_string)])

dpriStep_ts = np.zeros([steps,len(param_t_diff_string)])
dpriStep_ts_a = np.zeros([steps,len(param_t_diff_string)])
dpriStep_ts_b = np.zeros([steps,len(param_t_diff_string)])
dpriStep_ts_d = np.zeros([steps,len(param_t_diff_string)])

# end changed


alt_vol0 = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])
alt_vol0_a = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])
alt_vol0_b = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])
alt_vol0_d = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string)])

alt_vol = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
alt_vol_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
alt_vol_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
alt_vol_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])

alt_col_mean = np.zeros([len(xCell),steps,len(param_t_diff_string)+1])
alt_col_mean_top_half = np.zeros([len(xCell),steps,len(param_t_diff_string)+1])
alt_col_mean_top_cell = np.zeros([len(xCell),steps,len(param_t_diff_string)+1])

fe_col_mean = np.zeros([len(xCell),steps,len(param_t_diff_string)+1])
fe_col_mean_top_half = np.zeros([len(xCell),steps,len(param_t_diff_string)+1])
fe_col_mean_top_cell = np.zeros([len(xCell),steps,len(param_t_diff_string)+1])


ternK = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
ternK_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
ternK_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
ternK_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])

ternMg = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
ternMg_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
ternMg_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
ternMg_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])

ternFe = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
ternFe_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
ternFe_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])
ternFe_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string)])

tern_list = np.zeros([len(yCell)*len(xCell),steps,3,len(param_t_diff_string)])
tern_list_a = np.zeros([len(yCell)*len(xCell),steps,3,len(param_t_diff_string)])
tern_list_b = np.zeros([len(yCell)*len(xCell),steps,3,len(param_t_diff_string)])
tern_list_d = np.zeros([len(yCell)*len(xCell),steps,3,len(param_t_diff_string)])

#todo: loop through param_t_diff
for ii in range(len(param_t_diff)):
    print " "
    print " "
    print "param:" , param_t_diff_string[ii]

    # ii_path goes here
    ii_path = batch_path + prefix_string + param_t_diff_string[ii] + suffix_string
    print "ii_path: " , ii_path



    any_min = []

    #hack: load in chem data
    if ii == 0:

        ch_path = ii_path + 'ch_s/'
        for j in range(1,minNum):
            if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
                if not np.any(any_min == j):
                    any_min = np.append(any_min,j)
                print j , secondary[j] ,
                secMat[:,:,j,ii] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
                secMat[:,:,j,ii] = secMat[:,:,j,ii]*molar[j]/density[j]
                dsecMat[:,2*len(xCell):,j,ii] = secMat[:,len(xCell):-len(xCell),j,ii] - secMat[:,2*len(xCell):,j,ii]
        dic0 = np.loadtxt(ch_path + 'z_sol_c.txt')
        ca0 = np.loadtxt(ch_path + 'z_sol_ca.txt')
        mg0 = np.loadtxt(ch_path + 'z_sol_mg.txt')
        na0 = np.loadtxt(ch_path + 'z_sol_na.txt')
        cl0 = np.loadtxt(ch_path + 'z_sol_cl.txt')
        k0 = np.loadtxt(ch_path + 'z_sol_k.txt')
        fe0 = np.loadtxt(ch_path + 'z_sol_fe.txt')
        si0 = np.loadtxt(ch_path + 'z_sol_si.txt')
        al0 = np.loadtxt(ch_path + 'z_sol_al.txt')
        ph0 = np.loadtxt(ch_path + 'z_sol_ph.txt')
        alk0 = np.loadtxt(ch_path + 'z_sol_alk.txt')
        solw0 = np.loadtxt(ch_path + 'z_sol_w.txt')
        glass0 = np.loadtxt(ch_path + 'z_pri_glass.txt')*molar_pri[3]/density_pri[3]
        priMat[:,:,ii] = glass0
        dpriMat[:,2*len(xCell):,ii] = priMat[:,len(xCell):-len(xCell),ii] - priMat[:,2*len(xCell):,ii]

        pri_total0 = glass0
        print " "

    if ii > 0:
        secMat[:,:,:,ii] = secMat[:,:,:,ii-1]


    ch_path = ii_path + 'ch_a/'
    for j in range(1,minNum):
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            print j , secondary[j] ,
            secMat_a[:,:,j,ii] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat_a[:,:,j,ii] = secMat_a[:,:,j,ii]*molar[j]/density[j]
            dsecMat_a[:,2*len(xCell):,j,ii] = secMat_a[:,len(xCell):-len(xCell),j,ii] - secMat_a[:,2*len(xCell):,j,ii]
    dic0_a = np.loadtxt(ch_path + 'z_sol_c.txt')
    ca0_a = np.loadtxt(ch_path + 'z_sol_ca.txt')
    mg0_a = np.loadtxt(ch_path + 'z_sol_mg.txt')
    na0_a = np.loadtxt(ch_path + 'z_sol_na.txt')
    cl0_a = np.loadtxt(ch_path + 'z_sol_cl.txt')
    k0_a = np.loadtxt(ch_path + 'z_sol_k.txt')
    fe0_a = np.loadtxt(ch_path + 'z_sol_fe.txt')
    si0_a = np.loadtxt(ch_path + 'z_sol_si.txt')
    al0_a = np.loadtxt(ch_path + 'z_sol_al.txt')
    ph0_a = np.loadtxt(ch_path + 'z_sol_ph.txt')
    alk0_a = np.loadtxt(ch_path + 'z_sol_alk.txt')
    solw0_a = np.loadtxt(ch_path + 'z_sol_w.txt')
    glass0_a = np.loadtxt(ch_path + 'z_pri_glass.txt')*molar_pri[3]/density_pri[3]
    priMat_a[:,:,ii] = glass0_a
    dpriMat_a[:,2*len(xCell):,ii] = priMat_a[:,len(xCell):-len(xCell),ii] - priMat_a[:,2*len(xCell):,ii]

    pri_total0_a = glass0_a
    print " "


    ch_path = ii_path + 'ch_b/'
    for j in range(1,minNum):
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            print j , secondary[j] ,
            secMat_b[:,:,j,ii] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat_b[:,:,j,ii] = secMat_b[:,:,j,ii]*molar[j]/density[j]
            dsecMat_a[:,2*len(xCell):,j,ii] = secMat_b[:,len(xCell):-len(xCell),j,ii] - secMat_b[:,2*len(xCell):,j,ii]
    dic0_b = np.loadtxt(ch_path + 'z_sol_c.txt')
    ca0_b = np.loadtxt(ch_path + 'z_sol_ca.txt')
    mg0_b = np.loadtxt(ch_path + 'z_sol_mg.txt')
    na0_b = np.loadtxt(ch_path + 'z_sol_na.txt')
    cl0_b = np.loadtxt(ch_path + 'z_sol_cl.txt')
    k0_b = np.loadtxt(ch_path + 'z_sol_k.txt')
    fe0_b = np.loadtxt(ch_path + 'z_sol_fe.txt')
    si0_b = np.loadtxt(ch_path + 'z_sol_si.txt')
    al0_b = np.loadtxt(ch_path + 'z_sol_al.txt')
    ph0_b = np.loadtxt(ch_path + 'z_sol_ph.txt')
    alk0_b = np.loadtxt(ch_path + 'z_sol_alk.txt')
    solw0_b = np.loadtxt(ch_path + 'z_sol_w.txt')
    glass0_b = np.loadtxt(ch_path + 'z_pri_glass.txt')*molar_pri[3]/density_pri[3]
    priMat_b[:,:,ii] = glass0_b
    dpriMat_b[:,2*len(xCell):,ii] = priMat_b[:,len(xCell):-len(xCell),ii] - priMat_b[:,2*len(xCell):,ii]

    pri_total0_b = glass0_b
    print " "



    ch_path = ii_path + 'ch_d/'
    for j in range(1,minNum):
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            print j , secondary[j] ,
            secMat_d[:,:,j,ii] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat_d[:,:,j,ii] = secMat_d[:,:,j,ii]*molar[j]/density[j]
            dsecMat_d[:,2*len(xCell):,j,ii] = secMat_d[:,len(xCell):-len(xCell),j,ii] - secMat_d[:,2*len(xCell):,j,ii]
    dic0_d = np.loadtxt(ch_path + 'z_sol_c.txt')
    ca0_d = np.loadtxt(ch_path + 'z_sol_ca.txt')
    mg0_d = np.loadtxt(ch_path + 'z_sol_mg.txt')
    na0_d = np.loadtxt(ch_path + 'z_sol_na.txt')
    cl0_d = np.loadtxt(ch_path + 'z_sol_cl.txt')
    k0_d = np.loadtxt(ch_path + 'z_sol_k.txt')
    fe0_d = np.loadtxt(ch_path + 'z_sol_fe.txt')
    si0_d = np.loadtxt(ch_path + 'z_sol_si.txt')
    al0_d = np.loadtxt(ch_path + 'z_sol_al.txt')
    ph0_d = np.loadtxt(ch_path + 'z_sol_ph.txt')
    alk0_d = np.loadtxt(ch_path + 'z_sol_alk.txt')
    solw0_d = np.loadtxt(ch_path + 'z_sol_w.txt')
    glass0_d = np.loadtxt(ch_path + 'z_pri_glass.txt')*molar_pri[3]/density_pri[3]
    priMat_d[:,:,ii] = glass0_d
    dpriMat_d[:,2*len(xCell):,ii] = priMat_d[:,len(xCell):-len(xCell),ii] - priMat_d[:,2*len(xCell):,ii]

    pri_total0_d = glass0_d
    print " "




    for j in range(len(xCell)*steps):
        for k in range(len(yCell)):
            if pri_total0[k,j] > 0.0:
                alt_vol0[k,j,ii] = np.sum(secMat[k,j,:,ii])/(pri_total0[k,j]+np.sum(secMat[k,j,:,ii]))


    for j in range(len(xCell)*steps):
        for k in range(len(yCell)):
            if pri_total0_a[k,j] > 0.0:
                alt_vol0_a[k,j,ii] = np.sum(secMat_a[k,j,:,ii])/(pri_total0_a[k,j]+np.sum(secMat_a[k,j,:,ii]))


    for j in range(len(xCell)*steps):
        for k in range(len(yCell)):
            if pri_total0_b[k,j] > 0.0:
                alt_vol0_b[k,j,ii] = np.sum(secMat_b[k,j,:,ii])/(pri_total0_b[k,j]+np.sum(secMat_b[k,j,:,ii]))


    for j in range(len(xCell)*steps):
        for k in range(len(yCell)):
                if pri_total0_d[k,j] > 0.0:
                    alt_vol0_d[k,j,ii] = np.sum(secMat_d[k,j,:,ii])/(pri_total0_d[k,j]+np.sum(secMat_d[k,j,:,ii]))



    xd_move = 0
    #todo: loop through steps
    for i in range(0,max_step,1):

        if np.any(moves== i + restart):
            xd_move = xd_move + 1
        # print xd_move

        #hack: cut up chem data
        for j in range(len(any_min)):
            secStep[:,:,any_min[j],i,ii] = cut_chem(secMat[:,:,any_min[j],ii],i)
            dsecStep[:,:,any_min[j],i,ii] = cut_chem(dsecMat[:,:,any_min[j],ii],i)
            secStep_ts[i,any_min[j],ii] = np.sum(secStep[:,:,any_min[j],i,ii])
            x_secStep_ts[i,any_min[j],ii] = np.sum(secStep[:,xd_move,any_min[j],i,ii])
            if i > 0:
                dsecStep_ts[i,any_min[j],ii] = secStep_ts[i,any_min[j],ii] - secStep_ts[i-1,any_min[j],ii]
                x_dsecStep_ts[i,any_min[j],ii] = x_secStep_ts[i,any_min[j],ii] - x_secStep_ts[i-1,any_min[j],ii]
        dic = cut_chem(dic0,i)
        ca = cut_chem(ca0,i)
        ph = cut_chem(ph0,i)
        alk = cut_chem(alk0,i)
        solw = cut_chem(solw0,i)
        mg = cut_chem(mg0,i)
        fe = cut_chem(fe0,i)
        si = cut_chem(si0,i)
        k1 = cut_chem(k0,i)
        na = cut_chem(na0,i)
        cl = cut_chem(cl0,i)
        al = cut_chem(al0,i)
        glass = cut_chem(glass0,i)
        alt_vol[:,:,i,ii] = cut_chem(alt_vol0[:,:,ii],i)
        pri_total = cut_chem(pri_total0,i)

        priStep[:,:,i,ii] = cut_chem(priMat[:,:,ii],i)
        dpriStep[:,:,i,ii] = cut_chem(dpriMat[:,:,ii],i)
        priStep_ts[i,ii] = np.sum(priStep[:,:,i,ii])
        x_priStep_ts[i,ii] = np.sum(priStep[:,xd_move,i,ii])

        if i > 0:
            dpriStep_ts[i,ii] = priStep_ts[i,ii] - priStep_ts[i-1,ii]
            x_dpriStep_ts[i,ii] = x_priStep_ts[i,ii] - x_priStep_ts[i-1,ii]






        for j in range(len(any_min)):
            secStep_a[:,:,any_min[j],i,ii] = cut_chem(secMat_a[:,:,any_min[j],ii],i)
            dsecStep_a[:,:,any_min[j],i,ii] = cut_chem(dsecMat_a[:,:,any_min[j],ii],i)
            secStep_ts_a[i,any_min[j],ii] = np.sum(secStep_a[:,:,any_min[j],i,ii])
            x_secStep_ts_a[i,any_min[j],ii] = np.sum(secStep_a[:,xd_move,any_min[j],i,ii])
            if i > 0:
                dsecStep_ts_a[i,any_min[j],ii] = secStep_ts_a[i,any_min[j],ii] - secStep_ts_a[i-1,any_min[j],ii]
                x_dsecStep_ts_a[i,any_min[j],ii] = x_secStep_ts_a[i,any_min[j],ii] - x_secStep_ts_a[i-1,any_min[j],ii]
        dic_a = cut_chem(dic0,i)
        ca_a = cut_chem(ca0,i)
        ph_a = cut_chem(ph0,i)
        alk_a = cut_chem(alk0,i)
        solw_a = cut_chem(solw0,i)
        mg_a = cut_chem(mg0,i)
        fe_a = cut_chem(fe0,i)
        si_a = cut_chem(si0,i)
        k1_a = cut_chem(k0,i)
        na_a = cut_chem(na0,i)
        cl_a = cut_chem(cl0,i)
        al_a = cut_chem(al0,i)
        glass_a = cut_chem(glass0,i)
        alt_vol_a[:,:,i,ii] = cut_chem(alt_vol0_a[:,:,ii],i)
        pri_total_a = cut_chem(pri_total0,i)

        priStep_a[:,:,i,ii] = cut_chem(priMat_a[:,:,ii],i)
        dpriStep_a[:,:,i,ii] = cut_chem(dpriMat_a[:,:,ii],i)
        priStep_ts_a[i,ii] = np.sum(priStep_a[:,:,i,ii])
        x_priStep_ts_a[i,ii] = np.sum(priStep_a[:,xd_move,i,ii])
        if i > 0:
            dpriStep_ts_a[i,ii] = priStep_ts_a[i,ii] - priStep_ts_a[i-1,ii]
            x_dpriStep_ts_a[i,ii] = x_priStep_ts_a[i,ii] - x_priStep_ts_a[i-1,ii]





        for j in range(len(any_min)):
            secStep_b[:,:,any_min[j],i,ii] = cut_chem(secMat_b[:,:,any_min[j],ii],i)
            dsecStep_b[:,:,any_min[j],i,ii] = cut_chem(dsecMat_b[:,:,any_min[j],ii],i)
            secStep_ts_b[i,any_min[j],ii] = np.sum(secStep_b[:,:,any_min[j],i,ii])
            x_secStep_ts_b[i,any_min[j],ii] = np.sum(secStep_b[:,xd_move,any_min[j],i,ii])
            if i > 0:
                dsecStep_ts_b[i,any_min[j],ii] = secStep_ts_b[i,any_min[j],ii] - secStep_ts_b[i-1,any_min[j],ii]
                x_dsecStep_ts_b[i,any_min[j],ii] = x_secStep_ts_b[i,any_min[j],ii] - x_secStep_ts_b[i-1,any_min[j],ii]
        dic_b = cut_chem(dic0,i)
        ca_b = cut_chem(ca0,i)
        ph_b = cut_chem(ph0,i)
        alk_b = cut_chem(alk0,i)
        solw_b = cut_chem(solw0,i)
        mg_b = cut_chem(mg0,i)
        fe_b = cut_chem(fe0,i)
        si_b = cut_chem(si0,i)
        k1_b = cut_chem(k0,i)
        na_b = cut_chem(na0,i)
        cl_b = cut_chem(cl0,i)
        al_b = cut_chem(al0,i)
        glass_b = cut_chem(glass0,i)
        alt_vol_b[:,:,i,ii] = cut_chem(alt_vol0_b[:,:,ii],i)
        pri_total_b = cut_chem(pri_total0,i)

        priStep_b[:,:,i,ii] = cut_chem(priMat_b[:,:,ii],i)
        dpriStep_b[:,:,i,ii] = cut_chem(dpriMat_b[:,:,ii],i)
        priStep_ts_b[i,ii] = np.sum(priStep_b[:,:,i,ii])
        x_priStep_ts_b[i,ii] = np.sum(priStep_b[:,xd_move,i,ii])
        if i > 0:
            dpriStep_ts_b[i,ii] = priStep_ts_b[i,ii] - priStep_ts_b[i-1,ii]
            x_dpriStep_ts_b[i,ii] = x_priStep_ts_b[i,ii] - x_priStep_ts_b[i-1,ii]






        for j in range(len(any_min)):
            secStep_d[:,:,any_min[j],i,ii] = cut_chem(secMat_d[:,:,any_min[j],ii],i)
            dsecStep_d[:,:,any_min[j],i,ii] = cut_chem(dsecMat_d[:,:,any_min[j],ii],i)
            secStep_ts_d[i,any_min[j],ii] = np.sum(secStep_d[:,:,any_min[j],i,ii])
            x_secStep_ts_d[i,any_min[j],ii] = np.sum(secStep_d[:,xd_move,any_min[j],i,ii])
            if i > 0:
                dsecStep_ts_d[i,any_min[j],ii] = secStep_ts_d[i,any_min[j],ii] - secStep_ts_d[i-1,any_min[j],ii]
                x_dsecStep_ts_d[i,any_min[j],ii] = x_secStep_ts_d[i,any_min[j],ii] - x_secStep_ts_d[i-1,any_min[j],ii]
        dic_d = cut_chem(dic0,i)
        ca_d = cut_chem(ca0,i)
        ph_d = cut_chem(ph0,i)
        alk_d = cut_chem(alk0,i)
        solw_d = cut_chem(solw0,i)
        mg_d = cut_chem(mg0,i)
        fe_d = cut_chem(fe0,i)
        si_d = cut_chem(si0,i)
        k1_d = cut_chem(k0,i)
        na_d = cut_chem(na0,i)
        cl_d = cut_chem(cl0,i)
        al_d = cut_chem(al0,i)
        glass_d = cut_chem(glass0,i)
        alt_vol_d[:,:,i,ii] = cut_chem(alt_vol0_d[:,:,ii],i)
        pri_total_d = cut_chem(pri_total0,i)

        priStep_d[:,:,i,ii] = cut_chem(priMat_d[:,:,ii],i)
        dpriStep_d[:,:,i,ii] = cut_chem(dpriMat_d[:,:,ii],i)
        priStep_ts_d[i,ii] = np.sum(priStep_d[:,:,i,ii])
        x_priStep_ts_d[i,ii] = np.sum(priStep_d[:,xd_move,i,ii])
        if i > 0:
            dpriStep_ts_d[i,ii] = priStep_ts_d[i,ii] - priStep_ts_d[i-1,ii]
            x_dpriStep_ts_d[i,ii] = x_priStep_ts_d[i,ii] - x_priStep_ts_d[i-1,ii]






        #hack: ternary K, Fe, Mg processing
        for k in range(len(yCell)):
            for j in range(len(xCell)):

                ternK[k,j,i,ii] = 1.0*secStep[k,j,14,i,ii]
                ternMg[k,j,i,ii] = 0.165*secStep[k,j,22,i,ii] + 5.0*secStep[k,j,31,i,ii] + 3.0*secStep[k,j,11,i,ii] + 3.165*secStep[k,j,2,i,ii] + 3.0*secStep[k,j,33,i,ii]
                ternFe[k,j,i,ii] = 1.0*secStep[k,j,14,i,ii] + 2.0*secStep[k,j,22,i,ii] + 2.0*secStep[k,j,17,i,ii] + 1.0*secStep[k,j,5,i,ii] + 2.0*secStep[k,j,15,i,ii]

                ternK_d[k,j,i,ii] = 1.0*secStep_d[k,j,14,i,ii]
                ternMg_d[k,j,i,ii] = 0.165*secStep_d[k,j,22,i,ii] + 5.0*secStep_d[k,j,31,i,ii] + 3.0*secStep_d[k,j,11,i,ii] + 3.165*secStep_d[k,j,2,i,ii] + 3.0*secStep_d[k,j,33,i,ii]
                ternFe_d[k,j,i,ii] = 1.0*secStep_d[k,j,14,i,ii] + 2.0*secStep_d[k,j,22,i,ii] + 2.0*secStep_d[k,j,17,i,ii] + 1.0*secStep_d[k,j,5,i,ii] + 2.0*secStep_d[k,j,15,i,ii]

                ternK_a[k,j,i,ii] = 1.0*secStep_a[k,j,14,i,ii]
                ternMg_a[k,j,i,ii] = 0.165*secStep_a[k,j,22,i,ii] + 5.0*secStep_a[k,j,31,i,ii] + 3.0*secStep_a[k,j,11,i,ii] + 3.165*secStep_a[k,j,2,i,ii] + 3.0*secStep_a[k,j,33,i,ii]
                ternFe_a[k,j,i,ii] = 1.0*secStep_a[k,j,14,i,ii] + 2.0*secStep_a[k,j,22,i,ii] + 2.0*secStep_a[k,j,17,i,ii] + 1.0*secStep_a[k,j,5,i,ii] + 2.0*secStep_a[k,j,15,i,ii]

                ternK_b[k,j,i,ii] = 1.0*secStep_b[k,j,14,i,ii]
                ternMg_b[k,j,i,ii] = 0.165*secStep_b[k,j,22,i,ii] + 5.0*secStep_b[k,j,31,i,ii] + 3.0*secStep_b[k,j,11,i,ii] + 3.165*secStep_b[k,j,2,i,ii] + 3.0*secStep_b[k,j,33,i,ii]
                ternFe_b[k,j,i,ii] = 1.0*secStep_b[k,j,14,i,ii] + 2.0*secStep_b[k,j,22,i,ii] + 2.0*secStep_b[k,j,17,i,ii] + 1.0*secStep_b[k,j,5,i,ii] + 2.0*secStep_b[k,j,15,i,ii]

        # ternK = 39.0*ternK
        # ternK_d = 39.0*ternK_d
        # ternK_a = 39.0*ternK_a
        # ternK_b = 39.0*ternK_b
        #
        # ternFe = 55.0*ternFe
        # ternFe_d = 55.0*ternFe_d
        # ternFe_a = 55.0*ternFe_a
        # ternFe_b = 55.0*ternFe_b
        #
        # ternMg = 23.0*ternMg
        # ternMg_d = 23.0*ternMg_d
        # ternMg_a = 23.0*ternMg_a
        # ternMg_b = 23.0*ternMg_b

        tern_count = 0
        for k in range(len(yCell)):
            for j in range(len(xCell)):
                tern_list[tern_count,i,0,ii] = 39.0*ternK[k,j,i,ii]
                tern_list[tern_count,i,1,ii] = 23.0*ternMg[k,j,i,ii]
                tern_list[tern_count,i,2,ii] = 56.0*ternFe[k,j,i,ii]
                #tern_list[tern_count,i,:,ii] = tern_list[tern_count,i,:,ii]/np.sum(tern_list[tern_count,i,:,ii])
                if np.max(tern_list[tern_count,i,:,ii]) > 0.0:
                    tern_list[tern_count,i,:,ii] = tern_list[tern_count,i,:,ii]/(1.0*tern_list[tern_count,i,0,ii] + 1.0*tern_list[tern_count,i,1,ii] + 1.0*tern_list[tern_count,i,2,ii])
                    # if i == 20:
                    #     print "S" , tern_list[tern_count,i,:,ii]

                tern_list_d[tern_count,i,0,ii] = 39.0*ternK_d[k,j,i,ii]
                tern_list_d[tern_count,i,1,ii] = 23.0*ternMg_d[k,j,i,ii]
                tern_list_d[tern_count,i,2,ii] = 56.0*ternFe_d[k,j,i,ii]
                #tern_list_d[tern_count,i,:,ii] = tern_list_d[tern_count,i,:,ii]/np.sum(tern_list_d[tern_count,i,:,ii])
                if np.max(tern_list_d[tern_count,i,:,ii]) > 0.0:
                    tern_list_d[tern_count,i,:,ii] = tern_list_d[tern_count,i,:,ii]/(1.0*tern_list_d[tern_count,i,0,ii] + 1.0*tern_list_d[tern_count,i,1,ii] + 1.0*tern_list_d[tern_count,i,2,ii])
                    # if i == 20:
                    #     print "D" , tern_list_d[tern_count,i,:,ii]

                tern_list_a[tern_count,i,0,ii] = 39.0*ternK_a[k,j,i,ii]
                tern_list_a[tern_count,i,1,ii] = 23.0*ternMg_a[k,j,i,ii]
                tern_list_a[tern_count,i,2,ii] = 56.0*ternFe_a[k,j,i,ii]
                #tern_list_a[tern_count,i,:,ii] = tern_list_a[tern_count,i,:,ii]/np.sum(tern_list_a[tern_count,i,:,ii])
                if np.max(tern_list_a[tern_count,i,:,ii]) > 0.0:
                    tern_list_a[tern_count,i,:,ii] = tern_list_a[tern_count,i,:,ii]/(1.0*tern_list_a[tern_count,i,0,ii] + 1.0*tern_list_a[tern_count,i,1,ii] + 1.0*tern_list_a[tern_count,i,2,ii])
                    # if i == 20:
                    #     print "A" , tern_list_a[tern_count,i,:,ii]

                tern_list_b[tern_count,i,0,ii] = 39.0*ternK_b[k,j,i,ii]
                tern_list_b[tern_count,i,1,ii] = 23.0*ternMg_b[k,j,i,ii]
                tern_list_b[tern_count,i,2,ii] = 56.0*ternFe_b[k,j,i,ii]
                #tern_list_b[tern_count,i,:,ii] = tern_list_b[tern_count,i,:,ii]/np.sum(tern_list_b[tern_count,i,:,ii])
                if np.max(tern_list_b[tern_count,i,:,ii]) > 0.0:
                    tern_list_b[tern_count,i,:,ii] = tern_list_b[tern_count,i,:,ii]/(1.0*tern_list_b[tern_count,i,0,ii] + 1.0*tern_list_b[tern_count,i,1,ii] + 1.0*tern_list_b[tern_count,i,2,ii])
                    # if i == 20:
                    #     print "B" , tern_list_b[tern_count,i,:,ii]

                tern_count = tern_count + 1




        #hack: sec binary
        f_thresh = 10.0
        for k in range(minNum+1):
            for j in range(len(xCell)):

                #if np.max(secStep[:,j,k,i,ii]) > 0.0:
                if np.max(secStep[:,j,k,i,ii]) > np.max(secStep[:,:,k,i,ii])/f_thresh:
                    sec_binary[j,i,k,ii] = 1.0
                else:
                    sec_binary[j,i,k,ii] = None

                if np.max(secStep_a[:,j,k,i,ii]) > np.max(secStep_a[:,:,k,i,ii])/f_thresh:
                    sec_binary_a[j,i,k,ii] = 1.0
                else:
                    sec_binary_a[j,i,k,ii] = None

                if np.max(secStep_b[:,j,k,i,ii]) > np.max(secStep_b[:,:,k,i,ii])/f_thresh:
                    sec_binary_b[j,i,k,ii] = 1.0
                else:
                    sec_binary_b[j,i,k,ii] = None

                if np.max(secStep_d[:,j,k,i,ii]) > np.max(secStep_d[:,:,k,i,ii])/f_thresh:
                    sec_binary_d[j,i,k,ii] = 1.0
                else:
                    sec_binary_d[j,i,k,ii] = None




        #hack: alt data

        for j in range(len(xCell)):
            # full column average

            if ii == 0:
                above_zero = alt_vol[:,j,i,ii]*100.0
                above_zero = above_zero[above_zero>0.0]
                alt_col_mean[j,i,len(param_t_diff_string)] = np.mean(above_zero)

            above_zero = alt_vol_d[:,j,i,ii]*100.0
            above_zero = above_zero[above_zero>0.0]
            #print above_zero
            alt_col_mean[j,i,ii] = np.mean(above_zero)


            # # top half of column average
            # above_zero = alt_vol[:,j]*100.0
            # above_zero = above_zero[above_zero>0.0]
            # alt_col_mean_top_half_s[j] = np.mean(above_zero[len(above_zero)/2:])
            #
            # above_zero = alt_vol_d[:,j]*100.0
            # above_zero = above_zero[above_zero>0.0]
            # alt_col_mean_top_half_d[j] = np.mean(above_zero[len(above_zero)/2:])
            #
            #
            # # top cell of column
            # above_zero = alt_vol[:,j]*100.0
            # above_zero = above_zero[above_zero>0.0]
            # alt_col_mean_top_cell_s[j] = np.mean(above_zero[-1:])
            #
            # above_zero = alt_vol_d[:,j]*100.0
            # above_zero = above_zero[above_zero>0.0]
            # alt_col_mean_top_cell_d[j] = np.mean(above_zero[-1:])



            feo_col_mean_temp = np.zeros(len(xCell))
            feot_col_mean_temp = np.zeros(len(xCell))

            if ii == 0:


                secStep_temp = secStep[:,:,:,i,ii]
                alt_vol_temp = pri_total
                glass_temp = glass

                above_zero = alt_vol_temp[:,j]*100.0
                above_zero = above_zero[above_zero>0.0]
                above_zero_ind = np.nonzero(alt_vol_temp[:,j])

                for j in range(len(xCell)):

                    feo_col_mean_temp[j] = 0.0
                    # feo glass
                    # feo_col_mean_temp[j] = 0.149*np.mean(glass_temp[above_zero_ind,j])*(density_pri[0]/molar_pri[0])
                    # feo olivine
                    feo_col_mean_temp[j] = feo_col_mean_temp[j] + 0.166*np.mean(glass_temp[above_zero_ind,j])*(density_pri[3]/molar_pri[3])
                    # # feo pyrite
                    # feo_col_mean_temp[j] = feo_col_mean_temp[j] + np.mean(secStep_temp[above_zero_ind,j,5])*(density[5]/molar[5])
                    # feo fe-celad
                    feo_col_mean_temp[j] = feo_col_mean_temp[j] + np.mean(secStep_temp[above_zero_ind,j,14])*(density[14]/molar[14])


                    feot_col_mean_temp[j] = 0.0
                    # feot goethite
                    feot_col_mean_temp[j] = 0.8998*.0234*2.0*np.mean(glass_temp[above_zero_ind,j])*(density_pri[3]/molar_pri[3])
                    # feo goethite
                    feot_col_mean_temp[j] = feot_col_mean_temp[j] + 0.8998*np.mean(secStep_temp[above_zero_ind,j,7])*(density[7]/molar[7])
                    # # feot pyrite
                    # feot_col_mean_temp[j] = feot_col_mean_temp[j] + 1.0*0.8998*np.mean(secStep_temp[above_zero_ind,j,5])*(density[5]/molar[5])
                    # feot hematite
                    feot_col_mean_temp[j] = feot_col_mean_temp[j] + 2.0*0.8998*np.mean(secStep_temp[above_zero_ind,j,17])*(density[17]/molar[17])
                    # feot nont-mg
                    feot_col_mean_temp[j] = feot_col_mean_temp[j] + 2.0*0.8998*np.mean(secStep_temp[above_zero_ind,j,13])*(density[13]/molar[13])
                    # feot nont-ca
                    feot_col_mean_temp[j] = feot_col_mean_temp[j] + 2.0*0.8998*np.mean(secStep_temp[above_zero_ind,j,15])*(density[15]/molar[15])
                    # feot nont-na
                    feot_col_mean_temp[j] = feot_col_mean_temp[j] + 2.0*0.8998*np.mean(secStep_temp[above_zero_ind,j,12])*(density[12]/molar[12])

                    fe_col_mean[j,i,len(param_t_diff_string)] = feo_col_mean_temp[j] / (feo_col_mean_temp[j] + feot_col_mean_temp[j])


            secStep_temp = secStep_d[:,:,:,i,ii]
            alt_vol_temp = pri_total_d
            glass_temp = glass_d

            above_zero = alt_vol_temp[:,j]*100.0
            above_zero = above_zero[above_zero>0.0]
            above_zero_ind = np.nonzero(alt_vol_temp[:,j])

            for j in range(len(xCell)):
                feo_col_mean_temp[j] = 0.0
                # feo glass
                # feo_col_mean_temp[j] = 0.149*np.mean(glass_temp[above_zero_ind,j])*(density_pri[0]/molar_pri[0])
                # feo olivine
                feo_col_mean_temp[j] = feo_col_mean_temp[j] + 0.166*np.mean(glass_temp[above_zero_ind,j])*(density_pri[3]/molar_pri[3])
                # # feo pyrite
                # feo_col_mean_temp[j] = feo_col_mean_temp[j] + np.mean(secStep_temp[above_zero_ind,j,5])*(density[5]/molar[5])
                # feo fe-celad
                feo_col_mean_temp[j] = feo_col_mean_temp[j] + np.mean(secStep_temp[above_zero_ind,j,14])*(density[14]/molar[14])


                feot_col_mean_temp[j] = 0.0
                # feot goethite
                feot_col_mean_temp[j] = 0.8998*.0234*2.0*np.mean(glass_temp[above_zero_ind,j])*(density_pri[3]/molar_pri[3])
                # feo goethite
                feot_col_mean_temp[j] = feot_col_mean_temp[j] + 0.8998*np.mean(secStep_temp[above_zero_ind,j,7])*(density[7]/molar[7])
                # # feot pyrite
                # feot_col_mean_temp[j] = feot_col_mean_temp[j] + 1.0*0.8998*np.mean(secStep_temp[above_zero_ind,j,5])*(density[5]/molar[5])
                # feot hematite
                feot_col_mean_temp[j] = feot_col_mean_temp[j] + 2.0*0.8998*np.mean(secStep_temp[above_zero_ind,j,17])*(density[17]/molar[17])
                # feot nont-mg
                feot_col_mean_temp[j] = feot_col_mean_temp[j] + 2.0*0.8998*np.mean(secStep_temp[above_zero_ind,j,13])*(density[13]/molar[13])
                # feot nont-ca
                feot_col_mean_temp[j] = feot_col_mean_temp[j] + 2.0*0.8998*np.mean(secStep_temp[above_zero_ind,j,15])*(density[15]/molar[15])
                # feot nont-na
                feot_col_mean_temp[j] = feot_col_mean_temp[j] + 2.0*0.8998*np.mean(secStep_temp[above_zero_ind,j,12])*(density[12]/molar[12])

                fe_col_mean[j,i,ii] = feo_col_mean_temp[j] / (feo_col_mean_temp[j] + feot_col_mean_temp[j])











#todo: FIGURE: jdf_alt_plot, NXF
fig=plt.figure(figsize=(7.0,9.0))




nsites = 9
ebw = 800.0
dark_red = '#65091f'
plot_purple = '#b678f5'
plot_blue = '#4e94c1'
site_locations = np.array([22.742, 25.883, 33.872, 40.706, 45.633, 55.765, 75.368, 99.006, 102.491])
site_locations = (site_locations - 20.00)*1000.0
site_names = ["1023", "1024", "1025", "1031", "1028", "1029", "1032", "1026", "1027"]
alt_values = np.array([0.3219, 2.1072, 2.3626, 2.9470, 10.0476, 4.2820, 8.9219, 11.8331, 13.2392])
lower_eb = np.array([0.3219, 0.04506, 0.8783, 1.7094, 5.0974, 0.8994, 5.3745, 2.5097, 3.0084])
upper_eb = np.array([1.7081, 2.9330, 3.7662, 4.9273, 11.5331, 5.0247, 10.7375, 17.8566, 27.4308])


# alteration data
ax=fig.add_subplot(2, 1, 1, frameon=True)
ax.grid(True)
plt.scatter(site_locations,alt_values,edgecolor=dark_red,color=dark_red,zorder=10,s=60, label="data from sites")
plt.plot(site_locations,alt_values,color=dark_red,linestyle='-')

for j in range(nsites):
    # error bar height
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red)
    # lower error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red)
    # upper error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red)

# plot model column mean
for ii in range(len(param_t_diff)+1):

    plt.plot(xCell,alt_col_mean[:,max_step-1,ii],color=plot_col[ii],lw=1.5, label=plot_strings[ii])
    # print " "
    # print " "
    # print ii
    # print alt_col_mean[:,max_step-1,ii]


plt.legend(fontsize=10,loc=2)
plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=12)
plt.xlim([0.0, 90000.0])
plt.xlabel('Distance along transect [km]')

plt.yticks([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],fontsize=12)
plt.ylim([0.0, 30.0])
plt.ylabel('Alteration volume $\%$')




#todo: FIGURE: FeO / FeOt plot
# FeO / FeOt data
fe_values = np.array([0.7753, 0.7442, 0.7519, 0.7610, 0.6714, 0.7416, 0.7039, 0.6708, 0.6403])
lower_eb_fe = np.array([0.7753, 0.7442, 0.7208, 0.7409, 0.6240, 0.7260, 0.6584, 0.6299, 0.6084])
upper_eb_fe = np.array([0.7753, 0.7442, 0.7519, 0.7812, 0.7110, 0.7610, 0.7396, 0.7104, 0.7026])


ax=fig.add_subplot(2, 1, 2, frameon=True)
ax.grid(True)
plt.scatter(site_locations,fe_values,edgecolor=dark_red,color=dark_red,zorder=10,s=60, label="data from sites")
plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')


for j in range(nsites):
    # error bar height
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red)
    # lower error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    # upper error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)

# plot model column mean
for ii in range(len(param_t_diff)+1):

    plt.plot(xCell,fe_col_mean[:,max_step-1,ii],color=plot_col[ii],lw=1.5, label=plot_strings[ii])
    # print " "
    # print " "
    # print ii
    # print fe_col_mean[:,max_step-1,ii]





#plt.legend(fontsize=10)
plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=12)
plt.xlim([0.0, 90000.0])
plt.xlabel('Distance along transect [km]', fontsize=9)

#plt.yticks([0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80])
plt.yticks([0.6, 0.65, 0.7, 0.75, 0.80])
#plt.ylim([0.6, 0.8])
plt.ylim([0.6, 0.8])
plt.ylabel('FeO / FeOt')


#plt.subplots_adjust( wspace=0.05 , bottom=0.2, top=0.95, left=0.03, right=0.975)
plt.savefig(batch_path+prefix_string+"batch_alt.png",bbox_inches='tight')
#plt.savefig(batch_path+prefix_string+"batch_alt.eps")







#todo: FIGURE: x_pri figure
fig=plt.figure(figsize=(10.0,3.0))

ax=fig.add_subplot(1, 3, 1, frameon=True)
for ii in range(len(param_t_diff)):
    plt.plot(range(steps),x_priStep_ts_d[:,ii]/np.max(x_priStep_ts_d[:,ii]),color=plot_col[ii],lw=1.5, label=plot_strings[ii])
plt.plot(range(steps),x_priStep_ts[:,0]/np.max(x_priStep_ts[:,0]),color=plot_col[len(param_t_diff)],lw=1.5, label=plot_strings[len(param_t_diff)])

temp_pri_min_mat = x_priStep_ts_d/np.max(x_priStep_ts_d)
temp_pri_min = np.min(temp_pri_min_mat[temp_pri_min_mat>0.0])
plt.ylim([temp_pri_min,1.01])
plt.legend(fontsize=8,labelspacing=-0.1,columnspacing=0.0)
plt.title('dual')



ax=fig.add_subplot(1, 3, 2, frameon=True)
for ii in range(len(param_t_diff)):
    plt.plot(range(steps),x_priStep_ts_a[:,ii]/np.max(x_priStep_ts_a[:,ii]),color=plot_col[ii],lw=1.5, label=plot_strings[ii])

temp_pri_min_mat = x_priStep_ts_a/np.max(x_priStep_ts_a)
temp_pri_min = np.min(temp_pri_min_mat[temp_pri_min_mat>0.0])
plt.ylim([temp_pri_min,1.01])
plt.title('a only')



ax=fig.add_subplot(1, 3, 3, frameon=True)
for ii in range(len(param_t_diff)):
    plt.plot(range(steps),x_priStep_ts_b[:,ii]/np.max(x_priStep_ts_b[:,ii]),color=plot_col[ii],lw=1.5, label=plot_strings[ii])

temp_pri_min_mat = x_priStep_ts_b/np.max(x_priStep_ts_b)
temp_pri_min = np.min(temp_pri_min_mat[temp_pri_min_mat>0.0])
plt.ylim([temp_pri_min,1.01])
plt.title('b only')


# for ii in range(len(param_t_diff)):
#     print x_priStep_ts[:,ii]
#     print " "

plt.savefig(batch_path+prefix_string+"pri.png",bbox_inches='tight')



# #todo: FIGURE: sec_binary
# fig=plt.figure(figsize=(9.0,7.0))
#
# site_locations = site_locations + 5000.0
#
# # mindex = [2,3,5,14,17,22,26,30,31]
# mindex = [2,14,5,17,13,26,16,31]
# data_col = '#222222'
#
# site_binary = np.zeros([9,minNum])
#
# # mg sap
# site_binary[:,2] = np.array([None, 1.0, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#
# # # celadonite
# # site_binary[:,14] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#
# # pyrite
# site_binary[:,5] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#
# # fe-celad
# site_binary[:,14] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#
# # hematite
# site_binary[:,17] = np.array([None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#
# # mont mg
# site_binary[:,22] = np.array([None, None, None, None, None, None, 1.0, 1.0, 1.0])
#
# # talc
# site_binary[:,26] = np.array([None, None, None, None, 1.0, None, 1.0, 1.0, 1.0])
#
# # scol (u zeolites)
# site_binary[:,30] = np.array([None, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#
# # chlorite (clinochlore)
# site_binary[:,31] = np.array([None, None, None, None, 1.0, None, 1.0, None, 1.0])
#
#
# ax=fig.add_subplot(1, 1, 1, frameon=True)
# #ax.grid(True)
#
# for k in range(len(mindex)):
#     plt.plot(xCell,(k+1.0)*sec_binary[:,max_step-1,mindex[k],0]+.15*(0-1.0)-.15,color=plot_col[len(param_t_diff)],lw=3)
#
# for ii in range(len(param_t_diff)):
#     for k in range(len(mindex)):
#         # plt.scatter(xCell,(k+1.0)*sec_binary_d[:,max_step-1,mindex[k],ii]+.2*(ii-1.0),color=plot_col[ii],edgecolor=plot_col[ii],s=5)
#         plt.plot(xCell,(k+1.0)*sec_binary_d[:,max_step-1,mindex[k],ii]+.15*(ii-1.0),color=plot_col[ii],lw=3)
#         plt.scatter(site_locations,(k+1.0)*site_binary[:,mindex[k]],color=data_col,edgecolor=data_col,s=50,lw=2.5,zorder=10,marker='x')
#
# # for k in range(len(mindex)):
# #     for ii in range(len(site_locations)):
# #         plt.plot([site_locations[ii], site_locations[ii]],[(k+1.0)*site_binary[ii,mindex[k]]-.17,(k+1.0)*site_binary[ii,mindex[k]]+0.14],color=data_col,lw=3,zorder=10)
#
#
#
# plt.yticks(range(1,len(mindex)+1), secondary[mindex], fontsize=14)
# #plt.ylim([0.6, len(mindex)+1])
# plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=14)
# plt.xlim([0.0, 90000.0])
# plt.xlabel('Distance along transect [km]', fontsize=9)
# plt.title('dual vs solo')
#
#
#
#
# # ax=fig.add_subplot(1, 3, 2, frameon=True)
# #
# # for k in range(len(mindex)):
# #     plt.plot(xCell,(k+1.0)*sec_binary[:,max_step-1,mindex[k],0]+.15*(0-1.0)-.15,color=plot_col[len(param_t_diff)],lw=2)
# #
# # for ii in range(len(param_t_diff)):
# #     for k in range(len(mindex)):
# #         plt.plot(xCell,(k+1.0)*sec_binary_a[:,max_step-1,mindex[k],ii]+.15*(ii-1.0),color=plot_col[ii],lw=2)
# #         plt.scatter(site_locations,(k+1.0)*site_binary[:,mindex[k]],color=data_col,edgecolor=data_col,s=50,lw=2.5,zorder=10,marker='x')
# #
# # #plt.yticks(range(1,len(mindex)+1), secondary[mindex])
# # plt.yticks([])
# # plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=9)
# # plt.xlim([0.0, 90000.0])
# # plt.xlabel('Distance along transect [km]', fontsize=9)
# # plt.title('a only')
# #
#
#
#
# # ax=fig.add_subplot(1, 3, 3, frameon=True)
# #
# # for k in range(len(mindex)):
# #     plt.plot(xCell,(k+1.0)*sec_binary[:,max_step-1,mindex[k],0]+.15*(0-1.0)-.15,color=plot_col[len(param_t_diff)],lw=2)
# #     plt.scatter(site_locations,(k+1.0)*site_binary[:,mindex[k]],color=data_col,edgecolor=data_col,s=50,lw=2.5,zorder=10,marker='x')
# #
# # for ii in range(len(param_t_diff)):
# #     for k in range(len(mindex)):
# #         plt.plot(xCell,(k+1.0)*sec_binary_b[:,max_step-1,mindex[k],ii]+.15*(ii-1.0),color=plot_col[ii],lw=2)
# #
# #
# # plt.yticks([])
# # plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=9)
# # plt.xlim([0.0, 90000.0])
# # plt.xlabel('Distance along transect [km]', fontsize=9)
# # plt.title('b only')
#
#
#
# plt.subplots_adjust( wspace=0.05 , bottom=0.05, top=0.95, left=0.15, right=0.975)
# plt.savefig(batch_path+prefix_string+"batch_binary.png")






#todo: FIGURE: ternary K, Fe, Mg
fig=plt.figure(figsize=(11.0,5.5))


# ternary values for explicit phases
# K, Mg, Fe
tern_size = 8
tern_size_small = 10
tern_saponite_mg = [[0.0, 1.0, 0.0]]
tern_fe_celadonite = [[0.5, 0.0, 0.5]]
tern_fe_oxide = [[0.0, 0.0, 1.0]]
tern_min_col = '#524aaf'


# tern_min_kwargs = {'marker': 's', 'color': tern_min_col}
tern_min_kwargs = dict(color=tern_min_col, marker='s', markersize=tern_size, markeredgecolor='k', linewidth=0.0, zorder=10)
tern_model_kwargs = dict(marker='.', markersize=tern_size_small, markeredgecolor='none', linewidth=0.0)
tern_model_kwargs_big = dict(marker='.', markersize=tern_size_small*2.0, markeredgecolor='none', linewidth=0.0)

ax=fig.add_subplot(2, 4, 1)
fig, tax = ternary.figure(ax=ax,scale=1.0)
tax.boundary()

tax.gridlines(multiple=0.2, color="black")
tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
tax.plot(tern_fe_celadonite, **tern_min_kwargs)
tax.plot(tern_fe_oxide, **tern_min_kwargs)
for ii in range(len(param_t_diff)):
#for ii in [3]:
    tax.plot(tern_list[:,max_step-1,:,ii], color=plot_col[ii], label=plot_strings[ii], **tern_model_kwargs)
tax.set_title("solo")
tax.legend(fontsize=9, bbox_to_anchor=(1.48, 1.1), ncol=1,labelspacing=0.0,columnspacing=0.0,numpoints=1)
tax.clear_matplotlib_ticks()

tax.get_axes().axis('off')




ax=fig.add_subplot(2, 4, 2)
fig, tax = ternary.figure(ax=ax,scale=1.0)
tax.boundary()


tax.gridlines(multiple=0.2, color="black")
tax.plot(tern_saponite_mg, **tern_min_kwargs)
tax.plot(tern_fe_celadonite, **tern_min_kwargs)
tax.plot(tern_fe_oxide, **tern_min_kwargs)
#for ii in range(len(param_t_diff)):
for ii in [final_index]:
    tax.plot(tern_list_d[:,max_step-1,:,ii], color=plot_col[ii], label=plot_strings[ii]+'d', **tern_model_kwargs)
for ii in [0]:
    tax.plot(tern_list_d[:,max_step-1,:,ii], color=plot_col[ii], label=plot_strings[ii]+'d', zorder=0, **tern_model_kwargs_big)
tax.set_title("dual")
tax.clear_matplotlib_ticks()

tax.get_axes().axis('off')



ax=fig.add_subplot(2, 4, 3)
fig, tax = ternary.figure(ax=ax,scale=1.0)
tax.boundary()


tax.gridlines(multiple=0.2, color="black")
tax.plot(tern_saponite_mg, **tern_min_kwargs)
tax.plot(tern_fe_celadonite, **tern_min_kwargs)
tax.plot(tern_fe_oxide, **tern_min_kwargs)
#for ii in range(len(param_t_diff)):
for ii in [final_index]:
    tax.plot(tern_list_a[:,max_step-1,:,ii], color=plot_col[ii], label=plot_strings[ii]+'a', **tern_model_kwargs)
for ii in [0]:
    tax.plot(tern_list_a[:,max_step-1,:,ii], color=plot_col[ii], label=plot_strings[ii]+'a', zorder=0, **tern_model_kwargs_big)
tax.set_title("a")
tax.clear_matplotlib_ticks()

tax.get_axes().axis('off')





ax=fig.add_subplot(2, 4, 4)
fig, tax = ternary.figure(ax=ax,scale=1.0)
tax.boundary()


tax.gridlines(multiple=0.2, color="black")
tax.plot(tern_saponite_mg, **tern_min_kwargs)
tax.plot(tern_fe_celadonite, **tern_min_kwargs)
tax.plot(tern_fe_oxide, **tern_min_kwargs)
#for ii in range(len(param_t_diff)):
for ii in [final_index]:
    tax.plot(tern_list_b[:,max_step-1,:,ii], color=plot_col[ii], label=plot_strings[ii]+'b', **tern_model_kwargs)
for ii in [0]:
    tax.plot(tern_list_b[:,max_step-1,:,ii], color=plot_col[ii], label=plot_strings[ii]+'b', zorder=0, **tern_model_kwargs_big)
tax.set_title("b")
tax.clear_matplotlib_ticks()

tax.get_axes().axis('off')












ax=fig.add_subplot(2, 4, 6)
fig, tax = ternary.figure(ax=ax,scale=1.0)
tax.boundary()


tax.gridlines(multiple=0.2, color="black")
tax.plot(tern_saponite_mg, **tern_min_kwargs)
tax.plot(tern_fe_celadonite, **tern_min_kwargs)
tax.plot(tern_fe_oxide, **tern_min_kwargs)
for ii in range(len(param_t_diff)):
#for ii in [0]:
    tax.plot(tern_list_d[:,max_step-1,:,ii], color=plot_col[ii], label=plot_strings[ii]+'d', **tern_model_kwargs)
tax.set_title("dual")
tax.clear_matplotlib_ticks()

tax.get_axes().axis('off')



ax=fig.add_subplot(2, 4, 7)
fig, tax = ternary.figure(ax=ax,scale=1.0)
tax.boundary()


tax.gridlines(multiple=0.2, color="black")
tax.plot(tern_saponite_mg, **tern_min_kwargs)
tax.plot(tern_fe_celadonite, **tern_min_kwargs)
tax.plot(tern_fe_oxide, **tern_min_kwargs)
for ii in range(len(param_t_diff)):
#for ii in [0]:
    tax.plot(tern_list_a[:,max_step-1,:,ii], color=plot_col[ii], label=plot_strings[ii]+'a', **tern_model_kwargs)
tax.set_title("a")
tax.clear_matplotlib_ticks()

tax.get_axes().axis('off')





ax=fig.add_subplot(2, 4, 8)
fig, tax = ternary.figure(ax=ax,scale=1.0)
tax.boundary()


tax.gridlines(multiple=0.2, color="black")
tax.plot(tern_saponite_mg, **tern_min_kwargs)
tax.plot(tern_fe_celadonite, **tern_min_kwargs)
tax.plot(tern_fe_oxide, **tern_min_kwargs)
for ii in range(len(param_t_diff)):
#for ii in [0]:
    tax.plot(tern_list_b[:,max_step-1,:,ii], color=plot_col[ii], label=plot_strings[ii]+'b', **tern_model_kwargs)
tax.set_title("b")
tax.clear_matplotlib_ticks()

tax.get_axes().axis('off')

plt.subplots_adjust(bottom=0.07, top=0.93, left=0.03, right=0.975)
plt.savefig(batch_path+prefix_string+"ternary_"+str(max_step)+".png")
#plt.savefig(batch_path+prefix_string+"ternary_"+str(max_step)+".eps")


other_step = 39
