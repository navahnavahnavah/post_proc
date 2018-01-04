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

plot_col = ['#940000', '#cf6948', '#fc9700', '#2ab407', '#6aabf7', '#bb43e6']

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

# molar_pri = np.array([277.0, 153.0, 158.81, 110.0])
molar_pri = np.array([277.0, 153.0, 158.81, 110.0])

density_pri = np.array([2.7, 3.0, 3.0, 2.7])

#hack: path stuff
prefix_string = "su_80_75_05_"
suffix_string = "/"
batch_path = "../output/revival/summer_coarse_grid/"
batch_path_ex = "../output/revival/summer_coarse_grid/"+prefix_string+"50A_50B_2e10/"



# param_t_diff = np.array([8e10, 6e10, 4e10, 2e10])
# param_t_diff_string = ['8e10', '6e10' , '4e10', '2e10']
# plot_t_diff_strings = ['8e10 (least mix)', '6e10', '4e10', '2e10 (most mix)', 'solo']

param_t_diff = np.array([10e10, 8e10, 6e10, 4e10, 2e10])
param_t_diff_string = ['10e10', '8e10' , '6e10' , '4e10', '2e10']
plot_t_diff_strings = ['10e10 (least mix)', '8e10', '6e10', '4e10', '2e10 (most mix)', 'solo']

# param_sim = np.array([20, 40, 60, 80])
# param_sim_string = ['20A_80B', '40A_60B', '60A_40B' , '80A_20B']
# plot_sim_strings = ['20A_80B', '40A_60B', '60A_40B' , '80A_20B']

param_sim = np.array([20, 30, 40, 50, 60, 70, 80])
param_sim_string = ['20A_80B', '30A_70B', '40A_60B', '50A_50B', '60A_40B', '70A_30B', '80A_20B']
plot_sim_strings = ['20A_80B', '30A_70B', '40A_60B', '50A_50B', '60A_40B', '70A_30B', '80A_20B']


x0 = np.loadtxt(batch_path_ex + 'x.txt',delimiter='\n')
y0 = np.loadtxt(batch_path_ex + 'y.txt',delimiter='\n')

#hack: params here
cellx = 90
celly = 1
steps = 50
minNum = 41
# even number
max_step = 20
final_index = 4
restart = 1

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
secMat = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
secMat_a = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
secMat_b = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
secMat_d = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])

secStep = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])
secStep_a = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])
secStep_b = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])
secStep_d = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])

secStep_last = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])
secStep_last_a = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])
secStep_last_b = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])
secStep_last_d = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])

dsecMat = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
dsecMat_a = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
dsecMat_b = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
dsecMat_d = np.zeros([len(yCell),len(xCell)*steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])

dsecStep = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])
dsecStep_a = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])
dsecStep_b = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])
dsecStep_d = np.zeros([len(yCell),len(xCell),minNum+1,steps,len(param_t_diff_string),len(param_sim_string)])

secStep_ts = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
secStep_ts_a = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
secStep_ts_b = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
secStep_ts_d = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])

dsecStep_ts = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string) ])
dsecStep_ts_a = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
dsecStep_ts_b = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
dsecStep_ts_d = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])

x_secStep_ts = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
x_secStep_ts_a = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
x_secStep_ts_b = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
x_secStep_ts_d = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])

x_dsecStep_ts = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
x_dsecStep_ts_a = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
x_dsecStep_ts_b = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
x_dsecStep_ts_d = np.zeros([steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])

sec_binary = np.zeros([len(xCell),steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
sec_binary_a = np.zeros([len(xCell),steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
sec_binary_b = np.zeros([len(xCell),steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
sec_binary_d = np.zeros([len(xCell),steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])
sec_binary_any = np.zeros([len(xCell),steps,minNum+1,len(param_t_diff_string),len(param_sim_string)])

x_d = -5
xd_move = 0
moves = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40, 42])



# new 12/29/17

priMat = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])
priMat_a = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])
priMat_b = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])
priMat_d = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])

priStep = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
priStep_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
priStep_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
priStep_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])

dpriMat = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])
dpriMat_a = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])
dpriMat_b = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])
dpriMat_d = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])

dpriStep = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
dpriStep_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
dpriStep_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
dpriStep_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])

x_priStep_ts = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
x_priStep_ts_a = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
x_priStep_ts_b = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
x_priStep_ts_d = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])

x_dpriStep_ts = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
x_dpriStep_ts_a = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
x_dpriStep_ts_b = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
x_dpriStep_ts_d = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])

#end new




# changed last argument from 6

priStep_ts = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
priStep_ts_a = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
priStep_ts_b = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
priStep_ts_d = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])

dpriStep_ts = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
dpriStep_ts_a = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
dpriStep_ts_b = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
dpriStep_ts_d = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])

# end changed




alt_vol0 = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])
alt_vol0_a = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])
alt_vol0_b = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])
alt_vol0_d = np.zeros([len(yCell),len(xCell)*steps,len(param_t_diff_string),len(param_sim_string)])

alt_vol = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
alt_vol_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
alt_vol_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
alt_vol_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])

alt_col_mean = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])
alt_col_mean_top_half = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])
alt_col_mean_top_cell = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])

fe_col_mean = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])
fe_col_mean_top_half = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])
fe_col_mean_top_cell = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])


pri_mean = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])
pri_mean_d = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])

sec_mean = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])
sec_mean_d = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])

#hack: slope arrays here
alt_col_mean_slope = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])
fe_col_mean_slope = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])
pri_mean_slope = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])
sec_mean_slope = np.zeros([len(xCell),steps,len(param_t_diff_string)+1,len(param_sim_string)+1])

ternK = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
ternK_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
ternK_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
ternK_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])

ternMg = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
ternMg_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
ternMg_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
ternMg_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])

ternFe = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
ternFe_a = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
ternFe_b = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])
ternFe_d = np.zeros([len(yCell),len(xCell),steps,len(param_t_diff_string),len(param_sim_string)])

tern_list = np.zeros([len(yCell)*len(xCell),steps,3,len(param_t_diff_string),len(param_sim_string)])
tern_list_a = np.zeros([len(yCell)*len(xCell),steps,3,len(param_t_diff_string),len(param_sim_string)])
tern_list_b = np.zeros([len(yCell)*len(xCell),steps,3,len(param_t_diff_string),len(param_sim_string)])
tern_list_d = np.zeros([len(yCell)*len(xCell),steps,3,len(param_t_diff_string),len(param_sim_string)])



#hack: net uptake arrays

x_elements = np.zeros([steps,15,len(param_t_diff_string),len(param_sim_string)])
x_elements_d = np.zeros([steps,15,len(param_t_diff_string),len(param_sim_string)])
x_elements_a = np.zeros([steps,15,len(param_t_diff_string),len(param_sim_string)])
x_elements_b = np.zeros([steps,15,len(param_t_diff_string),len(param_sim_string)])

x_pri_elements = np.zeros([steps,15,len(param_t_diff_string),len(param_sim_string)])
x_pri_elements_d = np.zeros([steps,15,len(param_t_diff_string),len(param_sim_string)])
x_pri_elements_a = np.zeros([steps,15,len(param_t_diff_string),len(param_sim_string)])
x_pri_elements_b = np.zeros([steps,15,len(param_t_diff_string),len(param_sim_string)])


elements_sec = np.zeros([minNum+1,15])
elements_pri = np.zeros([1,15])

# elements_pri[0,5] = 0.1178 # Ca
# elements_pri[0,6] = 0.097 # Mg
# elements_pri[0,7] = 0.047 # Na
# elements_pri[0,8] = 0.00219 # K
# elements_pri[0,9] = 0.1165 # Fe
# elements_pri[0,10] = 0.0 # S
# elements_pri[0,11] =  0.4655 # Si
# elements_pri[0,12] = 0.0 # Cl
# elements_pri[0,13] =  0.153 # Al

elements_pri[0,5] = 0.2151 # Ca
elements_pri[0,6] = 0.178 # Mg
elements_pri[0,7] = 0.086 # Na
elements_pri[0,8] = 0.004 # K
elements_pri[0,9] = 0.2128 # Fe
elements_pri[0,10] = 0.0 # S
elements_pri[0,11] =  0.85 # Si
elements_pri[0,12] = 0.0 # Cl
elements_pri[0,13] =  0.28 # Al

# 2 saponite_mg
elements_sec[2,5] = 0.0 # Ca
elements_sec[2,6] = 3.165 # Mg
elements_sec[2,7] = 0.0 # Na
elements_sec[2,8] = 0.0 # K
elements_sec[2,9] = 0.0 # Fe
elements_sec[2,10] = 0.0 # S
elements_sec[2,11] = 3.67 # Si
elements_sec[2,12] = 0.0 # Cl
elements_sec[2,13] = 0.33 # Al

# 5 pyrite
elements_sec[5,5] = 0.0 # Ca
elements_sec[5,6] = 0.0 # Mg
elements_sec[5,7] = 0.0 # Na
elements_sec[5,8] = 0.0 # K
elements_sec[5,9] = 1.0 # Fe
elements_sec[5,10] = 2.0 # S
elements_sec[5,11] = 0.0 # Si
elements_sec[5,12] = 0.0 # Cl
elements_sec[5,13] = 0.0 # Al

# saponite_na
elements_sec[11,5] = 0.0 # Ca
elements_sec[11,6] = 3.0 # Mg
elements_sec[11,7] = 0.33 # Na
elements_sec[11,8] = 0.0 # K
elements_sec[11,9] = 0.0 # Fe
elements_sec[11,10] = 0.0 # S
elements_sec[11,11] = 3.67 # Si
elements_sec[11,12] = 0.0 # Cl
elements_sec[11,13] = 0.33 # Al

# 13 nont_mg
elements_sec[13,5] = 0.0 # Ca
elements_sec[13,6] = 0.165 # Mg
elements_sec[13,7] = 0.0 # Na
elements_sec[13,8] = 0.0 # K
elements_sec[13,9] = 2.0 # Fe
elements_sec[13,10] = 0.0 # S
elements_sec[13,11] = 3.67 # Si
elements_sec[13,12] = 0.0 # Cl
elements_sec[13,13] = 0.33 # Al

# 14 fe_celad
elements_sec[14,5] = 0.0 # Ca
elements_sec[14,6] = 0.0 # Mg
elements_sec[14,7] = 0.0 # Na
elements_sec[14,8] = 1.0 # K
elements_sec[14,9] = 1.0 # Fe
elements_sec[14,10] = 0.0 # S
elements_sec[14,11] = 4.0 # Si
elements_sec[14,12] = 0.0 # Cl
elements_sec[14,13] = 1.0 # Al

# 16 mesolite
elements_sec[16,5] = 0.657 # Ca
elements_sec[16,6] = 0.0 # Mg
elements_sec[16,7] = 0.676 # Na
elements_sec[16,8] = 0.0 # K
elements_sec[16,9] = 0.0 # Fe
elements_sec[16,10] = 0.0 # S
elements_sec[16,11] = 3.01 # Si
elements_sec[16,12] = 0.0 # Cl
elements_sec[16,13] = 1.99 # Al

# 17 hematite
elements_sec[17,5] = 0.0 # Ca
elements_sec[17,6] = 0.0 # Mg
elements_sec[17,7] = 0.0 # Na
elements_sec[17,8] = 0.0 # K
elements_sec[17,9] = 2.0 # Fe
elements_sec[17,10] = 0.0 # S
elements_sec[17,11] = 0.0 # Si
elements_sec[17,12] = 0.0 # Cl
elements_sec[17,13] = 0.0 # Al

# clinochlore14a
elements_sec[31,5] = 0.0 # Ca
elements_sec[31,6] = 5.0 # Mg
elements_sec[31,7] = 0.0 # Na
elements_sec[31,8] = 0.0 # K
elements_sec[31,9] = 0.0 # Fe
elements_sec[31,10] = 0.0 # S
elements_sec[31,11] = 3.0 # Si
elements_sec[31,12] = 0.0 # Cl
elements_sec[31,13] = 2.0 # Al

# saponite_ca
elements_sec[33,5] = 0.165 # Ca
elements_sec[33,6] = 3.0 # Mg
elements_sec[33,7] = 0.0 # Na
elements_sec[33,8] = 0.0 # K
elements_sec[33,9] = 0.0 # Fe
elements_sec[33,10] = 0.0 # S
elements_sec[33,11] = 3.67 # Si
elements_sec[33,12] = 0.0 # Cl
elements_sec[33,13] = 0.33 # Al


#hack: 2D arrays go here
value_2d_alt_vol_sum = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
value_2d_alt_vol_mean_slope = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
value_2d_fe_mean_slope = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])

value_2d_pri_mean_slope = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])
value_2d_sec_mean_slope = np.zeros([steps,len(param_t_diff_string),len(param_sim_string)])

value_2d_net_uptake_x = np.zeros([steps,15,len(param_t_diff_string),len(param_sim_string)])

#todo: loop through param_t_diff
for ii in range(len(param_t_diff)):
    for iii in range(len(param_sim)):

        print " "
        print " "
        print "param_t_diff:" , param_t_diff_string[ii]
        print "param_sim:" , param_sim_string[iii]

        # ii_path goes here
        ii_path = batch_path + prefix_string + param_sim_string[iii] + '_' + param_t_diff_string[ii] + suffix_string
        print "ii_path: " , ii_path



        any_min = []

        #hack: load in chem data
        # if ii == 0:

        ch_path = ii_path + 'ch_s/'
        for j in range(1,minNum):
            if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
                if not np.any(any_min == j):
                    any_min = np.append(any_min,j)
                #print j , secondary[j] ,
                secMat[:,:,j,ii,iii] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
                secMat[:,:,j,ii,iii] = secMat[:,:,j,ii,iii]*molar[j]/density[j]
                dsecMat[:,2*len(xCell):,j,ii,iii] = secMat[:,len(xCell):-len(xCell),j,ii,iii] - secMat[:,2*len(xCell):,j,ii,iii]
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
        priMat[:,:,ii,iii] = glass0
        dpriMat[:,2*len(xCell):,ii,iii] = priMat[:,len(xCell):-len(xCell),ii,iii] - priMat[:,2*len(xCell):,ii,iii]

        pri_total0 = glass0
        print " "

        # if ii > 0:
        #
        #     secMat[:,:,:,ii,iii] = secMat[:,:,:,ii-1,iii-1]


        ch_path = ii_path + 'ch_a/'
        for j in range(1,minNum):
            if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
                if not np.any(any_min == j):
                    any_min = np.append(any_min,j)
                #print j , secondary[j] ,
                secMat_a[:,:,j,ii,iii] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
                secMat_a[:,:,j,ii,iii] = secMat_a[:,:,j,ii,iii]*molar[j]/density[j]
                dsecMat_a[:,2*len(xCell):,j,ii,iii] = secMat_a[:,len(xCell):-len(xCell),j,ii,iii] - secMat_a[:,2*len(xCell):,j,ii,iii]
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
        priMat_a[:,:,ii,iii] = glass0_a
        dpriMat_a[:,2*len(xCell):,ii,iii] = priMat_a[:,len(xCell):-len(xCell),ii,iii] - priMat_a[:,2*len(xCell):,ii,iii]

        pri_total0_a = glass0_a
        print " "


        ch_path = ii_path + 'ch_b/'
        for j in range(1,minNum):
            if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
                if not np.any(any_min == j):
                    any_min = np.append(any_min,j)
                #print j , secondary[j] ,
                secMat_b[:,:,j,ii,iii] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
                secMat_b[:,:,j,ii,iii] = secMat_b[:,:,j,ii,iii]*molar[j]/density[j]
                dsecMat_a[:,2*len(xCell):,j,ii,iii] = secMat_b[:,len(xCell):-len(xCell),j,ii,iii] - secMat_b[:,2*len(xCell):,j,ii,iii]
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
        priMat_b[:,:,ii,iii] = glass0_b
        dpriMat_b[:,2*len(xCell):,ii,iii] = priMat_b[:,len(xCell):-len(xCell),ii,iii] - priMat_b[:,2*len(xCell):,ii,iii]

        pri_total0_b = glass0_b
        print " "



        ch_path = ii_path + 'ch_d/'
        for j in range(1,minNum):
            if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
                if not np.any(any_min == j):
                    any_min = np.append(any_min,j)
                #print j , secondary[j] ,
                secMat_d[:,:,j,ii,iii] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
                secMat_d[:,:,j,ii,iii] = secMat_d[:,:,j,ii,iii]*molar[j]/density[j]
                dsecMat_d[:,2*len(xCell):,j,ii,iii] = secMat_d[:,len(xCell):-len(xCell),j,ii,iii] - secMat_d[:,2*len(xCell):,j,ii,iii]
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
        priMat_d[:,:,ii,iii] = glass0_d
        dpriMat_d[:,2*len(xCell):,ii,iii] = priMat_d[:,len(xCell):-len(xCell),ii,iii] - priMat_d[:,2*len(xCell):,ii,iii]

        pri_total0_d = glass0_d
        print " "
        print "any_min"
        for j in range(len(any_min)):
            print any_min[j], secondary[any_min[j]]




        for j in range(len(xCell)*steps):
            for k in range(len(yCell)):
                if pri_total0[k,j] > 0.0:
                    alt_vol0[k,j,ii,iii] = np.sum(secMat[k,j,:,ii,iii])/(pri_total0[k,j]+np.sum(secMat[k,j,:,ii,iii]))


        for j in range(len(xCell)*steps):
            for k in range(len(yCell)):
                if pri_total0_a[k,j] > 0.0:
                    alt_vol0_a[k,j,ii,iii] = np.sum(secMat_a[k,j,:,ii,iii])/(pri_total0_a[k,j]+np.sum(secMat_a[k,j,:,ii,iii]))


        for j in range(len(xCell)*steps):
            for k in range(len(yCell)):
                if pri_total0_b[k,j] > 0.0:
                    alt_vol0_b[k,j,ii,iii] = np.sum(secMat_b[k,j,:,ii,iii])/(pri_total0_b[k,j]+np.sum(secMat_b[k,j,:,ii,iii]))


        for j in range(len(xCell)*steps):
            for k in range(len(yCell)):
                    if pri_total0_d[k,j] > 0.0:
                        alt_vol0_d[k,j,ii,iii] = np.sum(secMat_d[k,j,:,ii,iii])/(pri_total0_d[k,j]+np.sum(secMat_d[k,j,:,ii,iii]))


        xd_move = 0
        #todo: loop through steps
        for i in range(0,max_step,1):

            if np.any(moves== i + restart):
                xd_move = xd_move + 1

            #hack: cut up chem data
            for j in range(len(any_min)):
                secStep[:,:,any_min[j],i,ii,iii] = cut_chem(secMat[:,:,any_min[j],ii,iii],i)
                dsecStep[:,:,any_min[j],i,ii,iii] = cut_chem(dsecMat[:,:,any_min[j],ii,iii],i)
                secStep_ts[i,any_min[j],ii,iii] = np.sum(secStep[:,:,any_min[j],i,ii,iii])
                x_secStep_ts[i,any_min[j],ii,iii] = np.sum(secStep[:,xd_move,any_min[j],i,ii,iii])
                if i > 0:
                    dsecStep_ts[i,any_min[j],ii,iii] = secStep_ts[i,any_min[j],ii,iii] - secStep_ts[i-1,any_min[j],ii,iii]
                    x_dsecStep_ts[i,any_min[j],ii,iii] = x_secStep_ts[i,any_min[j],ii,iii] - x_secStep_ts[i-1,any_min[j],ii,iii]
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
            alt_vol[:,:,i,ii,iii] = cut_chem(alt_vol0[:,:,ii,iii],i)
            pri_total = cut_chem(pri_total0,i)

            priStep[:,:,i,ii,iii] = cut_chem(priMat[:,:,ii,iii],i)
            dpriStep[:,:,i,ii,iii] = cut_chem(dpriMat[:,:,ii,iii],i)
            priStep_ts[i,ii,iii] = np.sum(priStep[:,:,i,ii,iii])
            x_priStep_ts[i,ii,iii] = np.sum(priStep[:,xd_move,i,ii,iii])

            if i > 0:
                dpriStep_ts[i,ii,iii] = priStep_ts[i,ii,iii] - priStep_ts[i-1,ii,iii]
                x_dpriStep_ts[i,ii,iii] = x_priStep_ts[i,ii,iii] - x_priStep_ts[i-1,ii,iii]





            for j in range(len(any_min)):
                secStep_a[:,:,any_min[j],i,ii,iii] = cut_chem(secMat_a[:,:,any_min[j],ii,iii],i)
                dsecStep_a[:,:,any_min[j],i,ii,iii] = cut_chem(dsecMat_a[:,:,any_min[j],ii,iii],i)
                secStep_ts_a[i,any_min[j],ii,iii] = np.sum(secStep_a[:,:,any_min[j],i,ii,iii])
                x_secStep_ts_a[i,any_min[j],ii,iii] = np.sum(secStep_a[:,xd_move,any_min[j],i,ii,iii])
                if i > 0:
                    dsecStep_ts_a[i,any_min[j],ii,iii] = secStep_ts_a[i,any_min[j],ii,iii] - secStep_ts_a[i-1,any_min[j],ii,iii]
                    x_dsecStep_ts_a[i,any_min[j],ii,iii] = x_secStep_ts_a[i,any_min[j],ii,iii] - x_secStep_ts_a[i-1,any_min[j],ii,iii]
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
            alt_vol_a[:,:,i,ii,iii] = cut_chem(alt_vol0_a[:,:,ii,iii],i)
            pri_total_a = cut_chem(pri_total0,i)

            priStep_a[:,:,i,ii,iii] = cut_chem(priMat_a[:,:,ii,iii],i)
            dpriStep_a[:,:,i,ii,iii] = cut_chem(dpriMat_a[:,:,ii,iii],i)
            priStep_ts_a[i,ii,iii] = np.sum(priStep_a[:,:,i,ii,iii])
            x_priStep_ts_a[i,ii,iii] = np.sum(priStep_a[:,xd_move,i,ii,iii])

            if i > 0:
                dpriStep_ts_a[i,ii,iii] = priStep_ts_a[i,ii,iii] - priStep_ts_a[i-1,ii,iii]
                x_dpriStep_ts_a[i,ii,iii] = x_priStep_ts_a[i,ii,iii] - x_priStep_ts_a[i-1,ii,iii]





            for j in range(len(any_min)):
                secStep_b[:,:,any_min[j],i,ii,iii] = cut_chem(secMat_b[:,:,any_min[j],ii,iii],i)
                dsecStep_b[:,:,any_min[j],i,ii,iii] = cut_chem(dsecMat_b[:,:,any_min[j],ii,iii],i)
                secStep_ts_b[i,any_min[j],ii,iii] = np.sum(secStep_b[:,:,any_min[j],i,ii,iii])
                x_secStep_ts_b[i,any_min[j],ii,iii] = np.sum(secStep_b[:,xd_move,any_min[j],i,ii,iii])
                if i > 0:
                    dsecStep_ts_b[i,any_min[j],ii,iii] = secStep_ts_b[i,any_min[j],ii,iii] - secStep_ts_b[i-1,any_min[j],ii,iii]
                    x_dsecStep_ts_b[i,any_min[j],ii,iii] = x_secStep_ts_b[i,any_min[j],ii,iii] - x_secStep_ts_b[i-1,any_min[j],ii,iii]
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
            alt_vol_b[:,:,i,ii,iii] = cut_chem(alt_vol0_b[:,:,ii,iii],i)
            pri_total_b = cut_chem(pri_total0,i)

            priStep_b[:,:,i,ii,iii] = cut_chem(priMat_b[:,:,ii,iii],i)
            dpriStep_b[:,:,i,ii,iii] = cut_chem(dpriMat_b[:,:,ii,iii],i)
            priStep_ts_b[i,ii,iii] = np.sum(priStep_b[:,:,i,ii,iii])
            x_priStep_ts_b[i,ii,iii] = np.sum(priStep_b[:,xd_move,i,ii,iii])

            if i > 0:
                dpriStep_ts_b[i,ii,iii] = priStep_ts_b[i,ii,iii] - priStep_ts_b[i-1,ii,iii]
                x_dpriStep_ts_b[i,ii,iii] = x_priStep_ts_b[i,ii,iii] - x_priStep_ts_b[i-1,ii,iii]





            for j in range(len(any_min)):
                secStep_d[:,:,any_min[j],i,ii,iii] = cut_chem(secMat_d[:,:,any_min[j],ii,iii],i)
                dsecStep_d[:,:,any_min[j],i,ii,iii] = cut_chem(dsecMat_d[:,:,any_min[j],ii,iii],i)
                secStep_ts_d[i,any_min[j],ii,iii] = np.sum(secStep_d[:,:,any_min[j],i,ii,iii])
                x_secStep_ts_d[i,any_min[j],ii,iii] = np.sum(secStep_d[:,xd_move,any_min[j],i,ii,iii])
                if i > 0:
                    dsecStep_ts_d[i,any_min[j],ii,iii] = secStep_ts_d[i,any_min[j],ii,iii] - secStep_ts_d[i-1,any_min[j],ii,iii]
                    x_dsecStep_ts_d[i,any_min[j],ii,iii] = x_secStep_ts_d[i,any_min[j],ii,iii] - x_secStep_ts_d[i-1,any_min[j],ii,iii]
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
            alt_vol_d[:,:,i,ii,iii] = cut_chem(alt_vol0_d[:,:,ii,iii],i)
            pri_total_d = cut_chem(pri_total0,i)

            priStep_d[:,:,i,ii,iii] = cut_chem(priMat_d[:,:,ii,iii],i)
            dpriStep_d[:,:,i,ii,iii] = cut_chem(dpriMat_d[:,:,ii,iii],i)
            priStep_ts_d[i,ii,iii] = np.sum(priStep_d[:,:,i,ii,iii])
            x_priStep_ts_d[i,ii,iii] = np.sum(priStep_d[:,xd_move,i,ii,iii])

            if i > 0:
                dpriStep_ts_d[i,ii,iii] = priStep_ts_d[i,ii,iii] - priStep_ts_d[i-1,ii,iii]
                x_dpriStep_ts_d[i,ii,iii] = x_priStep_ts_d[i,ii,iii] - x_priStep_ts_d[i-1,ii,iii]







            #hack: alt_vol data
            for j in range(len(xCell)):
                # full column average

                if ii == 0:
                    above_zero = alt_vol[:,j,i,ii,iii]*100.0
                    above_zero = above_zero[above_zero>0.0]
                    alt_col_mean[j,i,len(param_t_diff_string)] = np.mean(above_zero)

                    # pri_mean[j,i,ii,iii] = np.sum(pri_total[:,j])/7.0

                above_zero = alt_vol_d[:,j,i,ii,iii]*100.0
                above_zero = above_zero[above_zero>0.0]
                #print above_zero
                alt_col_mean[j,i,ii,iii] = np.mean(above_zero)

                pri_mean_d[j,i,ii,iii] = np.sum(priStep_d[j,:,i,ii,iii])/7.0
                sec_mean_d[j,i,ii,iii] = np.sum(secStep_d[j,:,:,i,ii,iii])/7.0

                if j > 0 and alt_col_mean[j-1,i,ii,iii] > 0.0 and alt_col_mean[j,i,ii,iii] > 0.0:
                    alt_col_mean_slope[j,i,ii,iii] = alt_col_mean[j,i,ii,iii] - alt_col_mean[j-1,i,ii,iii]

                if j > 0 and np.abs(pri_mean_d[j-1,i,ii,iii]) > 0.0 and np.abs(pri_mean_d[j,i,ii,iii]) > 0.0:
                    pri_mean_slope[j,i,ii,iii] = np.abs(pri_mean_d[j,i,ii,iii]) - np.abs(pri_mean_d[j-1,i,ii,iii])

                if j > 0 and np.abs(sec_mean_d[j-1,i,ii,iii]) > 0.0 and np.abs(sec_mean_d[j,i,ii,iii]):
                    sec_mean_slope[j,i,ii,iii] = sec_mean_d[j,i,ii,iii] - sec_mean_d[j-1,i,ii,iii]



            alt_col_mean_slope[alt_col_mean_slope==0.0] = None
            value_2d_alt_vol_mean_slope[i,ii,iii] = np.nanmean(alt_col_mean_slope[:,i,ii,iii])

            #pri_mean_slope[pri_mean_slope==0.0] = None
            value_2d_pri_mean_slope[i,ii,iii] = np.sum(pri_mean_slope[:,i,ii,iii])/float(max_step)
            print value_2d_pri_mean_slope[i,ii,iii]
            value_2d_sec_mean_slope[i,ii,iii] = np.sum(sec_mean_slope[:,i,ii,iii])/float(max_step)
            # print "time" , i , "2d_value" , value_2d_alt_vol_mean_slope[i,ii,iii]
            # print "slope array" , alt_col_mean_slope[:,i,ii,iii]
            # print "value array" , alt_col_mean[:,i,ii,iii]
            # print " "





            #hack: FeO / FeOt data
            for j in range(len(xCell)):

                feo_col_mean_temp = np.zeros(len(xCell))
                feot_col_mean_temp = np.zeros(len(xCell))

                if ii == 0:


                    secStep_temp = secStep[:,:,:,i,ii,iii]
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
                        feot_col_mean_temp[j] = 0.8998*.026*2.0*np.mean(glass_temp[above_zero_ind,j])*(density_pri[3]/molar_pri[3])
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


                secStep_temp = secStep_d[:,:,:,i,ii,iii]
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
                    feot_col_mean_temp[j] = 0.8998*.026*2.0*np.mean(glass_temp[above_zero_ind,j])*(density_pri[3]/molar_pri[3])
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

                    fe_col_mean[j,i,ii,iii] = feo_col_mean_temp[j] / (feo_col_mean_temp[j] + feot_col_mean_temp[j])

                    if j > 0 and np.abs(fe_col_mean[j-1,i,ii,iii]) > 0.0:
                        fe_col_mean_slope[j,i,ii,iii] = fe_col_mean[j,i,ii,iii] - fe_col_mean[j-1,i,ii,iii]

            fe_col_mean_slope[alt_col_mean_slope==0.0] = None
            value_2d_fe_mean_slope[i,ii,iii] = np.nanmean(fe_col_mean_slope[:,i,ii,iii])
            # print "time" , i , "2d_value fe" , value_2d_fe_mean_slope[i,ii,iii]





            #hack: net uptake data
            for j in range(len(any_min)):
                for jj in range(15):

                    # x_elements[i,jj] = x_elements[i,jj] + elements_sec[any_min[j],jj]*np.sum(dsecStep[:,xd_move,any_min[j]])*(density[any_min[j]]/molar[any_min[j]])
                    # # if j == 0:
                    # #     x_elements[i,jj] = x_elements[i,jj] - elements_pri[0,jj]*np.sum(dpriStep[:,xd_move,0])*(density_pri[3]/molar_pri[3])
                    #
                    # x_elements_d[i,jj] = x_elements_d[i,jj] + elements_sec[any_min[j],jj]*np.sum(dsecStep_d[:,xd_move,any_min[j]])*(density[any_min[j]]/molar[any_min[j]])
                    # # if j == 0:
                    # #     x_elements_d[i,jj] = x_elements_d[i,jj] - elements_pri[0,jj]*np.sum(dpriStep_d[:,xd_move,0])*(density_pri[3]/molar_pri[3])
                    #
                    # x_elements_a[i,jj] = x_elements_a[i,jj] + elements_sec[any_min[j],jj]*np.sum(dsecStep_a[:,xd_move,any_min[j]])*(density[any_min[j]]/molar[any_min[j]])
                    # # if j == 0:
                    # #     x_elements_a[i,jj] = x_elements_a[i,jj] - elements_pri[0,jj]*np.sum(dpriStep_a[:,xd_move,0])*(density_pri[3]/molar_pri[3])
                    #
                    # x_elements_b[i,jj] = x_elements_b[i,jj] + elements_sec[any_min[j],jj]*np.sum(dsecStep_b[:,xd_move,any_min[j]])*(density[any_min[j]]/molar[any_min[j]])
                    # # if j == 0:
                    # #     x_elements_b[i,jj] = x_elements_b[i,jj] - elements_pri[0,jj]*np.sum(dpriStep_b[:,xd_move,0])*(density_pri[3]/molar_pri[3])

                    x_elements[i,jj,ii,iii] = x_elements[i,jj,ii,iii] + elements_sec[any_min[j],jj]*x_dsecStep_ts[i,any_min[j],ii,iii]*(density[any_min[j]]/molar[any_min[j]])
                    x_elements_d[i,jj,ii,iii] = x_elements_d[i,jj,ii,iii] + elements_sec[any_min[j],jj]*x_dsecStep_ts_d[i,any_min[j],ii,iii]*(density[any_min[j]]/molar[any_min[j]])
                    x_elements_a[i,jj,ii,iii] = x_elements_a[i,jj,ii,iii] + elements_sec[any_min[j],jj]*x_dsecStep_ts_a[i,any_min[j],ii,iii]*(density[any_min[j]]/molar[any_min[j]])
                    x_elements_b[i,jj,ii,iii] = x_elements_b[i,jj,ii,iii] + elements_sec[any_min[j],jj]*x_dsecStep_ts_b[i,any_min[j],ii,iii]*(density[any_min[j]]/molar[any_min[j]])

                    # if j == 0:
                    #     x_elements[i,jj] = x_elements[i,jj] + elements_pri[0,jj]*x_dpriStep_ts[i,5]#*(density_pri[3]/molar_pri[3])
                    #     x_elements_d[i,jj] = x_elements_d[i,jj] + elements_pri[0,jj]*x_dpriStep_ts_d[i,5]#*(density_pri[3]/molar_pri[3])
                    #     x_elements_a[i,jj] = x_elements_a[i,jj] + elements_pri[0,jj]*x_dpriStep_ts_a[i,5]#*(density_pri[3]/molar_pri[3])
                    #     x_elements_b[i,jj] = x_elements_b[i,jj] + elements_pri[0,jj]*x_dpriStep_ts_b[i,5]#*(density_pri[3]/molar_pri[3])

                    if j == 0:
                        x_pri_elements[i,jj,ii,iii] = elements_pri[0,jj]*x_dpriStep_ts[i,ii,iii]*(density_pri[3]/molar_pri[3])
                        x_pri_elements_d[i,jj,ii,iii] = elements_pri[0,jj]*x_dpriStep_ts_d[i,ii,iii]*(density_pri[3]/molar_pri[3])
                        x_pri_elements_a[i,jj,ii,iii] = elements_pri[0,jj]*x_dpriStep_ts_a[i,ii,iii]*(density_pri[3]/molar_pri[3])
                        x_pri_elements_b[i,jj,ii,iii] = elements_pri[0,jj]*x_dpriStep_ts_b[i,ii,iii]*(density_pri[3]/molar_pri[3])









for iii in range(len(param_sim)):

    #todo: FIGURE: batch_alt, one per sim
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

        plt.plot(xCell,alt_col_mean[:,max_step-1,ii,iii],color=plot_col[ii],lw=1.5, label=plot_t_diff_strings[ii])


    plt.legend(fontsize=10,loc=2,labelspacing=-0.1)
    plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=12)
    plt.xlim([0.0, 90000.0])
    plt.xlabel('Distance along transect [km]', fontsize=9)

    plt.yticks([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],fontsize=12)
    plt.ylim([0.0, 30.0])
    plt.ylabel('Alteration volume $\%$')
    plt.title('sim:' + param_sim_string[iii])




    #todo: FIGURE: FeO / FeOt plot

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

        plt.plot(xCell,fe_col_mean[:,max_step-1,ii,iii],color=plot_col[ii],lw=1.5, label=plot_t_diff_strings[ii])

    #plt.legend(fontsize=10)
    plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=12)
    plt.xlim([0.0, 90000.0])
    plt.xlabel('Distance along transect [km]', fontsize=9)

    #plt.yticks([0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80])
    plt.yticks([0.6, 0.65, 0.7, 0.75, 0.80])
    #plt.ylim([0.6, 0.8])
    plt.ylim([0.6, 0.8])
    plt.ylabel('FeO / FeOt')


    plt.savefig(batch_path+prefix_string+"sim_"+param_sim_string[iii]+"_alt.png",bbox_inches='tight')







    #todo: FIGURE: x_pri figure
    fig=plt.figure(figsize=(10.0,3.0))

    ax=fig.add_subplot(1, 3, 1, frameon=True)
    for ii in range(len(param_t_diff)):
        plt.plot(range(steps),x_priStep_ts_d[:,ii,iii]/np.max(x_priStep_ts_d[:,ii,iii]),color=plot_col[ii],lw=1.5, label=plot_t_diff_strings[ii])
    plt.plot(range(steps),x_priStep_ts[:,0,0]/np.max(x_priStep_ts[:,0,0]),color=plot_col[len(param_t_diff)],lw=1.5, label=plot_t_diff_strings[len(param_t_diff)])

    temp_pri_min_mat = x_priStep_ts_d[:,:,iii]/np.max(x_priStep_ts_d[:,:,iii])
    temp_pri_min = np.min(temp_pri_min_mat[temp_pri_min_mat>0.0])
    plt.ylim([temp_pri_min,1.01])
    plt.legend(fontsize=8,labelspacing=-0.1,columnspacing=0.0)
    plt.title('dual vs solo')



    ax=fig.add_subplot(1, 3, 2, frameon=True)
    for ii in range(len(param_t_diff)):
        plt.plot(range(steps),x_priStep_ts_a[:,ii,iii]/np.max(x_priStep_ts_a[:,ii,iii]),color=plot_col[ii],lw=1.5, label=plot_t_diff_strings[ii])

    temp_pri_min_mat = x_priStep_ts_a[:,:,iii]/np.max(x_priStep_ts_a[:,:,iii])
    temp_pri_min = np.min(temp_pri_min_mat[temp_pri_min_mat>0.0])
    plt.ylim([temp_pri_min,1.01])
    plt.title('a only')



    ax=fig.add_subplot(1, 3, 3, frameon=True)
    for ii in range(len(param_t_diff)):
        plt.plot(range(steps),x_priStep_ts_b[:,ii,iii]/np.max(x_priStep_ts_b[:,ii,iii]),color=plot_col[ii],lw=1.5, label=plot_t_diff_strings[ii])

    temp_pri_min_mat = x_priStep_ts_b[:,:,iii]/np.max(x_priStep_ts_b[:,:,iii])
    temp_pri_min = np.min(temp_pri_min_mat[temp_pri_min_mat>0.0])
    plt.ylim([temp_pri_min,1.01])
    plt.title('b only')


    # for ii in range(len(param_t_diff)):
    #     print x_priStep_ts[:,ii]
    #     print " "

    plt.savefig(batch_path+prefix_string+"sim_"+param_sim_string[iii]+"_pri.png",bbox_inches='tight')






    # elements_sec[,5] = 0.0 # Ca
    # elements_sec[,6] = 0.0 # Mg
    # elements_sec[,7] = 0.0 # Na
    # elements_sec[,8] = 0.0 # K
    # elements_sec[,9] = 0.0 # Fe
    # elements_sec[,10] = 0.0 # S
    # elements_sec[,11] = 0.0 # Si
    # elements_sec[,12] = 0.0 # Cl
    # elements_sec[,13] = 0.0 # Al



    #todo: FIGURE: net_uptake_x
    fig=plt.figure(figsize=(15.0,8.0))
    print "net_uptake_x"

    net_uptake_kwargs = dict(lw=1.1)
    net_uptake_kwargs_a = dict(lw=1.1, linestyle='-')
    net_uptake_kwargs_b = dict(lw=1.1, linestyle='-')
    uptake_color_s = '#bd3706'
    uptake_color_d = '#073dc7'
    uptake_color_a = '#0793c7'
    uptake_color_b = '#0fe7e0'
    xt_fs = 8


    ax=fig.add_subplot(3, 4, 1, frameon=True)

    for ii in range(len(param_t_diff)):
        plt.plot(np.arange(steps),x_elements_d[:,6,ii,iii]+x_pri_elements_d[:,6,ii,iii],color=plot_col[ii], label=plot_t_diff_strings[ii], **net_uptake_kwargs)
    plt.plot(range(steps),x_elements[:,6,0,0]+x_pri_elements[:,6,0,0],color=plot_col[len(param_t_diff)],label=plot_t_diff_strings[len(param_t_diff)], **net_uptake_kwargs)
    plt.legend(fontsize=8,loc='best',labelspacing=-0.1,columnspacing=0.0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('dual vs solo Mg total uptake',fontsize=10)



    ax=fig.add_subplot(3, 4, 2, frameon=True)

    for ii in range(len(param_t_diff)):
        plt.plot(np.arange(steps),x_elements_d[:,5,ii,iii]+x_pri_elements_d[:,5,ii,iii],color=plot_col[ii], label=plot_t_diff_strings[ii], **net_uptake_kwargs)
    plt.plot(range(steps),x_elements[:,5,0,0]+x_pri_elements[:,5,0,0],color=plot_col[len(param_t_diff)],label=plot_t_diff_strings[len(param_t_diff)], **net_uptake_kwargs)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('dual vs solo Ca total uptake',fontsize=10)



    ax=fig.add_subplot(3, 4, 3, frameon=True)

    for ii in range(len(param_t_diff)):
        plt.plot(np.arange(steps),x_elements_d[:,8,ii,iii]+x_pri_elements_d[:,8,ii,iii],color=plot_col[ii], label=plot_t_diff_strings[ii], **net_uptake_kwargs)
    plt.plot(range(steps),x_elements[:,8,0,0]+x_pri_elements[:,8,0,0],color=plot_col[len(param_t_diff)],label=plot_t_diff_strings[len(param_t_diff)], **net_uptake_kwargs)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('dual vs solo K total uptake',fontsize=10)



    ax=fig.add_subplot(3, 4, 4, frameon=True)

    for ii in range(len(param_t_diff)):
        plt.plot(np.arange(steps),x_elements_d[:,9,ii,iii]+x_pri_elements_d[:,9,ii,iii],color=plot_col[ii], label=plot_t_diff_strings[ii], **net_uptake_kwargs)
    plt.plot(range(steps),x_elements[:,9,0,0]+x_pri_elements[:,9,0,0],color=plot_col[len(param_t_diff)],label=plot_t_diff_strings[len(param_t_diff)], **net_uptake_kwargs)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('dual vs solo Fe total uptake',fontsize=10)


        # plt.plot(np.arange(steps),x_elements[:,6],color=uptake_color_s,label='solo', **net_uptake_kwargs)
        # plt.plot(np.arange(steps),x_elements_d[:,6],color=uptake_color_d,label='dual', **net_uptake_kwargs)
        # plt.plot(np.arange(steps),x_elements_a[:,6],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
        # plt.plot(np.arange(steps),x_elements_b[:,6],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
        # plt.legend(fontsize=9,loc='best',ncol=2,labelspacing=0.0,columnspacing=0.0)
        # plt.xticks(fontsize=xt_fs)
        # plt.yticks(fontsize=xt_fs)
        # plt.title('Mg uptake in column')
        #
        #
        # ax=fig.add_subplot(3, 4, 2, frameon=True)
        # plt.plot(np.arange(steps),x_elements[:,5],color=uptake_color_s,label='solo', **net_uptake_kwargs)
        # plt.plot(np.arange(steps),x_elements_d[:,5],color=uptake_color_d,label='dual', **net_uptake_kwargs)
        # plt.plot(np.arange(steps),x_elements_a[:,5],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
        # plt.plot(np.arange(steps),x_elements_b[:,5],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
        # plt.xticks(fontsize=xt_fs)
        # plt.yticks(fontsize=xt_fs)
        # plt.title('Ca uptake in column')
        #
        #
        # ax=fig.add_subplot(3, 4, 3, frameon=True)
        # plt.plot(np.arange(steps),x_elements[:,8],color=uptake_color_s,label='solo', **net_uptake_kwargs)
        # plt.plot(np.arange(steps),x_elements_d[:,8],color=uptake_color_d,label='dual', **net_uptake_kwargs)
        # plt.plot(np.arange(steps),x_elements_a[:,8],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
        # plt.plot(np.arange(steps),x_elements_b[:,8],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
        # plt.xticks(fontsize=xt_fs)
        # plt.yticks(fontsize=xt_fs)
        # plt.title('K uptake in column')
        #
        #
        # ax=fig.add_subplot(3, 4, 4, frameon=True)
        # plt.plot(np.arange(steps),x_elements[:,9],color=uptake_color_s,label='solo', **net_uptake_kwargs)
        # plt.plot(np.arange(steps),x_elements_d[:,9],color=uptake_color_d,label='dual', **net_uptake_kwargs)
        # plt.plot(np.arange(steps),x_elements_a[:,9],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
        # plt.plot(np.arange(steps),x_elements_b[:,9],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
        # plt.xticks(fontsize=xt_fs)
        # plt.yticks(fontsize=xt_fs)
        # plt.title('Fe uptake in column')




    # ax=fig.add_subplot(3, 4, 5, frameon=True)
    # plt.plot(np.arange(steps),x_pri_elements[:,6],color=uptake_color_s,label='solo', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_pri_elements_d[:,6],color=uptake_color_d,label='dual', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_pri_elements_a[:,6],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
    # plt.plot(np.arange(steps),x_pri_elements_b[:,6],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
    # plt.legend(fontsize=9,loc='best',ncol=2,labelspacing=0.0,columnspacing=0.0)
    # plt.xticks(fontsize=xt_fs)
    # plt.yticks(fontsize=xt_fs)
    # plt.title('Mg loss pri')
    #
    # ax=fig.add_subplot(3, 4, 6, frameon=True)
    # plt.plot(np.arange(steps),x_pri_elements[:,5],color=uptake_color_s,label='solo', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_pri_elements_d[:,5],color=uptake_color_d,label='dual', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_pri_elements_a[:,5],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
    # plt.plot(np.arange(steps),x_pri_elements_b[:,5],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
    # plt.xticks(fontsize=xt_fs)
    # plt.yticks(fontsize=xt_fs)
    # plt.title('Ca loss pri')
    #
    # ax=fig.add_subplot(3, 4, 7, frameon=True)
    # plt.plot(np.arange(steps),x_pri_elements[:,8],color=uptake_color_s,label='solo', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_pri_elements_d[:,8],color=uptake_color_d,label='dual', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_pri_elements_a[:,8],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
    # plt.plot(np.arange(steps),x_pri_elements_b[:,8],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
    # plt.xticks(fontsize=xt_fs)
    # plt.yticks(fontsize=xt_fs)
    # plt.title('K loss pri')
    #
    # ax=fig.add_subplot(3, 4, 8, frameon=True)
    # plt.plot(np.arange(steps),x_pri_elements[:,9],color=uptake_color_s,label='solo', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_pri_elements_d[:,9],color=uptake_color_d,label='dual', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_pri_elements_a[:,9],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
    # plt.plot(np.arange(steps),x_pri_elements_b[:,9],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
    # plt.xticks(fontsize=xt_fs)
    # plt.yticks(fontsize=xt_fs)
    # plt.title('Fe loss pri')
    #
    #
    #
    #
    # ax=fig.add_subplot(3, 4, 9, frameon=True)
    # plt.plot(np.arange(steps),x_elements[:,6]+x_pri_elements[:,6],color='#bd3706',label='solo', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_elements_d[:,6]+x_pri_elements_d[:,6],color=uptake_color_d,label='dual', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_elements_a[:,6]+x_pri_elements_a[:,6],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
    # plt.plot(np.arange(steps),x_elements_b[:,6]+x_pri_elements_b[:,6],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
    # plt.legend(fontsize=9,loc='best',ncol=2,labelspacing=0.0,columnspacing=0.0)
    # plt.xticks(fontsize=xt_fs)
    # plt.yticks(fontsize=xt_fs)
    # plt.title('Mg net')
    #
    # ax=fig.add_subplot(3, 4, 10, frameon=True)
    # plt.plot(np.arange(steps),x_elements[:,5]+x_pri_elements[:,5],color='#bd3706',label='solo', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_elements_d[:,5]+x_pri_elements_d[:,5],color=uptake_color_d,label='dual', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_elements_a[:,5]+x_pri_elements_a[:,5],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
    # plt.plot(np.arange(steps),x_elements_b[:,5]+x_pri_elements_b[:,5],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
    # plt.xticks(fontsize=xt_fs)
    # plt.yticks(fontsize=xt_fs)
    # plt.title('Ca net')
    #
    # ax=fig.add_subplot(3, 4, 11, frameon=True)
    # plt.plot(np.arange(steps),x_elements[:,8]+x_pri_elements[:,8],color='#bd3706',label='solo', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_elements_d[:,8]+x_pri_elements_d[:,8],color=uptake_color_d,label='dual', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_elements_a[:,8]+x_pri_elements_a[:,8],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
    # plt.plot(np.arange(steps),x_elements_b[:,8]+x_pri_elements_b[:,8],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
    # plt.xticks(fontsize=xt_fs)
    # plt.yticks(fontsize=xt_fs)
    # plt.title('K net')
    #
    # ax=fig.add_subplot(3, 4, 12, frameon=True)
    # plt.plot(np.arange(steps),x_elements[:,9]+x_pri_elements[:,9],color='#bd3706',label='solo', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_elements_d[:,9]+x_pri_elements_d[:,9],color=uptake_color_d,label='dual', **net_uptake_kwargs)
    # plt.plot(np.arange(steps),x_elements_a[:,9]+x_pri_elements_a[:,9],color=uptake_color_a,label='a only', **net_uptake_kwargs_a)
    # plt.plot(np.arange(steps),x_elements_b[:,9]+x_pri_elements_b[:,9],color=uptake_color_b,label='b only', **net_uptake_kwargs_b)
    # plt.xticks(fontsize=xt_fs)
    # plt.yticks(fontsize=xt_fs)
    # plt.title('Fe net')




    plt.subplots_adjust( wspace=0.3 , hspace=0.3)
    plt.savefig(batch_path+prefix_string+"sim_"+param_sim_string[iii]+"_net.png",bbox_inches='tight')






#todo: 2D pcolor plot
fig=plt.figure(figsize=(16.0,7.0))

ax=fig.add_subplot(2, 4, 1, frameon=True)
plt.pcolor(value_2d_alt_vol_mean_slope[max_step-2,:,:])

the_xticks = range(len(param_sim))
for i in the_xticks:
    the_xticks[i] = the_xticks[i] + 0.5
print "the_xticks" , the_xticks
plt.xticks(the_xticks,param_sim_string, fontsize=8)
the_yticks = range(len(param_t_diff))
for i in the_yticks:
    the_yticks[i] = the_yticks[i] + 0.5
print "the_yticks" , the_yticks
plt.yticks(the_yticks,param_t_diff_string, fontsize=8)
plt.xlabel('primary basalt distribution',fontsize=8)
plt.ylabel('t_diff mixing time [s]',fontsize=8)

cbar = plt.colorbar(orientation='horizontal')
cbar.ax.tick_params(labelsize=8)
plt.title('alt_vol column mean slope')



ax=fig.add_subplot(2,4, 2, frameon=True)
plt.pcolor(np.abs(value_2d_fe_mean_slope[max_step-2,:,:]))

the_xticks = range(len(param_sim))
for i in the_xticks:
    the_xticks[i] = the_xticks[i] + 0.5
print "the_xticks" , the_xticks
plt.xticks(the_xticks,param_sim_string, fontsize=8)
the_yticks = range(len(param_t_diff))
for i in the_yticks:
    the_yticks[i] = the_yticks[i] + 0.5
print "the_yticks" , the_yticks
plt.yticks(the_yticks,param_t_diff_string, fontsize=8)
#plt.yticks([])
plt.xlabel('primary basalt distribution',fontsize=8)
plt.ylabel('t_diff mixing time [s]',fontsize=8)

cbar = plt.colorbar(orientation='horizontal')
cbar.ax.tick_params(labelsize=8)
plt.title('feo/feot column mean slope')







ax=fig.add_subplot(2, 4, 3, frameon=True)
plt.pcolor(np.abs(value_2d_pri_mean_slope[max_step-2,:,:]))

the_xticks = range(len(param_sim))
for i in the_xticks:
    the_xticks[i] = the_xticks[i] + 0.5
print "the_xticks" , the_xticks
plt.xticks(the_xticks,param_sim_string, fontsize=8)
the_yticks = range(len(param_t_diff))
for i in the_yticks:
    the_yticks[i] = the_yticks[i] + 0.5
print "the_yticks" , the_yticks
plt.yticks(the_yticks,param_t_diff_string, fontsize=8)
#plt.yticks([])
plt.xlabel('primary basalt distribution',fontsize=8)
plt.ylabel('t_diff mixing time [s]',fontsize=8)

cbar = plt.colorbar(orientation='horizontal')
cbar.ax.tick_params(labelsize=8)
plt.title('value pri slope')



ax=fig.add_subplot(2, 4, 4, frameon=True)
plt.pcolor(np.abs(value_2d_sec_mean_slope[max_step-2,:,:]))

the_xticks = range(len(param_sim))
for i in the_xticks:
    the_xticks[i] = the_xticks[i] + 0.5
print "the_xticks" , the_xticks
plt.xticks(the_xticks,param_sim_string, fontsize=8)
the_yticks = range(len(param_t_diff))
for i in the_yticks:
    the_yticks[i] = the_yticks[i] + 0.5
print "the_yticks" , the_yticks
plt.yticks(the_yticks,param_t_diff_string, fontsize=8)
#plt.yticks([])
plt.xlabel('primary basalt distribution',fontsize=8)
plt.ylabel('t_diff mixing time [s]',fontsize=8)

cbar = plt.colorbar(orientation='horizontal')
cbar.ax.tick_params(labelsize=8)
plt.title('value sec slope')



ax=fig.add_subplot(2, 4, 5, frameon=True)
plt.pcolor(np.abs(value_2d_sec_mean_slope[max_step-2,:,:])/np.abs(value_2d_pri_mean_slope[max_step-2,:,:]))

the_xticks = range(len(param_sim))
for i in the_xticks:
    the_xticks[i] = the_xticks[i] + 0.5
print "the_xticks" , the_xticks
plt.xticks(the_xticks,param_sim_string, fontsize=8)
the_yticks = range(len(param_t_diff))
for i in the_yticks:
    the_yticks[i] = the_yticks[i] + 0.5
print "the_yticks" , the_yticks
plt.yticks(the_yticks,param_t_diff_string, fontsize=8)
#plt.yticks([])
plt.xlabel('primary basalt distribution',fontsize=8)
plt.ylabel('t_diff mixing time [s]',fontsize=8)

cbar = plt.colorbar(orientation='horizontal')
cbar.ax.tick_params(labelsize=8)
plt.title('value sec slope')



plt.savefig(batch_path+prefix_string+"sum_test_"+prefix_string+".png",bbox_inches='tight')
