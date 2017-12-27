# revived_JDF.py

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
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams['axes.titlesize'] = 10
#plt.rcParams['hatch.linewidth'] = 0.1

plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #31aa22, #04776b"

col = ['#6e0202', '#fc385b', '#ff7411', '#19a702', '#00520d', '#00ffc2', '#609ff2', '#20267c','#8f00ff', '#ec52ff', '#6e6e6e', '#000000', '#c6813a', '#7d4e22', '#ffff00', '#df9a00', '#812700', '#6b3f67', '#0f9995', '#4d4d4d', '#d9d9d9', '#e9acff']

col_sel = ['#880000' , '#159600', '#609ff2']

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
# sap 480.19
# cel 429.02
# mont 549.07
# nont 495.90
# verm 504.19
# chlor 64.448

print secondary.shape
print density.shape
print molar.shape

# molar_pri = np.array([110.0, 153.0, 158.81, 277.0])
molar_pri = np.array([277.0, 153.0, 158.81, 110.0])

density_pri = np.array([2.7, 3.0, 3.0, 3.0])

##############
# INITIALIZE #
##############

steps = 50
corr = 0
minNum = 41
ison=10000
trace = 0
chem = 1
iso = 0
cell = 1
cellx = 90
celly = 1
sec_toggle = 0
xli = 45
alt_plots = 1
d_only = 0
param_b_g = 1


# all phenos

# pri_2_toggle = 1
# pri_3_toggle = 1
# pri_4_toggle = 1
# pri_5_toggle = 1
#
# pri_toggle = [0, 0, 1, 1, 1, 1]
# pri_toggle_d = [0, 0, 1, 1, 1, 1]
# pri_toggle_a = [0, 0, 1, 1, 1, 0]
# pri_toggle_b = [0, 0, 0, 0, 0, 1]
#



# # glass and olivine
#
# pri_2_toggle = 0
# pri_3_toggle = 0
# pri_4_toggle = 1
# pri_5_toggle = 1
#
# pri_toggle = [0, 0, 0, 0, 1, 1]
# pri_toggle_d = [0, 0, 0, 0, 1, 1]
# pri_toggle_a = [0, 0, 0, 0, 1, 0]
# pri_toggle_b = [0, 0, 0, 0, 0, 1]



# glass/meso only

pri_2_toggle = 0
pri_3_toggle = 0
pri_4_toggle = 0
pri_5_toggle = 1

pri_toggle = [0, 0, 0, 0, 0, 1]
pri_toggle_d = [0, 0, 0, 0, 0, 1]
pri_toggle_a = [0, 0, 0, 0, 0, 1]
pri_toggle_b = [0, 0, 0, 0, 0, 1]


#hack: input path
letter = "10e10"
outpath = "../output/revival/summer_coarse_grid/v_50A_50B_"+letter+"/"

print "LETTER" , letter
#outpath = "../output/revival/summer_coarse_grid/med_b/"
path = outpath
param_w = 300.0
param_w_rhs = 200.0

x0 = np.loadtxt(path + 'x.txt',delimiter='\n')
y0 = np.loadtxt(path + 'y.txt',delimiter='\n')

x=x0
y=y0

asp = np.abs(np.max(x)/np.min(y))/4.0
print asp
bitsx = len(x)
bitsy = len(y)
restart = 1
print "bitsy" , bitsy





dx = float(np.max(x))/float(bitsx)
dy = np.abs(float(np.max(np.abs(y)))/float(bitsy))

xCell = x0[1::cellx]
yCell = y0[0::celly]

print "xcell, ycell len: " , len(xCell) , len(yCell)

xg, yg = np.meshgrid(x[:],y[:])
xgh, ygh = np.meshgrid(x[:],y[:])
xgCell, ygCell = np.meshgrid(xCell[:],yCell[:])

mask = np.loadtxt(path + 'mask.txt')
maskP = np.loadtxt(path + 'maskP.txt')
mask_coarse = np.loadtxt(path + 'mask_coarse.txt')
h_coarse = np.loadtxt(path + 'h_coarse.txt')
u1 = np.loadtxt(path + 'u.txt')
v1 = np.loadtxt(path + 'v.txt')
u_coarse = np.loadtxt(path + 'u_coarse.txt')
v_coarse = np.loadtxt(path + 'v_coarse.txt')
psi_coarse = np.loadtxt(path + 'psi_coarse.txt')
perm = np.loadtxt(path + 'permeability.txt')


perm = np.log10(perm)


u_ts = np.zeros([steps])



#todo: FIGURE: coarse_plot.png
fig=plt.figure()
grd_msh = np.ones(u_coarse[len(y):,:].shape)

ax1=fig.add_subplot(2,2,1,frameon=False)
pgp = plt.pcolor(mask_coarse[:,:],cmap=cm.rainbow,zorder=-10)
plt.title('mask_coarse')
plt.colorbar(pgp,orientation='horizontal')

ax1=fig.add_subplot(2,2,2,frameon=False)
pgp = plt.pcolor(h_coarse,cmap=cm.rainbow,zorder=-10)
plt.title('h_coarse')
plt.colorbar(pgp,orientation='horizontal')

ax1=fig.add_subplot(2,2,3, frameon=False)
u_coarse = u_coarse*3.14e7
pgu = plt.pcolor(u_coarse,cmap=cm.rainbow,zorder=-10)
plt.title('u_coarse')
plt.colorbar(pgu,orientation='horizontal')

ax1=fig.add_subplot(2,2,4, frameon=False)
v_coarse = np.abs(v_coarse)
v_coarse[v_coarse==0.0] = 1.0e-15
v_coarse = np.log10(v_coarse)
pgv = plt.pcolor(v_coarse,cmap=cm.rainbow,zorder=-10)
plt.title('v_coarse')
plt.colorbar(pgv,orientation='horizontal')

fig.savefig(outpath+'coarse_plot.png')




def cut(geo0,index):
    #geo_cut = geo0[(index*len(y0)/cell):(index*len(y0)/cell+len(y0)/cell),:]
    geo_cut = geo0[:,(index*len(x0)):(index*len(x0)+len(x0))]
    geo_cut = np.append(geo_cut, geo_cut[-1:,:], axis=0)
    geo_cut = np.append(geo_cut, geo_cut[:,-1:], axis=1)
    return geo_cut


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
            contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),3)
        if cb_max!=10.0:
            contours = np.linspace(cb_min,cb_max,3)

        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4,frameon=False)
        pGlass = plt.pcolor(xCell,yCell,np.zeros(varStep.shape),cmap=cm.rainbow,vmin=contours[0], vmax=contours[-1])
        plt.yticks([])
        if ytix==1:
            plt.yticks([-450, -400, -350])
        if xtix==0:
            plt.xticks([])
        plt.ylim([np.min(yCell),0.])
        plt.title(cp_title,fontsize=7)
        plt.ylim([-510.0,-320.0])
        if cb==1:
            bbox = ax1.get_position()
            cax = fig.add_axes([bbox.xmin+bbox.width/10.0, bbox.ymin-0.28, bbox.width*0.8, bbox.height*0.13])
            cbar = plt.colorbar(pGlass, cax = cax,orientation='horizontal',ticks=contours[::contour_interval])
            cbar.ax.tick_params(labelsize=7)
            plt.title(cb_title,fontsize=7)
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")

    if np.abs(np.abs(np.max(varStep)) - np.abs(np.min(varStep))) > 0.0:
        if cb_min==-10.0 and cb_max==10.0:
            contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),3)
        if cb_max!=10.0:
            contours = np.linspace(cb_min,cb_max,3)
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
        plt.title(cp_title,fontsize=7)
        plt.ylim([-510.0,-320.0])

        pGlass.set_edgecolor("face")
        if cb==1:
            bbox = ax1.get_position()
            cax = fig.add_axes([bbox.xmin+bbox.width/10.0, bbox.ymin-0.28, bbox.width*0.8, bbox.height*0.13])
            cbar = plt.colorbar(pGlass, cax = cax,orientation='horizontal',ticks=contours[::contour_interval])
            cbar.ax.tick_params(labelsize=7)
            plt.title(cb_title,fontsize=7)
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")
    return chemplot




def chemplot24(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, xtix=1, ytix=1, cb=1, cb_title='', cb_min=-10.0, cb_max=10.0):
    if np.abs(np.abs(np.max(varStep)) - np.abs(np.min(varStep))) <= 0.0:
        if cb_min==-10.0 and cb_max==10.0:
            contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),3)
        if cb_max!=10.0:
            contours = np.linspace(cb_min,cb_max,3)

        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4,frameon=False)
        pGlass = plt.pcolor(xCell,yCell,np.zeros(varStep.shape),cmap=cm.rainbow,vmin=contours[0], vmax=contours[-1])
        plt.yticks([])
        if ytix==1:
            plt.yticks([-450, -400, -350])
        if xtix==0:
            plt.xticks([])
        plt.ylim([np.min(yCell),0.])
        plt.title(cp_title,fontsize=8)
        plt.ylim([-505.0,-320.0])
        if cb==1:
            bbox = ax1.get_position()
            cax = fig.add_axes([bbox.xmin+bbox.width/10.0, bbox.ymin-0.19, bbox.width*0.8, bbox.height*0.13])
            cbar = plt.colorbar(pGlass, cax = cax,orientation='horizontal',ticks=contours[::contour_interval])
            cbar.ax.tick_params(labelsize=8)
            plt.title(cb_title,fontsize=8)
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")

    if np.abs(np.abs(np.max(varStep)) - np.abs(np.min(varStep))) > 0.0:
        if cb_min==-10.0 and cb_max==10.0:
            contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),3)
        if cb_max!=10.0:
            contours = np.linspace(cb_min,cb_max,3)
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
        plt.ylim([-505.0,-320.0])
        pGlass.set_edgecolor("face")
        if cb==1:
            bbox = ax1.get_position()
            cax = fig.add_axes([bbox.xmin+bbox.width/10.0, bbox.ymin-0.19, bbox.width*0.8, bbox.height*0.13])
            cbar = plt.colorbar(pGlass, cax = cax,orientation='horizontal',ticks=contours[::contour_interval])
            cbar.ax.tick_params(labelsize=8)
            plt.title(cb_title,fontsize=8)
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")
    return chemplot24


def chemcont(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, xtix=0, ytix=0, frame_lines=0, min_color='r', hatching='', bg_alpha=0.5, ed_col='k'):
    # varStep[varStep>0.0] = 1.0
    varStep[varStep>0.1*np.max(varStep)] = 1.0
    varStep[varStep!=1.0] = 0.0
    if frame_lines==1:
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*2.75,frameon=True)
    if frame_lines==0:
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*2.75,frameon=False)
    if hatching!='':
        # varStep[varStep>0.0] = 1.0
        pGlass = plt.contourf(xCell,yCell,varStep,[0.5,1.0],colors=[min_color], alpha=0.0, edgecolors=[min_color],hatches=[hatching])
        #pGlass.set_linewidth(0.25)
    if hatching=='':
        if ed_col=='w':
            pGlass = plt.contourf(xCell,yCell,varStep,[0.5,1.0],colors=[min_color], alpha=bg_alpha, zorder=-10)
        if ed_col!='w':
            pGlass = plt.contour(xCell,yCell,varStep,[0.5,1.0],colors=[ed_col], linewidths=2, zorder=3)
    plt.yticks([])
    if ytix==1:
        plt.yticks([-500, -450, -400, -350, -300],fontsize=8)
    if xtix==0:
        plt.xticks([])
    plt.xticks(fontsize=8)
    plt.ylim([-520.0,-330.0])
    plt.title(cp_title,fontsize=10)
    return chemcont


def chemcont_vol(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, xtix=0, ytix=0, frame_lines=0, cb=0, cb_title=''):
    # varStep[varStep>0.0] = 1.0

    for mm in range(varStep.shape[0]-1):
        if np.max(varStep[mm,:]) == 0.0:
            varStep[mm,:] = varStep[mm+1,:]
    for nn in range(varStep.shape[1]-1):
        if np.max(varStep[:,nn+1]) == 0.0 and np.max(varStep[:,nn]) != 0.0:
            varStep[:,nn+1] = varStep[:,nn]

    if frame_lines==1:
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*2.75,frameon=True)
    if frame_lines==0:
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*2.75,frameon=False)
    contours = np.linspace(0.0,0.2,contour_interval)
    pGlass = plt.contourf(xCell,yCell,varStep,contours,cmap=cm.binary)
    for c in pGlass.collections:
        c.set_edgecolor("face")
    #pGlass.set_linewidth(0.0)

    plt.yticks([])
    if ytix==1:
        plt.yticks([-500, -450, -400, -350, -300])
    if xtix==0:
        plt.xticks([])
    plt.ylim([-520.0,-330.0])
    plt.title(cp_title,fontsize=10)

    if cb==1:
        #cbaxes = fig.add_axes([0.5, 0.5, 0.3, 0.03])
        #cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        bbox = ax1.get_position()
        #print bbox
        cax = fig.add_axes([bbox.xmin+bbox.width/7.5, bbox.ymin-0.56, bbox.width*0.8, bbox.height*0.13])
        cbar = plt.colorbar(pGlass, cax = cax,orientation='horizontal',ticks=[0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
        plt.title(cb_title,fontsize=10)
        #cbar = plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::contour_interval],shrink=0.9, pad = 0.5)
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor("face")
    return chemcont_vol


def chemcont_l(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, xtix=1, ytix=1,
perm_lines=1, frame_lines=1, min_cmap=cm.coolwarm, cb_min=-10.0, cb_max=10.0):
    if frame_lines==1:
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4*1.5,frameon=True)
    if frame_lines==0:
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4*1.5,frameon=False)
        #pGlass.set_linewidth(0.25)
    if np.max(varStep) > 0.0:
        #contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varStep),10)
        contours = np.linspace(cb_min,cb_max+cb_max/10.0,10)
        pGlass = plt.contourf(xCell,yCell,varStep,contours[:],cmap=min_cmap, alpha=1.0)
    if perm_lines==1:
        p = plt.contour(xgh,ygh,perm[:,:],[-14.9],colors='black',linewidths=np.array([1.0]),zorder=-3)
    plt.yticks([])
    if ytix==1:
        plt.yticks([-500, -400, -300, -200, -100, 0])
    if xtix==0:
        plt.xticks([])
    # plt.ylim([np.min(yCell),0.])
    #plt.ylim([-500.0,-300.0])
    plt.ylim([-525.0,-350.0])
    plt.title(cp_title,fontsize=10)

def fmt(x, pos):
    a, b = '{:.4e}'.format(x).split('e')
    b = int(b)
    return r'${}e{{{}}}$'.format(a, b)


    #pGlass.set_edgecolor("face")
    return chemcont_l

#hack: init big arrays

#secMat = np.zeros([len(yCell),len(xCell)*steps+1,minNum+1])
secMat = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])
secMat_a = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])
secMat_b = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])
secMat_d = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])


secVol0 = np.zeros([len(yCell),len(xCell)*steps+corr])
secVol0_a = np.zeros([len(yCell),len(xCell)*steps+corr])
secVol0_b = np.zeros([len(yCell),len(xCell)*steps+corr])
secVol0_d = np.zeros([len(yCell),len(xCell)*steps+corr])

secVol = np.zeros([len(yCell),len(xCell),minNum+1])
secVol_a = np.zeros([len(yCell),len(xCell),minNum+1])
secVol_b = np.zeros([len(yCell),len(xCell),minNum+1])
secVol_d = np.zeros([len(yCell),len(xCell),minNum+1])

#satMat = np.zeros([(bitsCy-1),(bitsCx-1)*steps,minNum+1])

secStep = np.zeros([len(yCell),len(xCell),minNum+1])
secStep_a = np.zeros([len(yCell),len(xCell),minNum+1])
secStep_b = np.zeros([len(yCell),len(xCell),minNum+1])
secStep_d = np.zeros([len(yCell),len(xCell),minNum+1])

secStep_last = np.zeros([len(yCell),len(xCell),minNum+1])
secStep_last_a = np.zeros([len(yCell),len(xCell),minNum+1])
secStep_last_b = np.zeros([len(yCell),len(xCell),minNum+1])
secStep_last_d = np.zeros([len(yCell),len(xCell),minNum+1])

dsecMat = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])
dsecMat_a = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])
dsecMat_b = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])
dsecMat_d = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])

dsecStep = np.zeros([len(yCell),len(xCell),minNum+1])
dsecStep_a = np.zeros([len(yCell),len(xCell),minNum+1])
dsecStep_b = np.zeros([len(yCell),len(xCell),minNum+1])
dsecStep_d = np.zeros([len(yCell),len(xCell),minNum+1])
#satStep = np.zeros([bitsCy,bitsCx,minNum+1])

dpriMat = np.zeros([len(yCell),len(xCell)*steps+corr,1])
dpriMat_a = np.zeros([len(yCell),len(xCell)*steps+corr,1])
dpriMat_b = np.zeros([len(yCell),len(xCell)*steps+corr,1])
dpriMat_d = np.zeros([len(yCell),len(xCell)*steps+corr,1])

dpriStep = np.zeros([len(yCell),len(xCell),1])
dpriStep_a = np.zeros([len(yCell),len(xCell),1])
dpriStep_b = np.zeros([len(yCell),len(xCell),1])
dpriStep_d = np.zeros([len(yCell),len(xCell),1])

secStep_ts = np.zeros([steps,minNum+1])
secStep_ts_a = np.zeros([steps,minNum+1])
secStep_ts_b = np.zeros([steps,minNum+1])
secStep_ts_d = np.zeros([steps,minNum+1])

dsecStep_ts = np.zeros([steps,minNum+1])
dsecStep_ts_a = np.zeros([steps,minNum+1])
dsecStep_ts_b = np.zeros([steps,minNum+1])
dsecStep_ts_d = np.zeros([steps,minNum+1])


x_secStep_ts = np.zeros([steps,minNum+1])
x_secStep_ts_a = np.zeros([steps,minNum+1])
x_secStep_ts_b = np.zeros([steps,minNum+1])
x_secStep_ts_d = np.zeros([steps,minNum+1])

x_dsecStep_ts = np.zeros([steps,minNum+1])
x_dsecStep_ts_a = np.zeros([steps,minNum+1])
x_dsecStep_ts_b = np.zeros([steps,minNum+1])
x_dsecStep_ts_d = np.zeros([steps,minNum+1])

x_d = -5
xd_move = 0
# moves = np.array([2, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40, 42])
moves = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40, 42])

priStep_ts = np.zeros([steps,6])
priStep_ts_a = np.zeros([steps,6])
priStep_ts_b = np.zeros([steps,6])
priStep_ts_d = np.zeros([steps,6])


x_priStep_ts = np.zeros([steps,6])
x_priStep_ts_a = np.zeros([steps,6])
x_priStep_ts_b = np.zeros([steps,6])
x_priStep_ts_d = np.zeros([steps,6])

x_dpriStep_ts = np.zeros([steps,6])
x_dpriStep_ts_a = np.zeros([steps,6])
x_dpriStep_ts_b = np.zeros([steps,6])
x_dpriStep_ts_d = np.zeros([steps,6])


dpriStep_ts = np.zeros([steps,6])
dpriStep_ts_a = np.zeros([steps,6])
dpriStep_ts_b = np.zeros([steps,6])
dpriStep_ts_d = np.zeros([steps,6])


ternK = np.zeros([len(yCell),len(xCell),steps])
ternK_a = np.zeros([len(yCell),len(xCell),steps])
ternK_b = np.zeros([len(yCell),len(xCell),steps])
ternK_d = np.zeros([len(yCell),len(xCell),steps])

ternMg = np.zeros([len(yCell),len(xCell),steps])
ternMg_a = np.zeros([len(yCell),len(xCell),steps])
ternMg_b = np.zeros([len(yCell),len(xCell),steps])
ternMg_d = np.zeros([len(yCell),len(xCell),steps])

ternFe = np.zeros([len(yCell),len(xCell),steps])
ternFe_a = np.zeros([len(yCell),len(xCell),steps])
ternFe_b = np.zeros([len(yCell),len(xCell),steps])
ternFe_d = np.zeros([len(yCell),len(xCell),steps])

tern_list = np.zeros([len(yCell)*len(xCell),steps,3])
tern_list_a = np.zeros([len(yCell)*len(xCell),steps,3])
tern_list_b = np.zeros([len(yCell)*len(xCell),steps,3])
tern_list_d = np.zeros([len(yCell)*len(xCell),steps,3])






tern_rb_K = np.zeros([len(yCell),len(xCell),steps])
tern_rb_K_a = np.zeros([len(yCell),len(xCell),steps])
tern_rb_K_b = np.zeros([len(yCell),len(xCell),steps])
tern_rb_K_d = np.zeros([len(yCell),len(xCell),steps])

tern_rb_Mg = np.zeros([len(yCell),len(xCell),steps])
tern_rb_Mg_a = np.zeros([len(yCell),len(xCell),steps])
tern_rb_Mg_b = np.zeros([len(yCell),len(xCell),steps])
tern_rb_Mg_d = np.zeros([len(yCell),len(xCell),steps])

tern_rb_Fe = np.zeros([len(yCell),len(xCell),steps])
tern_rb_Fe_a = np.zeros([len(yCell),len(xCell),steps])
tern_rb_Fe_b = np.zeros([len(yCell),len(xCell),steps])
tern_rb_Fe_d = np.zeros([len(yCell),len(xCell),steps])

tern_rb_list = np.zeros([len(yCell)*len(xCell),steps,3])
tern_rb_list_a = np.zeros([len(yCell)*len(xCell),steps,3])
tern_rb_list_b = np.zeros([len(yCell)*len(xCell),steps,3])
tern_rb_list_d = np.zeros([len(yCell)*len(xCell),steps,3])

tern_rb_point_K = np.zeros([steps])
tern_rb_point_Mg = np.zeros([steps])
tern_rb_point_Fe = np.zeros([steps])


tern_rb_point_K_d = np.zeros([steps])
tern_rb_point_Mg_d = np.zeros([steps])
tern_rb_point_Fe_d = np.zeros([steps])


tern_rb_point_K_a = np.zeros([steps])
tern_rb_point_Mg_a = np.zeros([steps])
tern_rb_point_Fe_a = np.zeros([steps])


tern_rb_point_K_b = np.zeros([steps])
tern_rb_point_Mg_b = np.zeros([steps])
tern_rb_point_Fe_b = np.zeros([steps])


tern_rb_point_list = np.zeros([steps,3])
tern_rb_point_list_d = np.zeros([steps,3])
tern_rb_point_list_a = np.zeros([steps,3])
tern_rb_point_list_b = np.zeros([steps,3])







tern_x_rb_K = np.zeros([len(yCell),len(xCell),steps])
tern_x_rb_K_a = np.zeros([len(yCell),len(xCell),steps])
tern_x_rb_K_b = np.zeros([len(yCell),len(xCell),steps])
tern_x_rb_K_d = np.zeros([len(yCell),len(xCell),steps])

tern_x_rb_Mg = np.zeros([len(yCell),len(xCell),steps])
tern_x_rb_Mg_a = np.zeros([len(yCell),len(xCell),steps])
tern_x_rb_Mg_b = np.zeros([len(yCell),len(xCell),steps])
tern_x_rb_Mg_d = np.zeros([len(yCell),len(xCell),steps])

tern_x_rb_Fe = np.zeros([len(yCell),len(xCell),steps])
tern_x_rb_Fe_a = np.zeros([len(yCell),len(xCell),steps])
tern_x_rb_Fe_b = np.zeros([len(yCell),len(xCell),steps])
tern_x_rb_Fe_d = np.zeros([len(yCell),len(xCell),steps])

tern_x_rb_list = np.zeros([len(yCell)*len(xCell),steps,3])
tern_x_rb_list_a = np.zeros([len(yCell)*len(xCell),steps,3])
tern_x_rb_list_b = np.zeros([len(yCell)*len(xCell),steps,3])
tern_x_rb_list_d = np.zeros([len(yCell)*len(xCell),steps,3])

tern_x_rb_point_K = np.zeros([steps])
tern_x_rb_point_Mg = np.zeros([steps])
tern_x_rb_point_Fe = np.zeros([steps])


tern_x_rb_point_K_d = np.zeros([steps])
tern_x_rb_point_Mg_d = np.zeros([steps])
tern_x_rb_point_Fe_d = np.zeros([steps])


tern_x_rb_point_K_a = np.zeros([steps])
tern_x_rb_point_Mg_a = np.zeros([steps])
tern_x_rb_point_Fe_a = np.zeros([steps])


tern_x_rb_point_K_b = np.zeros([steps])
tern_x_rb_point_Mg_b = np.zeros([steps])
tern_x_rb_point_Fe_b = np.zeros([steps])


tern_x_rb_point_list = np.zeros([steps,3])
tern_x_rb_point_list_d = np.zeros([steps,3])
tern_x_rb_point_list_a = np.zeros([steps,3])
tern_x_rb_point_list_b = np.zeros([steps,3])

#hack: net uptake arrays

x_elements = np.zeros([steps,15])
x_elements_d = np.zeros([steps,15])
x_elements_a = np.zeros([steps,15])
x_elements_b = np.zeros([steps,15])

x_pri_elements = np.zeros([steps,15])
x_pri_elements_d = np.zeros([steps,15])
x_pri_elements_a = np.zeros([steps,15])
x_pri_elements_b = np.zeros([steps,15])


elements_sec = np.zeros([minNum+1,15])
elements_pri = np.zeros([1,15])

elements_pri[0,5] = 0.1178 # Ca
elements_pri[0,6] = 0.097 # Mg
elements_pri[0,7] = 0.047 # Na
elements_pri[0,8] = 0.00219 # K
elements_pri[0,9] = 0.1165 # Fe
elements_pri[0,10] = 0.0 # S
elements_pri[0,11] =  0.4655 # Si
elements_pri[0,12] = 0.0 # Cl
elements_pri[0,13] =  0.153 # Al

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


# elements_sec[,5] = 0.0 # Ca
# elements_sec[,6] = 0.0 # Mg
# elements_sec[,7] = 0.0 # Na
# elements_sec[,8] = 0.0 # K
# elements_sec[,9] = 0.0 # Fe
# elements_sec[,10] = 0.0 # S
# elements_sec[,11] = 0.0 # Si
# elements_sec[,12] = 0.0 # Cl
# elements_sec[,13] = 0.0 # Al


any_min = []

#hack: load in chem data
if chem == 1:
    # IMPORT MINERALS
    print " "
    if d_only == 1:
        ch_path = path + 'ch_a/'
    if d_only == 0:
        ch_path = path + 'ch_s/'
    print ch_path
    for j in range(1,minNum):
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            print j , secondary[j] ,
            secMat[:,:,j] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat[:,:,j] = secMat[:,:,j]*molar[j]/density[j]
            dsecMat[:,2*len(xCell):,j] = secMat[:,len(xCell):-len(xCell),j] - secMat[:,2*len(xCell):,j]
            secVol0 = secVol0 + secMat[:,:,j]
            #print np.max(secVol0)
    inert0 = np.loadtxt(ch_path + 'z_sol_inert.txt')
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
    dpriMat[:,2*len(xCell):,0] = glass0[:,len(xCell):-len(xCell)] - glass0[:,2*len(xCell):]
    #glass0_p = glass0/(np.max(glass0))
    togg0 = np.loadtxt(ch_path + 'z_med_cell_toggle.txt')
    phiCalcIn0 = np.loadtxt(ch_path + 'z_phiCalc.txt')
    ph_fix0 = np.loadtxt(ch_path + 'z_ph_fix.txt')
    # speed1_0 = np.loadtxt(ch_path + 'z_speed_1.txt')
    # speed2_0 = np.loadtxt(ch_path + 'z_speed_2.txt')
    # speed3_0 = np.loadtxt(ch_path + 'z_speed_3.txt')

    # if d_only == 1:
    #     ch_path = path + 'ch_b/'
    ol0 = np.loadtxt(ch_path + 'z_pri_ol.txt')*molar_pri[2]/density_pri[2]
    pyr0 = np.loadtxt(ch_path + 'z_pri_pyr.txt')*molar_pri[1]/density_pri[1]
    plag0 = np.loadtxt(ch_path + 'z_pri_plag.txt')*molar_pri[0]/density_pri[0]
    pri_total0 = glass0[:,:]# + ol0 + pyr0 + plag0

    #pri_total0 = pri_total0/np.max(pri_total0)



    print " "
    ch_path = path + 'ch_a/'
    print ch_path
    for j in range(1,minNum):
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            print j , secondary[j] ,
            secMat_a[:,:,j] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat_a[:,:,j] = secMat_a[:,:,j]*molar[j]/density[j]
            dsecMat_a[:,2*len(xCell):,j] = secMat_a[:,len(xCell):-len(xCell),j] - secMat_a[:,2*len(xCell):,j]
            secVol0_a = secVol0_a + secMat_a[:,:,j]
    inert0_a = np.loadtxt(ch_path + 'z_sol_inert.txt')
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
    dpriMat_a[:,2*len(xCell):,0] = glass0_a[:,len(xCell):-len(xCell)] - glass0_a[:,2*len(xCell):]
    #glass0_p_a = glass0_a/(np.max(glass0_a))
    ol0_a = np.loadtxt(ch_path + 'z_pri_ol.txt')*molar_pri[2]/density_pri[2]
    pyr0_a = np.loadtxt(ch_path + 'z_pri_pyr.txt')*molar_pri[1]/density_pri[1]
    plag0_a = np.loadtxt(ch_path + 'z_pri_plag.txt')*molar_pri[0]/density_pri[0]
    phiCalcIn0_a = np.loadtxt(ch_path + 'z_phiCalc.txt')
    pri_total0_a = glass0_a[:,:]# + ol0_a + pyr0_a + plag0_a
    ph_fix0_a = np.loadtxt(ch_path + 'z_ph_fix.txt')
    #pri_total0_a = pri_total0_a/np.max(pri_total0_a)
    # speed1_0_a = np.loadtxt(ch_path + 'z_speed_1.txt')
    # speed2_0_a = np.loadtxt(ch_path + 'z_speed_2.txt')
    # speed3_0_a = np.loadtxt(ch_path + 'z_speed_3.txt')




    print " "
    ch_path = path + 'ch_b/'
    print ch_path
    for j in range(1,minNum):
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            print j , secondary[j] ,
            secMat_b[:,:,j] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat_b[:,:,j] = secMat_b[:,:,j]*molar[j]/density[j]
            dsecMat_b[:,2*len(xCell):,j] = secMat_b[:,len(xCell):-len(xCell),j] - secMat_b[:,2*len(xCell):,j]
            secVol0_b = secVol0_b + secMat_b[:,:,j]
    inert0_b = np.loadtxt(ch_path + 'z_sol_inert.txt')
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
    dpriMat_b[:,2*len(xCell):,0] = glass0_b[:,len(xCell):-len(xCell)] - glass0_b[:,2*len(xCell):]
    #glass0_p_b = glass0_b#/(np.max(glass0_b))
    ol0_b = np.loadtxt(ch_path + 'z_pri_ol.txt')*molar_pri[2]/density_pri[2]
    pyr0_b = np.loadtxt(ch_path + 'z_pri_pyr.txt')*molar_pri[1]/density_pri[1]
    plag0_b = np.loadtxt(ch_path + 'z_pri_plag.txt')*molar_pri[0]/density_pri[0]
    pri_total0_b = glass0_b[:,:]# + ol0_b + pyr0_b + plag0_b
    ph_fix0_b = np.loadtxt(ch_path + 'z_ph_fix.txt')
    #pri_total0_b = pri_total0_b/np.max(pri_total0_b)
    # speed1_0_b = np.loadtxt(ch_path + 'z_speed_1.txt')
    # speed2_0_b = np.loadtxt(ch_path + 'z_speed_2.txt')
    # speed3_0_b = np.loadtxt(ch_path + 'z_speed_3.txt')


    print " "
    ch_path = path + 'ch_d/'
    print ch_path
    for j in range(1,minNum):
        #if j % 5 ==0:
            #print 'loading minerals', str(j-5), "-", str(j)
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            print j , secondary[j] ,
            secMat_d[:,:,j] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat_d[:,:,j] = secMat_d[:,:,j]*molar[j]/density[j]
            dsecMat_d[:,2*len(xCell):,j] = secMat_d[:,len(xCell):-len(xCell),j] - secMat_d[:,2*len(xCell):,j]
            secVol0_d = secVol0_d + secMat_d[:,:,j]
    inert0_d = np.loadtxt(ch_path + 'z_sol_inert.txt')
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
    dpriMat_d[:,2*len(xCell):,0] = glass0_d[:,len(xCell):-len(xCell)] - glass0_d[:,2*len(xCell):]
    #glass0_p_d = glass0_d/(np.max(glass0_d))
    ol0_d = np.loadtxt(ch_path + 'z_pri_ol.txt')*molar_pri[2]/density_pri[2]
    pyr0_d = np.loadtxt(ch_path + 'z_pri_pyr.txt')*molar_pri[1]/density_pri[1]
    plag0_d = np.loadtxt(ch_path + 'z_pri_plag.txt')*molar_pri[0]/density_pri[0]
    pri_total0_d = glass0_d[:,:]# + ol0_d + pyr0_d + plag0_d
    #pri_total0_d = pri_total0_d/np.max(pri_total0_d)

    print " "
    print " "
    print "any_min" , any_min
    # any_min = sorted(any_min,key=int)
    # for j in range(len(any_min)):
    #     any_min[j] = int(any_min[j])
    # any_min = np.sort(any_min)
    # print "any_min sorted" , any_min


    ca0[np.isinf(ca0)] = 1.0
    ca0_a[np.isinf(ca0_a)] = 1.0
    ca0_b[np.isinf(ca0_b)] = 1.0
    ca0_d[np.isinf(ca0_d)] = 1.0


    dsecMat = np.abs(dsecMat)
    dsecMat_a = np.abs(dsecMat_a)
    dsecMat_b = np.abs(dsecMat_b)
    dsecMat_d = np.abs(dsecMat_d)


    dsecStep = np.abs(dsecStep)
    dsecStep_a = np.abs(dsecStep_a)
    dsecStep_b = np.abs(dsecStep_b)
    dsecStep_d = np.abs(dsecStep_d)

if chem == 1:

    #hack: make compound chem arrays

    alt_vol0 = np.zeros(secMat[:,:,1].shape)
    alt_vol0_a = np.zeros(secMat[:,:,1].shape)
    alt_vol0_b = np.zeros(secMat[:,:,1].shape)
    alt_vol0_d = np.zeros(secMat[:,:,1].shape)



    for j in range(len(xCell)*steps):
        for jj in range(len(yCell)):
            if pri_total0[jj,j] > 0.0:
                alt_vol0[jj,j] = np.sum(secMat[jj,j,:])/(pri_total0[jj,j]+np.sum(secMat[jj,j,:]))


    for j in range(len(xCell)*steps):
        for jj in range(len(yCell)):
            if pri_total0_a[jj,j] > 0.0:
                alt_vol0_a[jj,j] = np.sum(secMat_a[jj,j,:])/(pri_total0_a[jj,j]+np.sum(secMat_a[jj,j,:]))


    for j in range(len(xCell)*steps):
        for jj in range(len(yCell)):
            if pri_total0_b[jj,j] > 0.0:
                alt_vol0_b[jj,j] = np.sum(secMat_b[jj,j,:])/(pri_total0_b[jj,j]+np.sum(secMat_b[jj,j,:]))


    for j in range(len(xCell)*steps):
            for jj in range(len(yCell)):
                if pri_total0_d[jj,j] > 0.0:
                    alt_vol0_d[jj,j] = np.sum(secMat_d[jj,j,:])/(pri_total0_d[jj,j]+np.sum(secMat_d[jj,j,:]))


conv_mean_qu = 0.0
conv_max_qu = 0.0
conv_mean_psi = 0.0
conv_max_psi = 0.0
conv_tot_hf = 0.0
conv_count = 0



#for i in range(4,steps,5):
for i in range(0,steps,1):
    print " "
    print "step =", i

    if np.any(moves== i + restart):
        xd_move = xd_move + 1
    print xd_move

    if i == 1:
        #tod0: FIGURE: togg_plot.png
        fig=plt.figure()
        ax1=fig.add_subplot(1,1,1, frameon=False)
        plt.pcolor(togg)
        plt.pcolor(togg, cmap=cm.Greys_r, facecolor='none', edgecolor='w', zorder=2)
        fig.savefig(outpath+'togg_plot.png')


    #hack: CUT UP ALL CHEMS
    if chem == 1:
        for j in range(len(any_min)):
            secStep[:,:,any_min[j]] = cut_chem(secMat[:,:,any_min[j]],i)
            dsecStep[:,:,any_min[j]] = cut_chem(dsecMat[:,:,any_min[j]],i)
            secStep_ts[i,any_min[j]] = np.sum(secStep[:,:,any_min[j]])
            x_secStep_ts[i,any_min[j]] = np.sum(secStep[:,xd_move,any_min[j]])
            if i > 0:
                dsecStep_ts[i,any_min[j]] = secStep_ts[i,any_min[j]] - secStep_ts[i-1,any_min[j]]
                x_dsecStep_ts[i,any_min[j]] = x_secStep_ts[i,any_min[j]] - x_secStep_ts[i-1,any_min[j]]
        inert = cut_chem(inert0,i)
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
        dpriStep[:,:,0] = cut_chem(dpriMat[:,:,0],i)
        #glass_p = cut_chem(glass0_p,i)
        togg = cut_chem(togg0,i)
        alt_vol = cut_chem(alt_vol0,i)
        pri_total = cut_chem(pri_total0,i)
        secVol = cut_chem(secVol0,i)
        ol = cut_chem(ol0,i)
        pyr = cut_chem(pyr0,i)
        plag = cut_chem(plag0,i)
        phiCalcIn = cut_chem(phiCalcIn0,i)
        ph_fix = cut_chem(ph_fix0,i)
        # speed1 = cut_chem(speed1_0,i)
        # speed2 = cut_chem(speed2_0,i)
        # speed3 = cut_chem(speed3_0,i)

        x_priStep_ts[i,2] = np.sum(plag[:,xd_move])
        x_priStep_ts[i,3] = np.sum(pyr[:,xd_move])
        x_priStep_ts[i,4] = np.sum(ol[:,xd_move])
        x_priStep_ts[i,5] = np.sum(glass[:,xd_move])

        priStep_ts[i,5] = np.sum(glass)
        priStep_ts[i,4] = np.sum(ol)
        priStep_ts[i,3] = np.sum(pyr)
        priStep_ts[i,2] = np.sum(plag)
        if i > 0:
            dpriStep_ts[i,5] = priStep_ts[i,5] - priStep_ts[i-1,5]
            dpriStep_ts[i,4] = priStep_ts[i,4] - priStep_ts[i-1,4]
            dpriStep_ts[i,3] = priStep_ts[i,3] - priStep_ts[i-1,3]
            dpriStep_ts[i,2] = priStep_ts[i,2] - priStep_ts[i-1,2]

            x_dpriStep_ts[i,5] = x_priStep_ts[i,5] - x_priStep_ts[i-1,5]
            #print " X DPRI STEP TS" , x_dpriStep_ts[i,5]
            x_dpriStep_ts[i,4] = x_priStep_ts[i,4] - x_priStep_ts[i-1,4]
            x_dpriStep_ts[i,3] = x_priStep_ts[i,3] - x_priStep_ts[i-1,3]
            x_dpriStep_ts[i,2] = x_priStep_ts[i,2] - x_priStep_ts[i-1,2]
            x_dpriStep_ts[i,1] = x_priStep_ts[i,1] - x_priStep_ts[i-1,1]




        for j in range(len(any_min)):
            secStep_a[:,:,any_min[j]] = cut_chem(secMat_a[:,:,any_min[j]],i)
            dsecStep_a[:,:,any_min[j]] = cut_chem(dsecMat_a[:,:,any_min[j]],i)
            secStep_ts_a[i,any_min[j]] = np.sum(secStep_a[:,:,any_min[j]])
            x_secStep_ts_a[i,any_min[j]] = np.sum(secStep_a[:,xd_move,any_min[j]])
            if i > 0:
                dsecStep_ts_a[i,any_min[j]] = secStep_ts_a[i,any_min[j]] - secStep_ts_a[i-1,any_min[j]]
                x_dsecStep_ts_a[i,any_min[j]] = x_secStep_ts_a[i,any_min[j]] - x_secStep_ts_a[i-1,any_min[j]]
        inert_a = cut_chem(inert0_a,i)
        dic_a = cut_chem(dic0_a,i)
        ca_a = cut_chem(ca0_a,i)
        ph_a = cut_chem(ph0_a,i)
        alk_a = cut_chem(alk0_a,i)
        solw_a = cut_chem(solw0_a,i)
        mg_a = cut_chem(mg0_a,i)
        fe_a = cut_chem(fe0_a,i)
        si_a = cut_chem(si0_a,i)
        k1_a = cut_chem(k0_a,i)
        na_a = cut_chem(na0_a,i)
        cl_a = cut_chem(cl0_a,i)
        al_a = cut_chem(al0_a,i)
        glass_a = cut_chem(glass0_a,i)
        dpriStep_a[:,:,0] = cut_chem(dpriMat_a[:,:,0],i)
        #glass_p_a = cut_chem(glass0_p_a,i)
        alt_vol_a = cut_chem(alt_vol0_a,i)
        pri_total_a = cut_chem(pri_total0_a,i)
        secVol_a = cut_chem(secVol0_a,i)
        ol_a = cut_chem(ol0_a,i)
        pyr_a = cut_chem(pyr0_a,i)
        plag_a = cut_chem(plag0_a,i)
        phiCalcIn_a = cut_chem(phiCalcIn0_a,i)
        ph_fix_a = cut_chem(ph_fix0_a,i)
        # speed1_a = cut_chem(speed1_0_a,i)
        # speed2_a = cut_chem(speed2_0_a,i)
        # speed3_a = cut_chem(speed3_0_a,i)

        x_priStep_ts_a[i,2] = np.sum(plag_a[:,xd_move])
        x_priStep_ts_a[i,3] = np.sum(pyr_a[:,xd_move])
        x_priStep_ts_a[i,4] = np.sum(ol_a[:,xd_move])
        x_priStep_ts_a[i,5] = np.sum(glass_a[:,xd_move])

        priStep_ts_a[i,5] = np.sum(glass_a)
        priStep_ts_a[i,4] = np.sum(ol_a)
        priStep_ts_a[i,3] = np.sum(pyr_a)
        priStep_ts_a[i,2] = np.sum(plag_a)
        if i > 0:
            dpriStep_ts_a[i,5] = priStep_ts_a[i,5] - priStep_ts_a[i-1,5]
            dpriStep_ts_a[i,4] = priStep_ts_a[i,4] - priStep_ts_a[i-1,4]
            dpriStep_ts_a[i,3] = priStep_ts_a[i,3] - priStep_ts_a[i-1,3]
            dpriStep_ts_a[i,2] = priStep_ts_a[i,2] - priStep_ts_a[i-1,2]

            x_dpriStep_ts_a[i,5] = x_priStep_ts_a[i,5] - x_priStep_ts_a[i-1,5]
            x_dpriStep_ts_a[i,4] = x_priStep_ts_a[i,4] - x_priStep_ts_a[i-1,4]
            x_dpriStep_ts_a[i,3] = x_priStep_ts_a[i,3] - x_priStep_ts_a[i-1,3]
            x_dpriStep_ts_a[i,2] = x_priStep_ts_a[i,2] - x_priStep_ts_a[i-1,2]
            x_dpriStep_ts_a[i,1] = x_priStep_ts_a[i,1] - x_priStep_ts_a[i-1,1]




        for j in range(len(any_min)):
            secStep_b[:,:,any_min[j]] = cut_chem(secMat_b[:,:,any_min[j]],i)
            dsecStep_b[:,:,any_min[j]] = cut_chem(dsecMat_b[:,:,any_min[j]],i)
            secStep_ts_b[i,any_min[j]] = np.sum(secStep_b[:,:,any_min[j]])
            x_secStep_ts_b[i,any_min[j]] = np.sum(secStep_b[:,xd_move,any_min[j]])
            if i > 0:
                dsecStep_ts_b[i,any_min[j]] = secStep_ts_b[i,any_min[j]] - secStep_ts_b[i-1,any_min[j]]
                x_dsecStep_ts_b[i,any_min[j]] = x_secStep_ts_b[i,any_min[j]] - x_secStep_ts_b[i-1,any_min[j]]
        inert_b = cut_chem(inert0_b,i)
        dic_b = cut_chem(dic0_b,i)
        ca_b = cut_chem(ca0_b,i)
        ph_b = cut_chem(ph0_b,i)
        alk_b = cut_chem(alk0_b,i)
        solw_b = cut_chem(solw0_b,i)
        mg_b = cut_chem(mg0_b,i)
        fe_b = cut_chem(fe0_b,i)
        si_b = cut_chem(si0_b,i)
        k1_b = cut_chem(k0_b,i)
        na_b = cut_chem(na0_b,i)
        cl_b = cut_chem(cl0_b,i)
        al_b = cut_chem(al0_b,i)
        glass_b = cut_chem(glass0_b,i)
        dpriStep_b[:,:,0] = cut_chem(dpriMat_b[:,:,0],i)
        #glass_p_b = cut_chem(glass0_p_b,i)
        alt_vol_b = cut_chem(alt_vol0_b,i)
        pri_total_b = cut_chem(pri_total0_b,i)
        secVol_b = cut_chem(secVol0_b,i)
        ol_b = cut_chem(ol0_b,i)
        pyr_b = cut_chem(pyr0_b,i)
        plag_b = cut_chem(plag0_b,i)
        ph_fix_b = cut_chem(ph_fix0_b,i)
        # speed1_b = cut_chem(speed1_0_b,i)
        # speed2_b = cut_chem(speed2_0_b,i)
        # speed3_b = cut_chem(speed3_0_b,i)

        x_priStep_ts_b[i,2] = np.sum(plag_b[:,xd_move])
        x_priStep_ts_b[i,3] = np.sum(pyr_b[:,xd_move])
        x_priStep_ts_b[i,4] = np.sum(ol_b[:,xd_move])
        x_priStep_ts_b[i,5] = np.sum(glass_b[:,xd_move])

        priStep_ts_b[i,5] = np.sum(glass_b)
        priStep_ts_b[i,4] = np.sum(ol_b)
        priStep_ts_b[i,3] = np.sum(pyr_b)
        priStep_ts_b[i,2] = np.sum(plag_b)

        if i > 0:
            dpriStep_ts_b[i,5] = priStep_ts_b[i,5] - priStep_ts_b[i-1,5]
            dpriStep_ts_b[i,4] = priStep_ts_b[i,4] - priStep_ts_b[i-1,4]
            dpriStep_ts_b[i,3] = priStep_ts_b[i,3] - priStep_ts_b[i-1,3]
            dpriStep_ts_b[i,2] = priStep_ts_b[i,2] - priStep_ts_b[i-1,2]

            x_dpriStep_ts_b[i,5] = x_priStep_ts_b[i,5] - x_priStep_ts_b[i-1,5]
            x_dpriStep_ts_b[i,4] = x_priStep_ts_b[i,4] - x_priStep_ts_b[i-1,4]
            x_dpriStep_ts_b[i,3] = x_priStep_ts_b[i,3] - x_priStep_ts_b[i-1,3]
            x_dpriStep_ts_b[i,2] = x_priStep_ts_b[i,2] - x_priStep_ts_b[i-1,2]
            x_dpriStep_ts_b[i,1] = x_priStep_ts_b[i,1] - x_priStep_ts_b[i-1,1]




        for j in range(len(any_min)):
            secStep_d[:,:,any_min[j]] = cut_chem(secMat_d[:,:,any_min[j]],i)
            dsecStep_d[:,:,any_min[j]] = cut_chem(dsecMat_d[:,:,any_min[j]],i)
            secStep_ts_d[i,any_min[j]] = np.sum(secStep_d[:,:,any_min[j]])
            x_secStep_ts_d[i,any_min[j]] = np.sum(secStep_d[:,xd_move,any_min[j]])
            if i > 0:
                dsecStep_ts_d[i,any_min[j]] = secStep_ts_d[i,any_min[j]] - secStep_ts_d[i-1,any_min[j]]
                x_dsecStep_ts_d[i,any_min[j]] = x_secStep_ts_d[i,any_min[j]] - x_secStep_ts_d[i-1,any_min[j]]
        inert_d = cut_chem(inert0_d,i)
        dic_d = cut_chem(dic0_d,i)
        ca_d = cut_chem(ca0_d,i)
        ph_d = cut_chem(ph0_d,i)
        alk_d = cut_chem(alk0_d,i)
        solw_d = cut_chem(solw0_d,i)
        mg_d = cut_chem(mg0_d,i)
        fe_d = cut_chem(fe0_d,i)
        si_d = cut_chem(si0_d,i)
        k1_d = cut_chem(k0_d,i)
        na_d = cut_chem(na0_d,i)
        cl_d = cut_chem(cl0_d,i)
        al_d = cut_chem(al0_d,i)
        glass_d = cut_chem(glass0_d,i)
        dpriStep_d[:,:,0] = cut_chem(dpriMat_d[:,:,0],i)
        #glass_p_d = cut_chem(glass0_p_d,i)
        alt_vol_d = cut_chem(alt_vol0_d,i)
        pri_total_d = cut_chem(pri_total0_d,i)
        secVol_d = cut_chem(secVol0_d,i)
        ol_d = cut_chem(ol0_d,i)
        pyr_d = cut_chem(pyr0_d,i)
        plag_d = cut_chem(plag0_d,i)

        x_priStep_ts_d[i,2] = np.sum(plag_d[:,xd_move])
        x_priStep_ts_d[i,3] = np.sum(pyr_d[:,xd_move])
        x_priStep_ts_d[i,4] = np.sum(ol_d[:,xd_move])
        x_priStep_ts_d[i,5] = np.sum(glass_d[:,xd_move])

        priStep_ts_d[i,5] = np.sum(glass_d)
        priStep_ts_d[i,4] = np.sum(ol_d)
        priStep_ts_d[i,3] = np.sum(pyr_d)
        priStep_ts_d[i,2] = np.sum(plag_d)
        if i > 0:
            dpriStep_ts_d[i,5] = priStep_ts_d[i,5] - priStep_ts_d[i-1,5]
            dpriStep_ts_d[i,4] = priStep_ts_d[i,4] - priStep_ts_d[i-1,4]
            dpriStep_ts_d[i,3] = priStep_ts_d[i,3] - priStep_ts_d[i-1,3]
            dpriStep_ts_d[i,2] = priStep_ts_d[i,2] - priStep_ts_d[i-1,2]

            x_dpriStep_ts_d[i,5] = x_priStep_ts_d[i,5] - x_priStep_ts_d[i-1,5]
            x_dpriStep_ts_d[i,4] = x_priStep_ts_d[i,4] - x_priStep_ts_d[i-1,4]
            x_dpriStep_ts_d[i,3] = x_priStep_ts_d[i,3] - x_priStep_ts_d[i-1,3]
            x_dpriStep_ts_d[i,2] = x_priStep_ts_d[i,2] - x_priStep_ts_d[i-1,2]
            x_dpriStep_ts_d[i,1] = x_priStep_ts_d[i,1] - x_priStep_ts_d[i-1,1]



    # alk_flux = np.zeros(secStep[:,:,1].shape)
    #
    # if chem == 1:
    #     for j in range(len(xCell)):
    #         for jj in range(len(yCell)):
    #             if secStep[jj,j,16] > 0.0 and water[jj,j] > 0.0:
    #                 alk_flux[jj,j] = (secStep[jj,j,16]*density[16]/molar[16] * 2.0) / (water[jj,j] * 10000.0 * (i+1.0) )
    #             if secStep[jj,j,16] == 0.0 and water[jj,j] > 0.0:
    #                 alk_flux[jj,j] = (alk[jj,j] - alk[1,1]) / (water[jj,j] * 1000.0)
    #     print "alk_flux sum" , np.sum(alk_flux)



    #hack: chem6 switch
    chem6 = 5

    if chem6 == 6:

        if chem == 1:

            if not os.path.exists(outpath+'jdf_block_sol/'):
                os.makedirs(outpath+'jdf_block_sol/')

            if np.max(ca) == 0.0:
                plt.close('all')
                quit()



            print "jdf_sol_block"
            #todo: FIGURE: jdf_sol_block
            fig=plt.figure(figsize=(13.0,4.75))
            plt.subplots_adjust( wspace=0.1, bottom=0.125, top=0.97, left=0.01, right=0.99)


            # # col 1
            plt_s = ca0
            plt_a = ca0_a
            plt_b = ca0_b
            plt_d = ca0_d
            plt_ss = ca
            plt_aa = ca_a
            plt_bb = ca_b
            plt_dd = ca_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)
            #c_max = 0.015

            chemplot(plt_s, plt_ss, 7, 7, 1, 1, 'ca s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Ca] concentration')
            chemplot(plt_d, plt_dd, 7, 7, 8, 1, 'ca d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 29, 1, 'ca a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_bb, plt_bb, 7, 14, 30, 1, 'ca b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)


            plt_s = si0
            plt_a = si0_a
            plt_b = si0_b
            plt_d = si0_d
            plt_ss = si
            plt_aa = si_a
            plt_bb = si_b
            plt_dd = si_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 29, 1, 'si s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Si] concentration')
            chemplot(plt_d, plt_dd, 7, 7, 36, 1, 'si d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 29+56, 1, 'si a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_bb, plt_bb, 7, 14, 30+56, 1, 'si b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

            # # col 2
            plt_s = mg0
            plt_a = mg0_a
            plt_b = mg0_b
            plt_d = mg0_d
            plt_ss = mg
            plt_aa = mg_a
            plt_bb = mg_b
            plt_dd = mg_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 2, 1, 'mg s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Mg] concentration')
            chemplot(plt_d, plt_dd, 7, 7, 9, 1, 'mg d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 31, 1, 'mg a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_b, plt_bb, 7, 14, 32, 1, 'mg b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)


            plt_s = fe0
            plt_a = fe0_a
            plt_b = fe0_b
            plt_d = fe0_d
            plt_ss = fe
            plt_aa = fe_a
            plt_bb = fe_b
            plt_dd = fe_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 30, 1, 'fe s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Fe] concentration')
            chemplot(plt_d, plt_dd, 7, 7, 37, 1, 'fe d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 31+56, 1, 'fe a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_bb, plt_bb, 7, 14, 32+56, 1, 'fe b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

            # # col 3
            plt_s = k0
            plt_a = k0_a
            plt_b = k0_b
            plt_d = k0_d
            plt_ss = k1
            plt_aa = k1_a
            plt_bb = k1_b
            plt_dd = k1_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 3, 1, 'k s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[K] concentration')
            chemplot(plt_d, plt_dd, 7, 7, 10, 1, 'k d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 33, 1, 'k a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_b, plt_bb, 7, 14, 34, 1, 'k b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)


            plt_s = alk0
            plt_a = alk0_a
            plt_b = alk0_b
            plt_d = alk0_d
            plt_ss = alk
            plt_aa = alk_a
            plt_bb = alk_b
            plt_dd = alk_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 31, 1, 'alk s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='Alkalinity')
            chemplot(plt_d, plt_dd, 7, 7, 38, 1, 'alk d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 33+56, 1, 'alk a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_bb, plt_bb, 7, 14, 34+56, 1, 'alk b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

            # # col 4
            plt_s = ph0
            plt_a = ph0_a
            plt_b = ph0_b
            plt_d = ph0_d
            plt_ss = ph
            plt_aa = ph_a
            plt_bb = ph_b
            plt_dd = ph_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 4, 1, 'ph_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='pH')
            chemplot(plt_d, plt_dd, 7, 7, 11, 1, 'ph_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 35, 1, 'ph_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_b, plt_bb, 7, 14, 36, 1, 'ph_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)


            plt_s = alt_vol0
            plt_a = alt_vol0_a
            plt_b = alt_vol0_b
            plt_d = alt_vol0_d
            plt_ss = alt_vol
            plt_aa = alt_vol_a
            plt_bb = alt_vol_b
            plt_dd = alt_vol_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 32, 1, 'alt s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='Alteration volume')
            chemplot(plt_d, plt_dd, 7, 7, 39, 1, 'alt d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 35+56, 1, 'alt a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_bb, plt_bb, 7, 14, 36+56, 1, 'alt b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)


            # # col 5
            plt_s = al0
            plt_a = al0_a
            plt_b = al0_b
            plt_d = al0_d
            plt_ss = al
            plt_aa = al_a
            plt_bb = al_b
            plt_dd = al_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 5, 1, 'al s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Al] concentration')
            chemplot(plt_s, plt_dd, 7, 7, 12, 1, 'al d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 37, 1, 'al a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_b, plt_bb, 7, 14, 38, 1, 'al b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)


            plt_s = dic0
            plt_a = dic0_a
            plt_b = dic0_b
            plt_d = dic0_d
            plt_ss = dic
            plt_aa = dic_a
            plt_bb = dic_b
            plt_dd = dic_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 33, 1, 'dic s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='DIC')
            chemplot(plt_d, plt_dd, 7, 7, 40, 1, 'dic d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 37+56, 1, 'dic a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_bb, plt_bb, 7, 14, 38+56, 1, 'dic b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)


            # # col 6
            plt_s = pri_total0
            plt_a = pri_total0_a
            plt_b = pri_total0_b
            plt_d = pri_total0_d
            plt_ss = pri_total
            plt_aa = pri_total_a
            plt_bb = pri_total_b
            plt_dd = pri_total_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 6, 1, 'pri s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='pri_total')
            chemplot(plt_s, plt_dd, 7, 7, 13, 1, 'pri d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 39, 1, 'pri a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_b, plt_bb, 7, 14, 40, 1, 'pri b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

            plt_s = na0
            plt_a = na0_a
            plt_b = na0_b
            plt_d = na0_d
            plt_ss = na
            plt_aa = na_a
            plt_bb = na_b
            plt_dd = na_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 34, 1, 'na s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Na] concentration')
            chemplot(plt_d, plt_dd, 7, 7, 41, 1, 'na d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 39+56, 1, 'na a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_bb, plt_bb, 7, 14, 40+56, 1, 'na b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)


            # # col 7
            plt_s = cl0
            plt_a = cl0_a
            plt_b = cl0_b
            plt_d = cl0_d
            plt_ss = cl
            plt_aa = cl_a
            plt_bb = cl_b
            plt_dd = cl_d
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 7, 7, 1, 'cl s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Cl] concentration')
            chemplot(plt_s, plt_dd, 7, 7, 14, 1, 'cl d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 14, 41, 1, 'cl a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_b, plt_bb, 7, 14, 42, 1, 'cl b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

            # plt_s = ph_fix0
            # plt_a = ph_fix0_a
            # plt_b = ph_fix0_b
            # plt_d = ph_fix0_b
            # plt_ss = ph_fix
            # plt_aa = ph_fix_a
            # plt_bb = ph_fix_b
            # plt_dd = ph_fix_b
            # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            # c_min = np.min(all_ch)
            # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            # c_max = np.max(all_ch)
            #
            # chemplot(plt_s, plt_ss, 7, 7, 35, 1, 'ph fix s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='pH fix')
            # #chemplot(plt_d, plt_dd, 7, 7, 42, 1, 'ph fix d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            # chemplot(plt_a, plt_aa, 7, 14, 41+56, 1, 'ph fix a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            # chemplot(plt_bb, plt_bb, 7, 14, 42+56, 1, 'ph fix b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

            plt.savefig(outpath+'jdf_block_sol/jdf_'+letter+'_block_sol_'+str(i+restart)+'.png')






            if not os.path.exists(outpath+'jdf_block_sec/'):
                os.makedirs(outpath+'jdf_block_sec/')



            #todo: FIGURE: jdf_sec_block
            print "jdf_sec_block"
            fig=plt.figure(figsize=(14.75,7.4))
            plt.subplots_adjust( wspace=0.05, bottom=0.05, top=0.97, left=0.01, right=0.99)

            the_list = len(any_min)
            if len(any_min) > 24:
                the_list = 24

            for am in range(the_list):

                if am < 8:
                    am_p = am
                    am_pp = 2*am+1
                    am_ppp = 2*(am-1)+1
                if am >= 8:
                    am_p = 32+(am-8)
                    am_pp = 64 + 2*(am-8)+1
                    am_ppp = 80 + 2*(am-7)
                if am >= 16:
                    am_p = 64+(am-16)
                    am_pp = 112 + 2*(am-8)+1
                    am_ppp = 80 + 2*(am-7)


                plt_s = secMat[:,:,any_min[am]]
                plt_a = secMat_a[:,:,any_min[am]]
                plt_b = secMat_b[:,:,any_min[am]]
                plt_d = secMat_d[:,:,any_min[am]]
                plt_ss = secStep[:,:,any_min[am]]
                plt_aa = secStep_a[:,:,any_min[am]]
                plt_bb = secStep_b[:,:,any_min[am]]
                plt_dd = secStep_d[:,:,any_min[am]]

                c_min = 0.0
                all_ch = [np.max(plt_s), np.max(plt_a), np.max(plt_b), np.max(plt_d)]
                c_max = np.max(all_ch)

                chemplot24(plt_s, plt_ss, 11, 8, 1+am_p, 1, 'solo', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title=secondary[any_min[am]])
                chemplot24(plt_d, plt_dd, 11, 8, 9+am_p, 1, 'dual', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
                chemplot24(plt_a, plt_aa, 11, 16, 32+am_pp, 1, 'chamber a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
                chemplot24(plt_bb, plt_bb, 11, 16, 33+am_pp, 1, 'chamber b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)


            plt.savefig(outpath+'jdf_block_sec/jdf_'+letter+'_block_sec_'+str(i+restart)+'.png',bbox_inches='tight')









            # if not os.path.exists(outpath+'jdf_block_roc/'):
            #     os.makedirs(outpath+'jdf_block_roc/')
            #
            #
            #
            # #todo: FIGURE: jdf_roc_block
            # print "jdf_roc_block"
            #
            # fig=plt.figure(figsize=(14.75,7.4))
            # plt.subplots_adjust( wspace=0.05, bottom=0.05, top=0.97, left=0.01, right=0.99)
            #
            # the_list = len(any_min)
            # if len(any_min) > 24:
            #     the_list = 24
            #
            # for am in range(the_list):
            #
            #     if am < 8:
            #         am_p = am
            #         am_pp = 2*am+1
            #         am_ppp = 2*(am-1)+1
            #     if am >= 8:
            #         am_p = 32+(am-8)
            #         am_pp = 64 + 2*(am-8)+1
            #         am_ppp = 80 + 2*(am-7)
            #     if am >= 16:
            #         am_p = 64+(am-16)
            #         am_pp = 112 + 2*(am-8)+1
            #         am_ppp = 80 + 2*(am-7)
            #
            #
            #     # plt_s = secMat[:,:,any_min[am]]
            #     # plt_a = secMat_a[:,:,any_min[am]]
            #     # plt_b = secMat_b[:,:,any_min[am]]
            #     # plt_d = secMat_d[:,:,any_min[am]]
            #     plt_s = dsecMat[:,:,any_min[am]]
            #     plt_a = dsecMat_a[:,:,any_min[am]]
            #     plt_b = dsecMat_b[:,:,any_min[am]]
            #     plt_d = dsecMat_d[:,:,any_min[am]]
            #     plt_ss = dsecStep[:,:,any_min[am]]
            #     plt_aa = dsecStep_a[:,:,any_min[am]]
            #     plt_bb = dsecStep_b[:,:,any_min[am]]
            #     plt_dd = dsecStep_d[:,:,any_min[am]]
            #
            #     c_min = 0.0
            #     all_ch = [np.max(plt_s), np.max(plt_a), np.max(plt_b), np.max(plt_d)]
            #     c_max = np.max(all_ch)
            #
            #     chemplot24(plt_s, plt_ss, 11, 8, 1+am_p, 1, 'solo roc', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title=secondary[any_min[am]])
            #     chemplot24(plt_d, plt_dd, 11, 8, 9+am_p, 1, 'dual roc', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #     chemplot24(plt_a, plt_aa, 11, 16, 32+am_pp, 1, 'chamber a roc', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #     chemplot24(plt_bb, plt_bb, 11, 16, 33+am_pp, 1, 'chamber b roc', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #
            #     # if np.any(plt_s > 0.0):
            #     #     chemplot24(plt_s, plt_ss, 11, 8, 1+am_p, 1, 'solo roc', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title=secondary[any_min[am]])
            #     # if np.any(plt_d > 0.0):
            #     #     chemplot24(plt_d, plt_dd, 11, 8, 9+am_p, 1, 'dual roc', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #     # if np.any(plt_a > 0.0):
            #     #     chemplot24(plt_a, plt_aa, 11, 16, 32+am_pp, 1, 'chamber a roc', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #     # if np.any(plt_b > 0.0):
            #     #     chemplot24(plt_bb, plt_bb, 11, 16, 33+am_pp, 1, 'chamber b roc', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #
            #
            # plt.savefig(outpath+'jdf_block_roc/jdf_'+letter+'_block_roc_'+str(i+restart)+'.png',bbox_inches='tight')
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #
            # if not os.path.exists(outpath+'jdf_block_roc_move/'):
            #     os.makedirs(outpath+'jdf_block_roc_move/')
            #
            #
            #
            # #todo: FIGURE: jdf_block_roc_move
            #
            # if not np.any(moves == i+1):
            #     print "jdf_block_roc_move"
            #
            #
            #     fig=plt.figure(figsize=(14.75,7.4))
            #     plt.subplots_adjust( wspace=0.05, bottom=0.05, top=0.97, left=0.01, right=0.99)
            #
            #     the_list = len(any_min)
            #     if len(any_min) > 24:
            #         the_list = 24
            #
            #     for am in range(the_list):
            #
            #         if am < 8:
            #             am_p = am
            #             am_pp = 2*am+1
            #             am_ppp = 2*(am-1)+1
            #         if am >= 8:
            #             am_p = 32+(am-8)
            #             am_pp = 64 + 2*(am-8)+1
            #             am_ppp = 80 + 2*(am-7)
            #         if am >= 16:
            #             am_p = 64+(am-16)
            #             am_pp = 112 + 2*(am-8)+1
            #             am_ppp = 80 + 2*(am-7)
            #
            #
            #         # plt_s = secMat[:,:,any_min[am]]
            #         # plt_a = secMat_a[:,:,any_min[am]]
            #         # plt_b = secMat_b[:,:,any_min[am]]
            #         # plt_d = secMat_d[:,:,any_min[am]]
            #         plt_s = dsecMat[:,:,any_min[am]]
            #         plt_a = dsecMat_a[:,:,any_min[am]]
            #         plt_b = dsecMat_b[:,:,any_min[am]]
            #         plt_d = dsecMat_d[:,:,any_min[am]]
            #         plt_ss = dsecStep[:,:,any_min[am]]
            #         plt_aa = dsecStep_a[:,:,any_min[am]]
            #         plt_bb = dsecStep_b[:,:,any_min[am]]
            #         plt_dd = dsecStep_d[:,:,any_min[am]]
            #
            #         c_min = 0.0
            #         all_ch = [np.max(plt_s), np.max(plt_a), np.max(plt_b), np.max(plt_d)]
            #         c_max = np.max(all_ch)
            #
            #         chemplot24(plt_s, plt_ss, 11, 8, 1+am_p, 1, 'solo roc MOVE', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title=secondary[any_min[am]])
            #         chemplot24(plt_d, plt_dd, 11, 8, 9+am_p, 1, 'dual roc MOVE', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #         chemplot24(plt_a, plt_aa, 11, 16, 32+am_pp, 1, 'a roc MOVE', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #         chemplot24(plt_bb, plt_bb, 11, 16, 33+am_pp, 1, 'b roc MOVE', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #
            #
            #     plt.savefig(outpath+'jdf_block_roc_move/'+letter+'_block_roc_move_'+str(i+restart)+'.png',bbox_inches='tight')
            #
            #






            if not os.path.exists(outpath+'jdf_block_pri/'):
                os.makedirs(outpath+'jdf_block_pri/')


            min_stand = 0.001


            #todo: FIGURE: jdf_pri_block
            print "jdf_pri_block"
            fig=plt.figure(figsize=(13.0,7.0))
            #plt.subplots_adjust( wspace=0.03, bottom=0.15, top=0.97, left=0.01, right=0.99)
            plt.subplots_adjust( wspace=0.03, bottom=0.1, top=0.97, left=0.01, right=0.99)

            # # col 1
            # plt_s = pri_total0
            # plt_a = pri_total0_d
            # plt_b = pri_total0_d
            # plt_d = pri_total0_d
            # plt_ss = pri_total
            # plt_aa = pri_total_d
            # plt_bb = pri_total_d
            # plt_dd = pri_total_d
            plt_s = pri_total0/np.max(pri_total0)
            plt_a = pri_total0_a/np.max(pri_total0_a)
            plt_b = pri_total0_b/np.max(pri_total0_b)
            plt_d = pri_total0_d/np.max(pri_total0_d)
            plt_ss = pri_total/np.max(pri_total)
            plt_aa = pri_total_a/np.max(pri_total_a)
            plt_bb = pri_total_b/np.max(pri_total_b)
            plt_dd = pri_total_d/np.max(pri_total_d)


            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 5, 1, 1, 'basalt remaining s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='basalt remaining')
            chemplot(plt_d, plt_dd, 7, 5, 6, 1, 'basalt remaining d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 10, 21, 1, 'basalt remaining a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_b, plt_bb, 7, 10, 22, 1, 'basalt remaining b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)




            #if pri_5_toggle == 1:

            # # col 2
            # plt_s = glass0
            # plt_a = glass0_a
            # plt_b = glass0_a
            # plt_d = glass0_a
            # plt_ss = glass
            # plt_aa = glass_a
            # plt_bb = glass_a
            # plt_dd = glass_a



            plt_s = glass0/np.max(glass0)
            plt_a = glass0_a/np.max(glass0_a)
            plt_b = glass0_b/np.max(glass0_b)
            plt_d = glass0_d/np.max(glass0_d)


            plt_ss = glass/np.max(glass)
            plt_aa = glass_a/np.max(glass_a)
            plt_bb = glass_b/np.max(glass_b)
            plt_dd = glass_d/np.max(glass_d)

            # if pri_toggle[5] == 0:
            #     plt_s[1,1] = min_stand
            #     plt_ss[1,1] = min_stand
            # if pri_toggle_d[5] == 0:
            #     plt_d[1,1] = min_stand
            #     plt_dd[1,1] = min_stand
            # if pri_toggle_a[5] == 0:
            #     plt_a[1,1] = min_stand
            #     plt_aa[1,1] = min_stand
            # if pri_toggle_b[5] == 0:
            #     plt_b[1,1] = min_stand
            #     plt_bb[1,1] = min_stand





            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = np.min(all_ch)
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            c_max = np.max(all_ch)

            # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_d[plt_d>0.0])]
            # c_min = np.min(all_ch)
            # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_d[plt_d>0.0])]
            # c_max = np.max(all_ch)

            chemplot(plt_s, plt_ss, 7, 5, 2, 1, 'glass_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='glass')
            chemplot(plt_d, plt_dd, 7, 5, 7, 1, 'glass_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 10, 23, 1, 'glass_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_bb, plt_bb, 7, 10, 24, 1, 'glass_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)




            # if pri_4_toggle == 1:
            #
            #     # # col 3
            #     # plt_s = ol0
            #     # plt_a = ol0_b
            #     # plt_b = ol0_b
            #     # plt_d = ol0_b
            #     # plt_ss = ol
            #     # plt_aa = ol_b
            #     # plt_bb = ol_b
            #     # plt_dd = ol_b
            #     plt_s = ol0/np.max(ol0)
            #     plt_a = ol0_a/np.max(ol0_a)
            #     plt_b = ol0_b/np.max(ol0_b)
            #     plt_d = ol0_d/np.max(ol0_d)
            #     plt_ss = ol/np.max(ol)
            #     plt_aa = ol_a/np.max(ol_a)
            #     plt_bb = ol_b/np.max(ol_b)
            #     plt_dd = ol_d/np.max(ol_d)
            #
            #     if pri_toggle[4] == 0:
            #         plt_s[1,1] = min_stand
            #         plt_ss[1,1] = min_stand
            #     if pri_toggle_d[4] == 0:
            #         plt_d[1,1] = min_stand
            #         plt_dd[1,1] = min_stand
            #     if pri_toggle_a[4] == 0:
            #         plt_a[1,1] = min_stand
            #         plt_aa[1,1] = min_stand
            #     if pri_toggle_b[4] == 0:
            #         plt_b[1,1] = min_stand
            #         plt_bb[1,1] = min_stand
            #
            #
            #     all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            #     c_min = np.min(all_ch[all_ch>min_stand])
            #     all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            #     c_max = np.max(all_ch)
            #
            #     chemplot(plt_s, plt_ss, 7, 5, 3, 1, 'ol_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='olivine')
            #     chemplot(plt_d, plt_dd, 7, 5, 8, 1, 'ol_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #     chemplot(plt_a, plt_aa, 7, 10, 25, 1, 'ol_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #     chemplot(plt_b, plt_bb, 7, 10, 26, 1, 'ol_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #
            #
            #
            # if pri_3_toggle == 1:
            #
            #     # # col 4
            #     # plt_s = pyr0
            #     # plt_a = pyr0_b
            #     # plt_b = pyr0_b
            #     # plt_d = pyr0_b
            #     # plt_ss = pyr
            #     # plt_aa = pyr_b
            #     # plt_bb = pyr_b
            #     # plt_dd = pyr_b
            #     plt_s = pyr0/np.max(pyr0)
            #     plt_a = pyr0_a/np.max(pyr0_a)
            #     plt_b = pyr0_b/np.max(pyr0_b)
            #     plt_d = pyr0_d/np.max(pyr0_d)
            #     plt_ss = pyr/np.max(pyr)
            #     plt_aa = pyr_a/np.max(pyr_a)
            #     plt_bb = pyr_b/np.max(pyr_b)
            #     plt_dd = pyr_d/np.max(pyr_d)
            #
            #     if pri_toggle[3] == 0:
            #         plt_s[1,1] = min_stand
            #         plt_ss[1,1] = min_stand
            #     if pri_toggle_d[3] == 0:
            #         plt_d[1,1] = min_stand
            #         plt_dd[1,1] = min_stand
            #     if pri_toggle_a[3] == 0:
            #         plt_a[1,1] = min_stand
            #         plt_aa[1,1] = min_stand
            #     if pri_toggle_b[3] == 0:
            #         plt_b[1,1] = min_stand
            #         plt_bb[1,1] = min_stand
            #
            #
            #     all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            #     c_min = np.amin(all_ch[all_ch>min_stand])
            #     all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            #     c_max = np.amax(all_ch[all_ch>0.0])
            #     chemplot(plt_s, plt_ss, 7, 5, 4, 1, 'pyr_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='pyroxene')
            #     chemplot(plt_d, plt_dd, 7, 5, 9, 1, 'pyr_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #     chemplot(plt_a, plt_aa, 7, 10, 27, 1, 'pyr_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #     chemplot(plt_b, plt_bb, 7, 10, 28, 1, 'pyr_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #
            #
            # if pri_2_toggle == 1:
            #
            #     # # col 5
            #     # plt_s = plag0
            #     # plt_a = plag0_b
            #     # plt_b = plag0_b
            #     # plt_d = plag0_b
            #     # plt_ss = plag
            #     # plt_aa = plag_b
            #     # plt_bb = plag_b
            #     # plt_dd = plag_b
            #     plt_s = plag0/np.max(plag0)
            #     plt_a = plag0_a/np.max(plag0_a)
            #     plt_b = plag0_b/np.max(plag0_b)
            #     plt_d = plag0_d/np.max(plag0_d)
            #     plt_ss = plag/np.max(plag)
            #     plt_aa = plag_a/np.max(plag_a)
            #     plt_bb = plag_b/np.max(plag_b)
            #     plt_dd = plag_d/np.max(plag_d)
            #
            #
            #     if pri_toggle[2] == 0:
            #         plt_s[1,1] = min_stand
            #         plt_ss[1,1] = min_stand
            #     if pri_toggle_d[2] == 0:
            #         plt_d[1,1] = min_stand
            #         plt_dd[1,1] = min_stand
            #     if pri_toggle_a[2] == 0:
            #         plt_a[1,1] = min_stand
            #         plt_aa[1,1] = min_stand
            #     if pri_toggle_b[2] == 0:
            #         plt_b[1,1] = min_stand
            #         plt_bb[1,1] = min_stand
            #
            #
            #     all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            #     c_min = np.min(all_ch[all_ch>min_stand])
            #     all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
            #     c_max = np.max(all_ch)
            #
            #     chemplot(plt_s, plt_ss, 7, 5, 5, 1, 'plag_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='plagioclase')
            #     chemplot(plt_d, plt_dd, 7, 5, 10, 1, 'plag_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #     chemplot(plt_a, plt_aa, 7, 10, 29, 1, 'plag_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            #     chemplot(plt_b, plt_bb, 7, 10, 30, 1, 'plag_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

            #fig.set_tight_layout(True)
            # plt.subplots_adjust( wspace=0.05 , bottom=0.04, top=0.97, left=0.03, right=0.975)
            plt.savefig(outpath+'jdf_block_pri/jdf_'+letter+'_block_pri_'+str(i+restart)+'.png',bbox_inches='tight')



            #todo:  FIGURE jdf_cont_sec
            print "jdf_cont_sec"
            if not os.path.exists(outpath+'jdf_cont_sec/'):
                os.makedirs(outpath+'jdf_cont_sec/')


            d_alpha = 0.5
            f_colors=['#eb4dcd', 'rgb(73, 106, 163)', 'rgb(204, 120, 32)', 'rgb(243, 255, 20)', 'rgb(0, 54, 147)']

            bin_u_smec = [1] #[11, 12, 6, 18, 22]
            bin_phil = [21]
            bin_sap = [2, 11, 33, 36, 37]
            bin_fe_sap = [36, 37]
            bin_verm = [19, 24, 34]
            bin_talc = [26]
            bin_nont = [12, 13]
            bin_celad = [3, 14]
            bin_chlor = []
            bin_goet = [7]
            bin_hem = [17]
            bin_pyrite = [5]
            bin_pyrr = [35]
            bin_u_zeo = [20, 25]


            c_u_smec = dict(name='u smectites', ind=[1],
                            min_color='#FFFFFF',
                            hatching='',
                            bg_alpha=1.0,
                            ed_col='#650300')

            c_phil = dict(name='phillipsite', ind=[21],
                            min_color='#888888',
                            hatching='',
                            bg_alpha=0.4,
                            ed_col='w')

            c_sap = dict(name='mg-ca-saponites', ind=[2, 11, 33, 36, 37],
                            min_color='#eb4dcd',
                            hatching='',
                            bg_alpha=d_alpha,
                            ed_col='w')


            c_pyrite = dict(name='pyrite', ind=[5],
                            min_color='#ffffff',
                            hatching='',
                            bg_alpha=1.0,
                            ed_col='#a38900')

            c_talc = dict(name='talc', ind=[26],
                            min_color='#fef40f',
                            hatching='',
                            bg_alpha=d_alpha,
                            ed_col='w')

            c_nont = dict(name='nontronite', ind=[12, 13],
                            min_color='#03016b',
                            hatching='',
                            bg_alpha=d_alpha,
                            ed_col='w')

            c_celad = dict(name='celadonite', ind=[3, 14],
                            min_color='#FFFFFF',
                            hatching='',
                            bg_alpha=1.0,
                            ed_col='#5c9400')

            c_goet = dict(name='goethite+hem', ind=[7,17],

                            min_color='#FFFFFF',
                            hatching='////',
                            bg_alpha=d_alpha,
                            ed_col='k')

            c_u_zeo = dict(name='u zeolites', ind=[20, 25],
                            min_color='#FFFFFF',
                            hatching='...',
                            bg_alpha=1.0,
                            ed_col='k')

            c_chlor = dict(name='chlorite', ind=[31, 32],
                            min_color='#FFFFFF',
                            hatching='.',
                            bg_alpha=1.0,
                            ed_col='k')



            c_fe_sap = dict(name='fe-saponites', ind=[36, 37],
                            min_color='#eb4dcd',
                            hatching='',
                            bg_alpha=d_alpha,
                            ed_col='w')

            c_verm = dict(name='vermiculite', ind=[19, 24, 34],
                            min_color='#FFFFFF',
                            hatching='',
                            bg_alpha=1.0,
                            ed_col='#347993')

            c_hem = dict(name='hematite', ind=[17],
                            min_color='#eb4dcd',
                            hatching='',
                            bg_alpha=d_alpha,
                            ed_col='w')

            c_pyrr = dict(name='pyrrhotite', ind=[5],
                            min_color='#eb4dcd',
                            hatching='',
                            bg_alpha=d_alpha,
                            ed_col='w')






            # c_fe_sap = 'rgb(204, 120, 32)'
            # c_verm = 'rgb(243, 255, 20)'
            # c_talc = 'rgb(0, 54, 147)'
            # c_nont = 'rgb(140, 203, 49)'
            # c_celad = 'rgb(64, 64, 64)'



            #todo: FIGURE: jdf_Sec_Cont (binary contour plot)
            print "jdf_Sec_Cont plot"
            fig=plt.figure(figsize=(11.0,4.6))

            # # # commented out 11/30/17 by NAVAH X

            # chemcont_vol(alt_vol0, alt_vol, 3, 2, 2, 10, 'solo chamber', xtix=1, ytix=0,frame_lines=1, cb=1, cb_title='alteration vol')
            #
            # chemcont_vol(alt_vol0_d, alt_vol_d, 3, 2, 4, 10, 'dual chamber', xtix=0, ytix=0,frame_lines=1)
            #
            # chemcont_vol(alt_vol0_a, alt_vol_a, 3, 4, 11, 10, 'chamber a only', xtix=0, ytix=0,frame_lines=1)
            #
            # chemcont_vol(alt_vol0_b, alt_vol_b, 3, 4, 12, 10, 'chamber b only', xtix=0, ytix=0,frame_lines=1)







            bind = c_sap
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, 'solo chamber', xtix=0, ytix=1, frame_lines=1,
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg1 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])


            bind = c_u_smec
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg2 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])


            bind = c_phil
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg3 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])


            bind = c_pyrite
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg4 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])


            bind = c_talc
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg5 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])


            bind = c_nont
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg6 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])


            bind = c_celad
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg7 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])


            bind = c_goet
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg8 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])


            bind = c_u_zeo
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg9 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])


            bind = c_chlor
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg10 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])


            bind = c_verm
            if np.any(secStep[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            lg11 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])








            # bind = c_chlor
            # chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            # min_color=bind['min_color'], to_hatch=bind['to_hatch'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], to_fill=bind['to_fill'])
            #
            # lg10 = Patch(facecolor=bind['min_color'], label=bind['name'], alpha=bind['bg_alpha'], hatch=bind['hatching'], edgecolor=bind['ed_col'])

            #
            # bind = c_fe_sap
            # chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            # min_color=bind['min_color'], to_hatch=bind['to_hatch'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], to_fill=bind['to_fill'])
            #
            # bind = c_verm
            # chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            # min_color=bind['min_color'], to_hatch=bind['to_hatch'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], to_fill=bind['to_fill'])
            #
            # bind = c_hem
            # chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            # min_color=bind['min_color'], to_hatch=bind['to_hatch'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], to_fill=bind['to_fill'])
            #
            # bind = c_pyrr
            # chemcont(np.sum(secMat[:,:,bind['ind']],axis=2), np.sum(secStep[:,:,bind['ind']],axis=2), 3, 2, 1, 1, '',
            # min_color=bind['min_color'], to_hatch=bind['to_hatch'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], to_fill=bind['to_fill'])







            # # commented out 11/30/17 by NAVAH X



            b_legend = plt.legend([lg1, lg2, lg3, lg4, lg5, lg6, lg7, lg8, lg9, lg10, lg11],[lg1.get_label(), lg2.get_label(), lg3.get_label(), lg4.get_label(), lg5.get_label(), lg6.get_label(), lg7.get_label(), lg8.get_label(), lg9.get_label(), lg10.get_label(), lg11.get_label()],fontsize=10,ncol=3,bbox_to_anchor=(0.5, -2.5),loc=8)
            # b_legend = plt.legend([lg1, lg2, lg4, lg8, lg10],[lg1.get_label(), lg2.get_label(),  lg4.get_label(),  lg8.get_label(),  lg10.get_label()],fontsize=10,ncol=3,bbox_to_anchor=(0.5, -3.2),loc=8)
            b_legend.get_frame().set_linewidth(0.0)

            cont_inter = 1



            bind = c_sap
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, 'dual chamber', xtix=1, ytix=1, frame_lines=1,
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_u_smec
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_phil
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_pyrite
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_talc
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_nont
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_celad
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_goet
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_u_zeo
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_chlor
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_verm
            if np.any(secStep_d[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_d[:,:,bind['ind']],axis=2), np.sum(secStep_d[:,:,bind['ind']],axis=2), 3, 2, 3, cont_inter, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])







            # # commented out 11/30/17 by NAVAH X


            bind = c_sap
            if np.any(secStep_a[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, 'chamber a only', xtix=0, ytix=0, frame_lines=1,
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_u_smec
            if np.any(secStep_a[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_phil
            if np.any(secStep_a[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_pyrite
            if np.any(secStep_a[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_talc
            if np.any(secStep_a[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_nont
            if np.any(secStep_a[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_celad
            if np.any(secStep_a[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_goet
            if np.any(secStep_a[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_u_zeo
            if np.any(secStep_a[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_chlor
            chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_verm
            if np.any(secStep_a[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_a[:,:,bind['ind']],axis=2), np.sum(secStep_a[:,:,bind['ind']],axis=2), 3, 2, 2, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            #
            #
            #
            #
            #
            #
            #
            #
            #
            bind = c_sap
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, 'chamber b only', xtix=1, ytix=0, frame_lines=1,
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_u_smec
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_phil
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_pyrite
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_talc
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_nont
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_celad
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_goet
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_u_zeo
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_chlor
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])

            bind = c_verm
            if np.any(secStep_b[:,:,bind['ind']] > 0.0):
                chemcont(np.sum(secMat_b[:,:,bind['ind']],axis=2), np.sum(secStep_b[:,:,bind['ind']],axis=2), 3, 2, 4, 1, '',
            min_color=bind['min_color'], hatching=bind['hatching'], bg_alpha=bind['bg_alpha'], ed_col=bind['ed_col'])




            plt.subplots_adjust( wspace=0.05 , bottom=0.2, top=0.95, left=0.03, right=0.975)
            plt.savefig(outpath+'/jdf_cont_sec/jdf_'+letter+'_cont_sec_'+str(i+restart)+'.png',bbox_inches='tight')

            if i == 50 or i == 49:
                plt.savefig(outpath+'/jdf_cont_sec/jdf_'+letter+'_cont_sec_'+str(i+restart+1)+'.eps',bbox_inches='tight')


            bar_bins = 5





            #
            # print "jdf_ph_fix plot"
            # #tod0: FIGURE: jdf_ph_fix
            #
            # fig=plt.figure(figsize=(16.0,8.0))
            #
            # ax=fig.add_subplot(2, 2, 1, frameon=False, aspect=asp*4)
            # varStep = ph_fix
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(ph_fix0[ph_fix0>0]),vmax=np.max(ph_fix0))
            # plt.ylim([-505.0,-325.0])
            # # plt.colorbar(orientation='horizontal',format=ticker.FuncFormatter(fmt))
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('ph_fix solo')
            # plt.xticks([])
            # plt.yticks([])
            #
            #
            # ax=fig.add_subplot(2, 2, 3, frameon=False, aspect=asp*4)
            # varStep = ph_fix_a
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(ph_fix0_a[ph_fix0_a>0]),vmax=np.max(ph_fix0_a))
            # plt.ylim([-505.0,-325.0])
            # # plt.colorbar(orientation='horizontal',format=ticker.FuncFormatter(fmt))
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('ph_fix a')
            # plt.xticks([])
            # plt.yticks([])
            #
            #
            # ax=fig.add_subplot(2, 2, 4, frameon=False, aspect=asp*4)
            # varStep = ph_fix_b
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(ph_fix0_b[ph_fix0_b>0]),vmax=np.max(ph_fix0_b))
            # plt.ylim([-505.0,-325.0])
            # # plt.colorbar(orientation='horizontal',format=ticker.FuncFormatter(fmt))
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('ph_fix b')
            # plt.xticks([])
            # plt.yticks([])
            #
            #
            # plt.subplots_adjust( wspace=0.1 , bottom=0.05, top=0.95, left=0.03, right=0.98)
            # plt.savefig(outpath+'jdf_'+letter+'_ph_fix_'+str(i+restart)+'.png')





            # print "jdf_SolW_Phi plot"
            # #tod0: FIGURE: jdf_SolW_Phi
            #
            # fig=plt.figure(figsize=(16.0,8.0))
            #
            # ax=fig.add_subplot(4, 4, 1, frameon=False, aspect=asp*4)
            # varStep = solw
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(solw0[solw0>0.0]),vmax=np.max(solw0))
            # plt.ylim([-505.0,-325.0])
            # # plt.colorbar(orientation='horizontal',format=ticker.FuncFormatter(fmt))
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('sol_w solo')
            # plt.xticks([])
            # plt.yticks([])
            #
            # ax=fig.add_subplot(4, 4, 2, frameon=False, aspect=asp*4)
            # varStep = solw_d
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(solw0_d[solw0_d>0.0]),vmax=np.max(solw0_d))
            # plt.ylim([-505.0,-325.0])
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('sol_w dual')
            # plt.xticks([])
            # plt.yticks([])
            #
            # ax=fig.add_subplot(4, 4, 5, frameon=False, aspect=asp*4)
            # varStep = solw_a
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(solw0_a[solw0_a>0.0]),vmax=np.max(solw0_a))
            # plt.ylim([-505.0,-325.0])
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('sol_w a')
            # plt.xticks([])
            # plt.yticks([])
            #
            # ax=fig.add_subplot(4, 4, 6, frameon=False, aspect=asp*4)
            # varStep = solw_b
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(solw0_b[solw0_b>0.0]),vmax=np.max(solw0_b))
            # plt.ylim([-505.0,-325.0])
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('sol_w b')
            # plt.xticks([])
            # plt.yticks([])
            #
            #
            #
            # phi_calc0 = solw0*1000.0/ (solw0*1000.0 + pri_total0 + secVol0)
            # phi_calc0[np.isinf(phi_calc0)] = 0.0
            # phi_calc0[np.isnan(phi_calc0)] = 0.0
            # phi_calc0[phi_calc0 == 1.0] = 0.0
            # phi_calc = solw*1000.0/ (solw*1000.0 + pri_total + secVol)
            # phi_calc[np.isinf(phi_calc)] = 0.0
            # phi_calc[np.isnan(phi_calc)] = 0.0
            # phi_calc[phi_calc == 1.0] = 0.0
            # ax=fig.add_subplot(4, 4, 3, frameon=False, aspect=asp*4)
            # varStep = phi_calc
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(phi_calc0[phi_calc0>0.0]),vmax=np.max(phi_calc0))
            # plt.ylim([-505.0,-325.0])
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('phi_calc solo')
            # plt.xticks([])
            # plt.yticks([])
            #
            # phi_calc0 = solw0_d*1000.0/ (solw0_d*1000.0 + pri_total0_d + secVol0_d)
            # phi_calc0[np.isinf(phi_calc0)] = 0.0
            # phi_calc0[np.isnan(phi_calc0)] = 0.0
            # phi_calc0[phi_calc0 == 1.0] = 0.0
            # phi_calc = solw_d*1000.0/ (solw_d*1000.0 + pri_total_d + secVol_d)
            # phi_calc[np.isinf(phi_calc)] = 0.0
            # phi_calc[np.isnan(phi_calc)] = 0.0
            # phi_calc[phi_calc == 1.0] = 0.0
            # ax=fig.add_subplot(4, 4, 4, frameon=False, aspect=asp*4)
            # varStep = phi_calc
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(phi_calc0[phi_calc0>0.0]),vmax=np.max(phi_calc0))
            # plt.ylim([-505.0,-325.0])
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('phi_calc dual')
            # plt.xticks([])
            # plt.yticks([])
            #
            # phi_calc0 = solw0_a*1000.0/ (solw0_a*1000.0 + pri_total0_a + secVol0_a)
            # phi_calc0[np.isinf(phi_calc0)] = 0.0
            # phi_calc0[np.isnan(phi_calc0)] = 0.0
            # phi_calc0[phi_calc0 == 1.0] = 0.0
            # phi_calc = solw_a*1000.0/ (solw_a*1000.0 + pri_total_a + secVol_a)
            # phi_calc[np.isinf(phi_calc)] = 0.0
            # phi_calc[np.isnan(phi_calc)] = 0.0
            # phi_calc[phi_calc == 1.0] = 0.0
            # ax=fig.add_subplot(4, 4, 7, frameon=False, aspect=asp*4)
            # varStep = phi_calc
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(phi_calc0[phi_calc0>0.0]),vmax=np.max(phi_calc0))
            # plt.ylim([-505.0,-325.0])
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('phi_calc chamber a')
            # plt.xticks([])
            # plt.yticks([])
            #
            # phi_calc0 = solw0_b*1000.0/ (solw0_b*1000.0 + pri_total0_b + secVol0_b)
            # phi_calc0[np.isinf(phi_calc0)] = 0.0
            # phi_calc0[np.isnan(phi_calc0)] = 0.0
            # phi_calc0[phi_calc0 == 1.0] = 0.0
            # phi_calc = solw_b*1000.0/ (solw_b*1000.0 + pri_total_b + secVol_b)
            # phi_calc[np.isinf(phi_calc)] = 0.0
            # phi_calc[np.isnan(phi_calc)] = 0.0
            # phi_calc[phi_calc == 1.0] = 0.0
            # ax=fig.add_subplot(4, 4, 8, frameon=False, aspect=asp*4)
            # varStep = phi_calc
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(phi_calc0[phi_calc0>0.0]),vmax=np.max(phi_calc0))
            # plt.ylim([-505.0,-325.0])
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('phi_calc chamber b')
            # plt.xticks([])
            # plt.yticks([])
            #
            #
            #
            #
            # ax=fig.add_subplot(4, 4, 11, frameon=False, aspect=asp*4)
            # varStep = phiCalcIn
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(phiCalcIn0[phiCalcIn0>0.0]),vmax=np.max(phiCalcIn0[phiCalcIn0<1.0]))
            # plt.ylim([-505.0,-325.0])
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('phiCalcIn')
            # plt.xticks([])
            # plt.yticks([])
            #
            #
            # ax=fig.add_subplot(4, 4, 15, frameon=False, aspect=asp*4)
            # varStep = phiCalcIn_a
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varStep
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=np.min(phiCalcIn0_a[phiCalcIn0_a>0.0]),vmax=np.max(phiCalcIn0_a[phiCalcIn0_a<1.0]))
            # plt.ylim([-505.0,-325.0])
            # cb = plt.colorbar(orientation='horizontal')
            # tick_locator = ticker.MaxNLocator(nbins=bar_bins)
            # cb.locator = tick_locator
            # cb.update_ticks()
            # plt.title('phiCalcIn_a')
            # plt.xticks([])
            # plt.yticks([])
            #
            # plt.subplots_adjust( wspace=0.1 , bottom=0.05, top=0.95, left=0.03, right=0.98)
            # plt.savefig(outpath+'jdf_'+letter+'_SolW_Phi_'+str(i+restart)+'.png')






    #hack: chem0 switch
    chem0 = 0

    if chem0 == 0:


        if alt_plots == 1:


            if not os.path.exists(outpath+'jdf_ternary/'):
                os.makedirs(outpath+'jdf_ternary/')



            #todo: FIG: jdf_ternary
            #fig=plt.figure(figsize=(9.0,9.0))
            print "jdf_ternary"
            fig=plt.figure(figsize=(11.0,2.25))


            for k in range(len(yCell)):
                for j in range(len(xCell)):

                    ternK[k,j,i] = 1.0*secStep[k,j,14]
                    ternMg[k,j,i] = 0.165*secStep[k,j,22] + 5.0*secStep[k,j,31] + 3.0*secStep[k,j,11] + 3.165*secStep[k,j,2] + 3.0*secStep[k,j,33]
                    ternFe[k,j,i] = 1.0*secStep[k,j,14] + 2.0*secStep[k,j,22] + 2.0*secStep[k,j,17] + 1.0*secStep[k,j,5] + 2.0*secStep[k,j,15]

                    ternK_d[k,j,i] = 1.0*secStep_d[k,j,14]
                    ternMg_d[k,j,i] = 0.165*secStep_d[k,j,22] + 5.0*secStep_d[k,j,31] + 3.0*secStep_d[k,j,11] + 3.165*secStep_d[k,j,2] + 3.0*secStep_d[k,j,33]
                    ternFe_d[k,j,i] = 1.0*secStep_d[k,j,14] + 2.0*secStep_d[k,j,22] + 2.0*secStep_d[k,j,17] + 1.0*secStep_d[k,j,5] + 2.0*secStep_d[k,j,15]

                    ternK_a[k,j,i] = 1.0*secStep_a[k,j,14]
                    ternMg_a[k,j,i] = 0.165*secStep_a[k,j,22] + 5.0*secStep_a[k,j,31] + 3.0*secStep_a[k,j,11] + 3.165*secStep_a[k,j,2] + 3.0*secStep_a[k,j,33]
                    ternFe_a[k,j,i] = 1.0*secStep_a[k,j,14] + 2.0*secStep_a[k,j,22] + 2.0*secStep_a[k,j,17] + 1.0*secStep_a[k,j,5] + 2.0*secStep_a[k,j,15]

                    ternK_b[k,j,i] = 1.0*secStep_b[k,j,14]
                    ternMg_b[k,j,i] = 0.165*secStep_b[k,j,22] + 5.0*secStep_b[k,j,31] + 3.0*secStep_b[k,j,11] + 3.165*secStep_b[k,j,2] + 3.0*secStep_b[k,j,33]
                    ternFe_b[k,j,i] = 1.0*secStep_b[k,j,14] + 2.0*secStep_b[k,j,22] + 2.0*secStep_b[k,j,17] + 1.0*secStep_b[k,j,5] + 2.0*secStep_b[k,j,15]



            tern_count = 0

            for k in range(len(yCell)):
                for j in range(len(xCell)):
                    tern_list[tern_count,i,0] = 39.0*ternK[k,j,i]
                    tern_list[tern_count,i,1] = 23.0*ternMg[k,j,i]
                    tern_list[tern_count,i,2] = 56.0*ternFe[k,j,i]
                    #tern_list[tern_count,i,:] = tern_list[tern_count,i,:]/np.sum(tern_list[tern_count,i,:])
                    if np.max(tern_list[tern_count,i,:]) > 0.0:
                        tern_list[tern_count,i,:] = tern_list[tern_count,i,:]/(1.0*tern_list[tern_count,i,0] + 1.0*tern_list[tern_count,i,1] + 1.0*tern_list[tern_count,i,2])
                        # if i == 20:
                        #     print "S" , tern_list[tern_count,i,:]

                    tern_list_d[tern_count,i,0] = 39.0*ternK_d[k,j,i]
                    tern_list_d[tern_count,i,1] = 23.0*ternMg_d[k,j,i]
                    tern_list_d[tern_count,i,2] = 56.0*ternFe_d[k,j,i]
                    #tern_list_d[tern_count,i,:] = tern_list_d[tern_count,i,:]/np.sum(tern_list_d[tern_count,i,:])
                    if np.max(tern_list_d[tern_count,i,:]) > 0.0:
                        tern_list_d[tern_count,i,:] = tern_list_d[tern_count,i,:]/(1.0*tern_list_d[tern_count,i,0] + 1.0*tern_list_d[tern_count,i,1] + 1.0*tern_list_d[tern_count,i,2])
                        # if i == 20:
                        #     print "D" , tern_list_d[tern_count,i,:]

                    tern_list_a[tern_count,i,0] = 39.0*ternK_a[k,j,i]
                    tern_list_a[tern_count,i,1] = 23.0*ternMg_a[k,j,i]
                    tern_list_a[tern_count,i,2] = 56.0*ternFe_a[k,j,i]
                    #tern_list_a[tern_count,i,:] = tern_list_a[tern_count,i,:]/np.sum(tern_list_a[tern_count,i,:])
                    if np.max(tern_list_a[tern_count,i,:]) > 0.0:
                        tern_list_a[tern_count,i,:] = tern_list_a[tern_count,i,:]/(1.0*tern_list_a[tern_count,i,0] + 1.0*tern_list_a[tern_count,i,1] + 1.0*tern_list_a[tern_count,i,2])
                        # if i == 20:
                        #     print "A" , tern_list_a[tern_count,i,:]

                    tern_list_b[tern_count,i,0] = 39.0*ternK_b[k,j,i]
                    tern_list_b[tern_count,i,1] = 23.0*ternMg_b[k,j,i]
                    tern_list_b[tern_count,i,2] = 56.0*ternFe_b[k,j,i]
                    #tern_list_b[tern_count,i,:] = tern_list_b[tern_count,i,:]/np.sum(tern_list_b[tern_count,i,:])
                    if np.max(tern_list_b[tern_count,i,:]) > 0.0:
                        tern_list_b[tern_count,i,:] = tern_list_b[tern_count,i,:]/(1.0*tern_list_b[tern_count,i,0] + 1.0*tern_list_b[tern_count,i,1] + 1.0*tern_list_b[tern_count,i,2])
                        # if i == 20:
                        #     print "B" , tern_list_b[tern_count,i,:]

                    tern_count = tern_count + 1



            # ternary values for explicit phases
            # K, Mg, Fe
            # tern_size = 10
            # tern_saponite_mg = [[0.0, 1.0, 0.0]]
            # tern_fe_celadonite = [[0.5, 0.5, 0.0]]
            # tern_fe_oxide = [[0.0, 0.0, 1.0]]


            # fig, tax = ternary.figure(scale=1.0)
            # tax.boundary()

            # #points = np.array([1.0, 1.0, 1.0], [0.6, 0.5, 0.4] [0.1, 0.2, 0.3])
            # tax.gridlines(multiple=0.2, color="black")
            # tax.plot(tern_saponite_mg, marker='o', markersize=tern_size, markeredgecolor='none', linewidth=0.0, label="sap-mg, sap-na, clin14a, ")
            # tax.plot(tern_fe_celadonite, marker='o', markersize=tern_size, markeredgecolor='none', linewidth=0.0, label="fe-celadonite")
            # tax.plot(tern_fe_oxide, marker='o', markersize=tern_size, markeredgecolor='none', linewidth=0.0, label="fe-oxides, pyrite")
            # tax.plot([[0.4, 0.4, 0.2],[0.1, 0.5, 0.4],[0.8, 0.02, 0.08]], marker='o', linewidth=2.0, label="Curve")
            # tax.set_title("Fe/Mg/K ternary plot")
            # tax.left_axis_label("Fe - Mg")
            # tax.right_axis_label("Mg - K")
            # tax.bottom_axis_label("Fe - K")
            # tax.legend(fontsize=9)
            # tax.clear_matplotlib_ticks()
            # #tax.ticks([0.0, 0.5, 1.0], axis='lbr', linewidth=1, fontsize=7, offset=0.02)
            #

            # tax.get_axes().axis('off')


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

            ax=fig.add_subplot(1, 4, 1)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            tax.plot(tern_list[:,i-1,:], color='r', **tern_model_kwargs)
            tax.set_title("solo")
            tax.legend(fontsize=9, bbox_to_anchor=(1.48, 1.1), ncol=1,labelspacing=0.0,columnspacing=0.0,numpoints=1)
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')




            ax=fig.add_subplot(1, 4, 2)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            tax.plot(tern_list_d[:,i-1,:], color='r', **tern_model_kwargs)
            tax.set_title("dual")
            #tax.legend(fontsize=9, bbox_to_anchor=(1.48, 1.1), ncol=1,labelspacing=0.0,columnspacing=0.0,numpoints=1)
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')



            ax=fig.add_subplot(1, 4, 3)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            tax.plot(tern_list_a[:,i-1,:], color='r', **tern_model_kwargs)
            tax.set_title("ch a")
            #tax.legend(fontsize=9, bbox_to_anchor=(1.48, 1.1), ncol=1,labelspacing=0.0,columnspacing=0.0,numpoints=1)
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')




            ax=fig.add_subplot(1, 4, 4)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            tax.plot(tern_list_b[:,i-1,:], color='r', **tern_model_kwargs)
            tax.set_title("ch b")
            #tax.legend(fontsize=9, bbox_to_anchor=(1.48, 1.1), ncol=1,labelspacing=0.0,columnspacing=0.0,numpoints=1)
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')




            #plt.subplots_adjust(hspace=0.15, bottom=0.05, top=0.9, left=0.1, right=0.95)
            plt.subplots_adjust(bottom=0.07, top=0.90, left=0.03, right=0.975)
            plt.savefig(outpath+"jdf_ternary/jdf_ternary_"+letter+"_"+str(i+restart)+".png")





















            if not os.path.exists(outpath+'jdf_tern_rainbow/'):
                os.makedirs(outpath+'jdf_tern_rainbow/')



            #todo: FIG: jdf_tern_rainbow
            #fig=plt.figure(figsize=(9.0,9.0))
            print "jdf_tern_rainbow"
            fig=plt.figure(figsize=(11.0,2.25))


            norm_tern_rb = np.zeros([len(yCell),len(xCell),minNum+1])
            norm_tern_rb_d = np.zeros([len(yCell),len(xCell),minNum+1])
            norm_tern_rb_a = np.zeros([len(yCell),len(xCell),minNum+1])
            norm_tern_rb_b = np.zeros([len(yCell),len(xCell),minNum+1])


            for j in range(len(any_min)):

                # possibly normalize these to a dsecMat instead of a dsecStep
                 if np.max(dsecStep[:,:,any_min[j]]) > 0.0:
                     norm_tern_rb[:,:,any_min[j]] = dsecStep[:,:,any_min[j]]/np.max(dsecStep[:,:,any_min[j]])

                 if np.max(dsecStep_d[:,:,any_min[j]]) > 0.0:
                     norm_tern_rb_d[:,:,any_min[j]] = dsecStep_d[:,:,any_min[j]]/np.max(dsecStep_d[:,:,any_min[j]])


                 if np.max(dsecStep_a[:,:,any_min[j]]) > 0.0:
                     norm_tern_rb_a[:,:,any_min[j]] = dsecStep_a[:,:,any_min[j]]/np.max(dsecStep_a[:,:,any_min[j]])

                 if np.max(dsecStep_b[:,:,any_min[j]]) > 0.0:
                     norm_tern_rb_b[:,:,any_min[j]] = dsecStep_b[:,:,any_min[j]]/np.max(dsecStep_b[:,:,any_min[j]])



            for k in range(len(yCell)):
                for j in range(len(xCell)):

                    tern_rb_K[k,j,i] = 1.0*norm_tern_rb[k,j,14]
                    tern_rb_Mg[k,j,i] = 0.165*norm_tern_rb[k,j,22] + 5.0*norm_tern_rb[k,j,31] + 3.0*norm_tern_rb[k,j,11] + 3.165*norm_tern_rb[k,j,2] + 3.0*norm_tern_rb[k,j,33]
                    tern_rb_Fe[k,j,i] = 1.0*norm_tern_rb[k,j,14] + 2.0*norm_tern_rb[k,j,22] + 2.0*norm_tern_rb[k,j,17] + 1.0*norm_tern_rb[k,j,5] + 2.0*norm_tern_rb[k,j,15]




                    tern_rb_K_d[k,j,i] = 1.0*norm_tern_rb_d[k,j,14]
                    tern_rb_Mg_d[k,j,i] = 0.165*norm_tern_rb_d[k,j,22] + 5.0*norm_tern_rb_d[k,j,31] + 3.0*norm_tern_rb_d[k,j,11] + 3.165*norm_tern_rb_d[k,j,2] + 3.0*norm_tern_rb_d[k,j,33]
                    tern_rb_Fe_d[k,j,i] = 1.0*norm_tern_rb_d[k,j,14] + 2.0*norm_tern_rb_d[k,j,22] + 2.0*norm_tern_rb_d[k,j,17] + 1.0*norm_tern_rb_d[k,j,5] + 2.0*norm_tern_rb_d[k,j,15]


                    tern_rb_K_a[k,j,i] = 1.0*norm_tern_rb_a[k,j,14]
                    tern_rb_Mg_a[k,j,i] = 0.165*norm_tern_rb_a[k,j,22] + 5.0*norm_tern_rb_a[k,j,31] + 3.0*norm_tern_rb_a[k,j,11] + 3.165*norm_tern_rb_a[k,j,2] + 3.0*norm_tern_rb_a[k,j,33]
                    tern_rb_Fe_a[k,j,i] = 1.0*norm_tern_rb_a[k,j,14] + 2.0*norm_tern_rb_a[k,j,22] + 2.0*norm_tern_rb_a[k,j,17] + 1.0*norm_tern_rb_a[k,j,5] + 2.0*norm_tern_rb_a[k,j,15]



                    tern_rb_K_b[k,j,i] = 1.0*norm_tern_rb_b[k,j,14]
                    tern_rb_Mg_b[k,j,i] = 0.165*norm_tern_rb_b[k,j,22] + 5.0*norm_tern_rb_b[k,j,31] + 3.0*norm_tern_rb_b[k,j,11] + 3.165*norm_tern_rb_b[k,j,2] + 3.0*norm_tern_rb_b[k,j,33]
                    tern_rb_Fe_b[k,j,i] = 1.0*norm_tern_rb_b[k,j,14] + 2.0*norm_tern_rb_b[k,j,22] + 2.0*norm_tern_rb_b[k,j,17] + 1.0*norm_tern_rb_b[k,j,5] + 2.0*norm_tern_rb_b[k,j,15]



            tern_count = 0

            for k in range(len(yCell)):
                for j in range(len(xCell)):
                    tern_rb_list[tern_count,i,0] = 39.0*tern_rb_K[k,j,i]
                    tern_rb_list[tern_count,i,1] = 23.0*tern_rb_Mg[k,j,i]
                    tern_rb_list[tern_count,i,2] = 56.0*tern_rb_Fe[k,j,i]
                    if np.max(tern_rb_list[tern_count,i,:]) > 0.0:
                        tern_rb_list[tern_count,i,:] = tern_rb_list[tern_count,i,:]/(1.0*tern_rb_list[tern_count,i,0] + 1.0*tern_rb_list[tern_count,i,1] + 1.0*tern_rb_list[tern_count,i,2])

                    tern_rb_list_d[tern_count,i,0] = 39.0*tern_rb_K_d[k,j,i]
                    tern_rb_list_d[tern_count,i,1] = 23.0*tern_rb_Mg_d[k,j,i]
                    tern_rb_list_d[tern_count,i,2] = 56.0*tern_rb_Fe_d[k,j,i]
                    if np.max(tern_rb_list_d[tern_count,i,:]) > 0.0:
                        tern_rb_list_d[tern_count,i,:] = tern_rb_list_d[tern_count,i,:]/(1.0*tern_rb_list_d[tern_count,i,0] + 1.0*tern_rb_list_d[tern_count,i,1] + 1.0*tern_rb_list_d[tern_count,i,2])


                    tern_rb_list_a[tern_count,i,0] = 39.0*tern_rb_K_a[k,j,i]
                    tern_rb_list_a[tern_count,i,1] = 23.0*tern_rb_Mg_a[k,j,i]
                    tern_rb_list_a[tern_count,i,2] = 56.0*tern_rb_Fe_a[k,j,i]
                    if np.max(tern_rb_list_a[tern_count,i,:]) > 0.0:
                        tern_rb_list_a[tern_count,i,:] = tern_rb_list_a[tern_count,i,:]/(1.0*tern_rb_list_a[tern_count,i,0] + 1.0*tern_rb_list_a[tern_count,i,1] + 1.0*tern_rb_list_a[tern_count,i,2])


                    tern_rb_list_b[tern_count,i,0] = 39.0*tern_rb_K_b[k,j,i]
                    tern_rb_list_b[tern_count,i,1] = 23.0*tern_rb_Mg_b[k,j,i]
                    tern_rb_list_b[tern_count,i,2] = 56.0*tern_rb_Fe_b[k,j,i]
                    if np.max(tern_rb_list_b[tern_count,i,:]) > 0.0:
                        tern_rb_list_b[tern_count,i,:] = tern_rb_list_b[tern_count,i,:]/(1.0*tern_rb_list_b[tern_count,i,0] + 1.0*tern_rb_list_b[tern_count,i,1] + 1.0*tern_rb_list_b[tern_count,i,2])




                    tern_count = tern_count + 1

            tern_rb_point_K[i] = np.sum(tern_rb_list[:,i,0])
            tern_rb_point_Mg[i] = np.sum(tern_rb_list[:,i,1])
            tern_rb_point_Fe[i] = np.sum(tern_rb_list[:,i,2])

            if np.max(np.sum(tern_rb_point_K[i]+tern_rb_point_Mg[i]+tern_rb_point_Fe[i])) > 0.0:
                tern_rb_point_K[i] = tern_rb_point_K[i]/(1.0*np.sum(tern_rb_list[:,i,0]) + 1.0*np.sum(tern_rb_list[:,i,1]) + 1.0*np.sum(tern_rb_list[:,i,2]))
                tern_rb_point_Mg[i] = tern_rb_point_Mg[i]/(1.0*np.sum(tern_rb_list[:,i,0]) + 1.0*np.sum(tern_rb_list[:,i,1]) + 1.0*np.sum(tern_rb_list[:,i,2]))
                tern_rb_point_Fe[i] = tern_rb_point_Fe[i]/(1.0*np.sum(tern_rb_list[:,i,0]) + 1.0*np.sum(tern_rb_list[:,i,1]) + 1.0*np.sum(tern_rb_list[:,i,2]))


            tern_rb_point_K_d[i] = np.sum(tern_rb_list_d[:,i,0])
            tern_rb_point_Mg_d[i] = np.sum(tern_rb_list_d[:,i,1])
            tern_rb_point_Fe_d[i] = np.sum(tern_rb_list_d[:,i,2])

            if np.max(np.sum(tern_rb_point_K_d[i]+tern_rb_point_Mg_d[i]+tern_rb_point_Fe_d[i])) > 0.0:
                tern_rb_point_K_d[i] = tern_rb_point_K_d[i]/(1.0*np.sum(tern_rb_list_d[:,i,0]) + 1.0*np.sum(tern_rb_list_d[:,i,1]) + 1.0*np.sum(tern_rb_list_d[:,i,2]))
                tern_rb_point_Mg_d[i] = tern_rb_point_Mg_d[i]/(1.0*np.sum(tern_rb_list_d[:,i,0]) + 1.0*np.sum(tern_rb_list_d[:,i,1]) + 1.0*np.sum(tern_rb_list_d[:,i,2]))
                tern_rb_point_Fe_d[i] = tern_rb_point_Fe_d[i]/(1.0*np.sum(tern_rb_list_d[:,i,0]) + 1.0*np.sum(tern_rb_list_d[:,i,1]) + 1.0*np.sum(tern_rb_list_d[:,i,2]))


            tern_rb_point_K_a[i] = np.sum(tern_rb_list_a[:,i,0])
            tern_rb_point_Mg_a[i] = np.sum(tern_rb_list_a[:,i,1])
            tern_rb_point_Fe_a[i] = np.sum(tern_rb_list_a[:,i,2])

            if np.max(np.sum(tern_rb_point_K_a[i]+tern_rb_point_Mg_a[i]+tern_rb_point_Fe_a[i])) > 0.0:
                tern_rb_point_K_a[i] = tern_rb_point_K_a[i]/(1.0*np.sum(tern_rb_list_a[:,i,0]) + 1.0*np.sum(tern_rb_list_a[:,i,1]) + 1.0*np.sum(tern_rb_list_a[:,i,2]))
                tern_rb_point_Mg_a[i] = tern_rb_point_Mg_a[i]/(1.0*np.sum(tern_rb_list_a[:,i,0]) + 1.0*np.sum(tern_rb_list_a[:,i,1]) + 1.0*np.sum(tern_rb_list_a[:,i,2]))
                tern_rb_point_Fe_a[i] = tern_rb_point_Fe_a[i]/(1.0*np.sum(tern_rb_list_a[:,i,0]) + 1.0*np.sum(tern_rb_list_a[:,i,1]) + 1.0*np.sum(tern_rb_list_a[:,i,2]))


            tern_rb_point_K_b[i] = np.sum(tern_rb_list_b[:,i,0])
            tern_rb_point_Mg_b[i] = np.sum(tern_rb_list_b[:,i,1])
            tern_rb_point_Fe_b[i] = np.sum(tern_rb_list_b[:,i,2])

            if np.max(np.sum(tern_rb_point_K_b[i]+tern_rb_point_Mg_b[i]+tern_rb_point_Fe_b[i])) > 0.0:
                tern_rb_point_K_b[i] = tern_rb_point_K_b[i]/(1.0*np.sum(tern_rb_list_b[:,i,0]) + 1.0*np.sum(tern_rb_list_b[:,i,1]) + 1.0*np.sum(tern_rb_list_b[:,i,2]))
                tern_rb_point_Mg_b[i] = tern_rb_point_Mg_b[i]/(1.0*np.sum(tern_rb_list_b[:,i,0]) + 1.0*np.sum(tern_rb_list_b[:,i,1]) + 1.0*np.sum(tern_rb_list_b[:,i,2]))
                tern_rb_point_Fe_b[i] = tern_rb_point_Fe_b[i]/(1.0*np.sum(tern_rb_list_b[:,i,0]) + 1.0*np.sum(tern_rb_list_b[:,i,1]) + 1.0*np.sum(tern_rb_list_b[:,i,2]))




            for j in range(steps):

                tern_rb_point_list[i,0] = tern_rb_point_K[i]
                tern_rb_point_list[i,1] = tern_rb_point_Mg[i]
                tern_rb_point_list[i,2] = tern_rb_point_Fe[i]

                tern_rb_point_list_d[i,0] = tern_rb_point_K_d[i]
                tern_rb_point_list_d[i,1] = tern_rb_point_Mg_d[i]
                tern_rb_point_list_d[i,2] = tern_rb_point_Fe_d[i]

                tern_rb_point_list_a[i,0] = tern_rb_point_K_a[i]
                tern_rb_point_list_a[i,1] = tern_rb_point_Mg_a[i]
                tern_rb_point_list_a[i,2] = tern_rb_point_Fe_a[i]

                tern_rb_point_list_b[i,0] = tern_rb_point_K_b[i]
                tern_rb_point_list_b[i,1] = tern_rb_point_Mg_b[i]
                tern_rb_point_list_b[i,2] = tern_rb_point_Fe_b[i]







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

            ax=fig.add_subplot(1, 4, 1)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            tax.plot(tern_rb_point_list, color='r', **tern_model_kwargs)
            tax.set_title("solo")
            tax.legend(fontsize=9, bbox_to_anchor=(1.48, 1.1), ncol=1,labelspacing=0.0,columnspacing=0.0,numpoints=1)
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')



            ax=fig.add_subplot(1, 4, 2)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            tax.plot(tern_rb_point_list_d, color='r', **tern_model_kwargs)
            tax.set_title("dual")
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')




            ax=fig.add_subplot(1, 4, 3)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            tax.plot(tern_rb_point_list_a, color='r', **tern_model_kwargs)
            tax.set_title("a")
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')



            ax=fig.add_subplot(1, 4, 4)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            tax.plot(tern_rb_point_list_b, color='r', **tern_model_kwargs)
            tax.set_title("b")
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')




            #plt.subplots_adjust(hspace=0.15, bottom=0.05, top=0.9, left=0.1, right=0.95)
            plt.subplots_adjust(bottom=0.07, top=0.90, left=0.03, right=0.975)
            plt.savefig(outpath+"jdf_tern_rainbow/jdf_tern_rainbow_"+letter+"_"+str(i+restart)+".png")





















            if not os.path.exists(outpath+'jdf_tern_rainbow_X/'):
                os.makedirs(outpath+'jdf_tern_rainbow_X/')



            #todo: FIG: jdf_tern_rainbow_X
            #fig=plt.figure(figsize=(9.0,9.0))
            print "jdf_tern_rainbow_X"
            fig=plt.figure(figsize=(11.0,2.25))


            norm_tern_rb = np.zeros([len(yCell),len(xCell),minNum+1])
            norm_tern_rb_d = np.zeros([len(yCell),len(xCell),minNum+1])
            norm_tern_rb_a = np.zeros([len(yCell),len(xCell),minNum+1])
            norm_tern_rb_b = np.zeros([len(yCell),len(xCell),minNum+1])


            for j in range(len(any_min)):

                # possibly normalize these to a dsecMat instead of a dsecStep
                #  if np.max(dsecStep[:,xd_move,any_min[j]]) > 0.0:
                #      norm_tern_rb[:,xd_move,any_min[j]] = dsecStep[:,xd_move,any_min[j]]/np.max(dsecStep[:,xd_move,any_min[j]])
                 #
                #  if np.max(dsecStep_d[:,xd_move,any_min[j]]) > 0.0:
                #      norm_tern_rb_d[:,xd_move,any_min[j]] = dsecStep_d[:,xd_move,any_min[j]]/np.max(dsecStep_d[:,xd_move,any_min[j]])
                 #
                 #
                #  if np.max(dsecStep_a[:,xd_move,any_min[j]]) > 0.0:
                #      norm_tern_rb_a[:,xd_move,any_min[j]] = dsecStep_a[:,xd_move,any_min[j]]/np.max(dsecStep_a[:,xd_move,any_min[j]])
                 #
                #  if np.max(dsecStep_b[:,xd_move,any_min[j]]) > 0.0:
                #      norm_tern_rb_b[:,xd_move,any_min[j]] = dsecStep_b[:,xd_move,any_min[j]]/np.max(dsecStep_b[:,xd_move,any_min[j]])




                if np.max(dsecStep[:,xd_move,any_min[j]]) > 0.0:
                    norm_tern_rb[:,xd_move,any_min[j]] = dsecStep[:,xd_move,any_min[j]]*density[any_min[j]]/molar[any_min[j]]#/np.max(dsecStep[:,xd_move,any_min[j]])

                if np.max(dsecStep_d[:,xd_move,any_min[j]]) > 0.0:
                    norm_tern_rb_d[:,xd_move,any_min[j]] = dsecStep_d[:,xd_move,any_min[j]]*density[any_min[j]]/molar[any_min[j]]#/np.max(dsecStep_d[:,xd_move,any_min[j]])


                if np.max(dsecStep_a[:,xd_move,any_min[j]]) > 0.0:
                    norm_tern_rb_a[:,xd_move,any_min[j]] = dsecStep_a[:,xd_move,any_min[j]]*density[any_min[j]]/molar[any_min[j]]#/np.max(dsecStep_a[:,xd_move,any_min[j]])

                if np.max(dsecStep_b[:,xd_move,any_min[j]]) > 0.0:
                    norm_tern_rb_b[:,xd_move,any_min[j]] = dsecStep_b[:,xd_move,any_min[j]]*density[any_min[j]]/molar[any_min[j]]#/np.max(dsecStep_b[:,xd_move,any_min[j]])




                #  if np.max(dsecStep[:,xd_move,any_min[j]]) > 0.0:
                #      norm_tern_rb[:,xd_move,any_min[j]] = dsecStep[:,xd_move,any_min[j]]/np.max(dsecMat[:,:,any_min[j]])
                 #
                #  if np.max(dsecStep_d[:,xd_move,any_min[j]]) > 0.0:
                #      norm_tern_rb_d[:,xd_move,any_min[j]] = dsecStep_d[:,xd_move,any_min[j]]/np.max(dsecMat_d[:,:,any_min[j]])
                 #
                 #
                #  if np.max(dsecStep_a[:,:,any_min[j]]) > 0.0:
                #      norm_tern_rb_a[:,xd_move,any_min[j]] = dsecStep_a[:,xd_move,any_min[j]]/np.max(dsecMat_a[:,:,any_min[j]])
                 #
                #  if np.max(dsecStep_b[:,xd_move,any_min[j]]) > 0.0:
                #      norm_tern_rb_b[:,xd_move,any_min[j]] = dsecStep_b[:,xd_move,any_min[j]]/np.max(dsecMat_b[:,:,any_min[j]])



            for k in range(len(yCell)):
                for j in [xd_move]:

                    tern_x_rb_K[k,j,i] = 1.0*norm_tern_rb[k,j,14]
                    tern_x_rb_Mg[k,j,i] = 0.165*norm_tern_rb[k,j,22] + 5.0*norm_tern_rb[k,j,31] + 3.0*norm_tern_rb[k,j,11] + 3.165*norm_tern_rb[k,j,2] + 3.0*norm_tern_rb[k,j,33]
                    tern_x_rb_Fe[k,j,i] = 1.0*norm_tern_rb[k,j,14] + 2.0*norm_tern_rb[k,j,22] + 2.0*norm_tern_rb[k,j,17] + 1.0*norm_tern_rb[k,j,5] + 2.0*norm_tern_rb[k,j,15]

                    tern_x_rb_K_d[k,j,i] = 1.0*norm_tern_rb_d[k,j,14]
                    tern_x_rb_Mg_d[k,j,i] = 0.165*norm_tern_rb_d[k,j,22] + 5.0*norm_tern_rb_d[k,j,31] + 3.0*norm_tern_rb_d[k,j,11] + 3.165*norm_tern_rb_d[k,j,2] + 3.0*norm_tern_rb_d[k,j,33]
                    tern_x_rb_Fe_d[k,j,i] = 1.0*norm_tern_rb_d[k,j,14] + 2.0*norm_tern_rb_d[k,j,22] + 2.0*norm_tern_rb_d[k,j,17] + 1.0*norm_tern_rb_d[k,j,5] + 2.0*norm_tern_rb_d[k,j,15]

                    tern_x_rb_K_a[k,j,i] = 1.0*norm_tern_rb_a[k,j,14]
                    tern_x_rb_Mg_a[k,j,i] = 0.165*norm_tern_rb_a[k,j,22] + 5.0*norm_tern_rb_a[k,j,31] + 3.0*norm_tern_rb_a[k,j,11] + 3.165*norm_tern_rb_a[k,j,2] + 3.0*norm_tern_rb_a[k,j,33]
                    tern_x_rb_Fe_a[k,j,i] = 1.0*norm_tern_rb_a[k,j,14] + 2.0*norm_tern_rb_a[k,j,22] + 2.0*norm_tern_rb_a[k,j,17] + 1.0*norm_tern_rb_a[k,j,5] + 2.0*norm_tern_rb_a[k,j,15]

                    tern_x_rb_K_b[k,j,i] = 1.0*norm_tern_rb_b[k,j,14]
                    tern_x_rb_Mg_b[k,j,i] = 0.165*norm_tern_rb_b[k,j,22] + 5.0*norm_tern_rb_b[k,j,31] + 3.0*norm_tern_rb_b[k,j,11] + 3.165*norm_tern_rb_b[k,j,2] + 3.0*norm_tern_rb_b[k,j,33]
                    tern_x_rb_Fe_b[k,j,i] = 1.0*norm_tern_rb_b[k,j,14] + 2.0*norm_tern_rb_b[k,j,22] + 2.0*norm_tern_rb_b[k,j,17] + 1.0*norm_tern_rb_b[k,j,5] + 2.0*norm_tern_rb_b[k,j,15]

            # print tern_x_rb_Mg_a[:,xd_move,i]


            tern_count = 0

            for k in range(len(yCell)):
                for j in [xd_move]:
                    tern_x_rb_list[tern_count,i,0] = 39.0*tern_x_rb_K[k,j,i]
                    tern_x_rb_list[tern_count,i,1] = 23.0*tern_x_rb_Mg[k,j,i]
                    tern_x_rb_list[tern_count,i,2] = 56.0*tern_x_rb_Fe[k,j,i]
                    if np.max(tern_x_rb_list[tern_count,i,:]) > 0.0:
                        tern_x_rb_list[tern_count,i,:] = tern_x_rb_list[tern_count,i,:]/(1.0*tern_x_rb_list[tern_count,i,0] + 1.0*tern_x_rb_list[tern_count,i,1] + 1.0*tern_x_rb_list[tern_count,i,2])

                    tern_x_rb_list_d[tern_count,i,0] = 39.0*tern_x_rb_K_d[k,j,i]
                    tern_x_rb_list_d[tern_count,i,1] = 23.0*tern_x_rb_Mg_d[k,j,i]
                    tern_x_rb_list_d[tern_count,i,2] = 56.0*tern_x_rb_Fe_d[k,j,i]
                    if np.max(tern_x_rb_list_d[tern_count,i,:]) > 0.0:
                        tern_x_rb_list_d[tern_count,i,:] = tern_x_rb_list_d[tern_count,i,:]/(1.0*tern_x_rb_list_d[tern_count,i,0] + 1.0*tern_x_rb_list_d[tern_count,i,1] + 1.0*tern_x_rb_list_d[tern_count,i,2])


                    tern_x_rb_list_a[tern_count,i,0] = 39.0*tern_x_rb_K_a[k,j,i]
                    tern_x_rb_list_a[tern_count,i,1] = 23.0*tern_x_rb_Mg_a[k,j,i]
                    tern_x_rb_list_a[tern_count,i,2] = 56.0*tern_x_rb_Fe_a[k,j,i]
                    if np.max(tern_x_rb_list_a[tern_count,i,:]) > 0.0:
                        tern_x_rb_list_a[tern_count,i,:] = tern_x_rb_list_a[tern_count,i,:]/(1.0*tern_x_rb_list_a[tern_count,i,0] + 1.0*tern_x_rb_list_a[tern_count,i,1] + 1.0*tern_x_rb_list_a[tern_count,i,2])


                    tern_x_rb_list_b[tern_count,i,0] = 39.0*tern_x_rb_K_b[k,j,i]
                    tern_x_rb_list_b[tern_count,i,1] = 23.0*tern_x_rb_Mg_b[k,j,i]
                    tern_x_rb_list_b[tern_count,i,2] = 56.0*tern_x_rb_Fe_b[k,j,i]
                    if np.max(tern_x_rb_list_b[tern_count,i,:]) > 0.0:
                        tern_x_rb_list_b[tern_count,i,:] = tern_x_rb_list_b[tern_count,i,:]/(1.0*tern_x_rb_list_b[tern_count,i,0] + 1.0*tern_x_rb_list_b[tern_count,i,1] + 1.0*tern_x_rb_list_b[tern_count,i,2])


                    # tern_x_rb_list[tern_count,i,0] = 39.0*tern_x_rb_K[k,j,i]
                    # tern_x_rb_list[tern_count,i,1] = 39.0*tern_x_rb_Mg[k,j,i]
                    # tern_x_rb_list[tern_count,i,2] = 39.0*tern_x_rb_Fe[k,j,i]
                    # if np.max(tern_x_rb_list[tern_count,i,:]) > 0.0:
                    #     tern_x_rb_list[tern_count,i,:] = tern_x_rb_list[tern_count,i,:]/(1.0*tern_x_rb_list[tern_count,i,0] + 1.0*tern_x_rb_list[tern_count,i,1] + 1.0*tern_x_rb_list[tern_count,i,2])
                    #
                    # tern_x_rb_list_d[tern_count,i,0] = 39.0*tern_x_rb_K_d[k,j,i]
                    # tern_x_rb_list_d[tern_count,i,1] = 39.0*tern_x_rb_Mg_d[k,j,i]
                    # tern_x_rb_list_d[tern_count,i,2] = 39.0*tern_x_rb_Fe_d[k,j,i]
                    # if np.max(tern_x_rb_list_d[tern_count,i,:]) > 0.0:
                    #     tern_x_rb_list_d[tern_count,i,:] = tern_x_rb_list_d[tern_count,i,:]/(1.0*tern_x_rb_list_d[tern_count,i,0] + 1.0*tern_x_rb_list_d[tern_count,i,1] + 1.0*tern_x_rb_list_d[tern_count,i,2])
                    #
                    #
                    # tern_x_rb_list_a[tern_count,i,0] = 39.0*tern_x_rb_K_a[k,j,i]
                    # tern_x_rb_list_a[tern_count,i,1] = 39.0*tern_x_rb_Mg_a[k,j,i]
                    # tern_x_rb_list_a[tern_count,i,2] = 39.0*tern_x_rb_Fe_a[k,j,i]
                    # if np.max(tern_x_rb_list_a[tern_count,i,:]) > 0.0:
                    #     tern_x_rb_list_a[tern_count,i,:] = tern_x_rb_list_a[tern_count,i,:]/(1.0*tern_x_rb_list_a[tern_count,i,0] + 1.0*tern_x_rb_list_a[tern_count,i,1] + 1.0*tern_x_rb_list_a[tern_count,i,2])
                    #
                    #
                    # tern_x_rb_list_b[tern_count,i,0] = 39.0*tern_x_rb_K_b[k,j,i]
                    # tern_x_rb_list_b[tern_count,i,1] = 39.0*tern_x_rb_Mg_b[k,j,i]
                    # tern_x_rb_list_b[tern_count,i,2] = 39.0*tern_x_rb_Fe_b[k,j,i]
                    # if np.max(tern_x_rb_list_b[tern_count,i,:]) > 0.0:
                    #     tern_x_rb_list_b[tern_count,i,:] = tern_x_rb_list_b[tern_count,i,:]/(1.0*tern_x_rb_list_b[tern_count,i,0] + 1.0*tern_x_rb_list_b[tern_count,i,1] + 1.0*tern_x_rb_list_b[tern_count,i,2])




                    tern_count = tern_count + 1

            tern_x_rb_point_K[i] = np.sum(tern_x_rb_list[:,i,0])
            tern_x_rb_point_Mg[i] = np.sum(tern_x_rb_list[:,i,1])
            tern_x_rb_point_Fe[i] = np.sum(tern_x_rb_list[:,i,2])

            if np.max(np.sum(tern_x_rb_point_K[i]+tern_x_rb_point_Mg[i]+tern_x_rb_point_Fe[i])) > 0.0:
                tern_x_rb_point_K[i] = tern_x_rb_point_K[i]/(1.0*np.sum(tern_x_rb_list[:,i,0]) + 1.0*np.sum(tern_x_rb_list[:,i,1]) + 1.0*np.sum(tern_x_rb_list[:,i,2]))
                tern_x_rb_point_Mg[i] = tern_x_rb_point_Mg[i]/(1.0*np.sum(tern_x_rb_list[:,i,0]) + 1.0*np.sum(tern_x_rb_list[:,i,1]) + 1.0*np.sum(tern_x_rb_list[:,i,2]))
                tern_x_rb_point_Fe[i] = tern_x_rb_point_Fe[i]/(1.0*np.sum(tern_x_rb_list[:,i,0]) + 1.0*np.sum(tern_x_rb_list[:,i,1]) + 1.0*np.sum(tern_x_rb_list[:,i,2]))

            # print " "
            # print "point K" , tern_x_rb_point_K[i]
            # print "point Mg" , tern_x_rb_point_Mg[i]
            # print "point Fe" , tern_x_rb_point_Fe[i]



            tern_x_rb_point_K_d[i] = np.sum(tern_x_rb_list_d[:,i,0])
            tern_x_rb_point_Mg_d[i] = np.sum(tern_x_rb_list_d[:,i,1])
            tern_x_rb_point_Fe_d[i] = np.sum(tern_x_rb_list_d[:,i,2])

            if np.max(np.sum(tern_x_rb_point_K_d[i]+tern_x_rb_point_Mg_d[i]+tern_x_rb_point_Fe_d[i])) > 0.0:
                tern_x_rb_point_K_d[i] = tern_x_rb_point_K_d[i]/(1.0*np.sum(tern_x_rb_list_d[:,i,0]) + 1.0*np.sum(tern_x_rb_list_d[:,i,1]) + 1.0*np.sum(tern_x_rb_list_d[:,i,2]))
                tern_x_rb_point_Mg_d[i] = tern_x_rb_point_Mg_d[i]/(1.0*np.sum(tern_x_rb_list_d[:,i,0]) + 1.0*np.sum(tern_x_rb_list_d[:,i,1]) + 1.0*np.sum(tern_x_rb_list_d[:,i,2]))
                tern_x_rb_point_Fe_d[i] = tern_x_rb_point_Fe_d[i]/(1.0*np.sum(tern_x_rb_list_d[:,i,0]) + 1.0*np.sum(tern_x_rb_list_d[:,i,1]) + 1.0*np.sum(tern_x_rb_list_d[:,i,2]))

            # print " "
            # print "point K_d" , tern_x_rb_point_K_d[i]
            # print "point Mg_d" , tern_x_rb_point_Mg_d[i]
            # print "point Fe_d" , tern_x_rb_point_Fe_d[i]




            tern_x_rb_point_K_a[i] = np.sum(tern_x_rb_list_a[:,i,0])
            tern_x_rb_point_Mg_a[i] = np.sum(tern_x_rb_list_a[:,i,1])
            tern_x_rb_point_Fe_a[i] = np.sum(tern_x_rb_list_a[:,i,2])

            if np.max(np.sum(tern_x_rb_point_K_a[i]+tern_x_rb_point_Mg_a[i]+tern_x_rb_point_Fe_a[i])) > 0.0:
                tern_x_rb_point_K_a[i] = tern_x_rb_point_K_a[i]/(1.0*np.sum(tern_x_rb_list_a[:,i,0]) + 1.0*np.sum(tern_x_rb_list_a[:,i,1]) + 1.0*np.sum(tern_x_rb_list_a[:,i,2]))
                tern_x_rb_point_Mg_a[i] = tern_x_rb_point_Mg_a[i]/(1.0*np.sum(tern_x_rb_list_a[:,i,0]) + 1.0*np.sum(tern_x_rb_list_a[:,i,1]) + 1.0*np.sum(tern_x_rb_list_a[:,i,2]))
                tern_x_rb_point_Fe_a[i] = tern_x_rb_point_Fe_a[i]/(1.0*np.sum(tern_x_rb_list_a[:,i,0]) + 1.0*np.sum(tern_x_rb_list_a[:,i,1]) + 1.0*np.sum(tern_x_rb_list_a[:,i,2]))

            # print " "
            # print "point K_a" , tern_x_rb_point_K_a[i]
            # print "point Mg_a" , tern_x_rb_point_Mg_a[i]
            # print "point Fe_a" , tern_x_rb_point_Fe_a[i]




            tern_x_rb_point_K_b[i] = np.sum(tern_x_rb_list_b[:,i,0])
            tern_x_rb_point_Mg_b[i] = np.sum(tern_x_rb_list_b[:,i,1])
            tern_x_rb_point_Fe_b[i] = np.sum(tern_x_rb_list_b[:,i,2])


            if np.max(np.sum(tern_x_rb_point_K_b[i]+tern_x_rb_point_Mg_b[i]+tern_x_rb_point_Fe_b[i])) > 0.0:
                tern_x_rb_point_K_b[i] = tern_x_rb_point_K_b[i]/(1.0*np.sum(tern_x_rb_list_b[:,i,0]) + 1.0*np.sum(tern_x_rb_list_b[:,i,1]) + 1.0*np.sum(tern_x_rb_list_b[:,i,2]))
                tern_x_rb_point_Mg_b[i] = tern_x_rb_point_Mg_b[i]/(1.0*np.sum(tern_x_rb_list_b[:,i,0]) + 1.0*np.sum(tern_x_rb_list_b[:,i,1]) + 1.0*np.sum(tern_x_rb_list_b[:,i,2]))
                tern_x_rb_point_Fe_b[i] = tern_x_rb_point_Fe_b[i]/(1.0*np.sum(tern_x_rb_list_b[:,i,0]) + 1.0*np.sum(tern_x_rb_list_b[:,i,1]) + 1.0*np.sum(tern_x_rb_list_b[:,i,2]))

            # print " "
            # print "point K_b" , tern_x_rb_point_K_b[i]
            # print "point Mg_b" , tern_x_rb_point_Mg_b[i]
            # print "point Fe_b" , tern_x_rb_point_Fe_b[i]
            #
            # print " "


            for j in range(steps):

                tern_x_rb_point_list[i,0] = tern_x_rb_point_K[i]
                tern_x_rb_point_list[i,1] = tern_x_rb_point_Mg[i]
                tern_x_rb_point_list[i,2] = tern_x_rb_point_Fe[i]

                tern_x_rb_point_list_d[i,0] = tern_x_rb_point_K_d[i]
                tern_x_rb_point_list_d[i,1] = tern_x_rb_point_Mg_d[i]
                tern_x_rb_point_list_d[i,2] = tern_x_rb_point_Fe_d[i]

                tern_x_rb_point_list_a[i,0] = tern_x_rb_point_K_a[i]
                tern_x_rb_point_list_a[i,1] = tern_x_rb_point_Mg_a[i]
                tern_x_rb_point_list_a[i,2] = tern_x_rb_point_Fe_a[i]

                tern_x_rb_point_list_b[i,0] = tern_x_rb_point_K_b[i]
                tern_x_rb_point_list_b[i,1] = tern_x_rb_point_Mg_b[i]
                tern_x_rb_point_list_b[i,2] = tern_x_rb_point_Fe_b[i]







            tern_size = 8
            tern_size_small = 25
            tern_saponite_mg = [[0.0, 1.0, 0.0]]
            tern_fe_celadonite = [[0.5, 0.0, 0.5]]
            tern_fe_oxide = [[0.0, 0.0, 1.0]]
            tern_min_col = '#524aaf'


            # tern_min_kwargs = {'marker': 's', 'color': tern_min_col}
            tern_min_kwargs = dict(color=tern_min_col, marker='s', markersize=tern_size, markeredgecolor='k', linewidth=0.0, zorder=10)
            tern_model_kwargs = dict(marker='.', markersize=tern_size_small, markeredgecolor='none', linewidth=0.0)
            tern_model_kwargs_big = dict(marker='.', markersize=tern_size_small*2.0, markeredgecolor='none', linewidth=0.0)

            tern_model_kwargs_scatter = dict(s=tern_size_small, edgecolor='none', zorder=10)
            colors=steps

            # colors = cm.rainbow(np.linspace(0, 1, len(Ys)))
            # cs = [colors[i//len(X)] for i in range(len(Ys)*len(X))]

            cmm = plt.get_cmap('jet')



            ax=fig.add_subplot(1, 4, 1)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            if i > 5:
                for iii in range(1,i):
                    colorz = cmm(1.*iii/steps)
                    tax.scatter(tern_x_rb_point_list[iii-1:iii,:], c=colorz, **tern_model_kwargs_scatter)
            tax.set_title("solo")
            tax.legend(fontsize=9, bbox_to_anchor=(1.48, 1.1), ncol=1,labelspacing=0.0,columnspacing=0.0,numpoints=1)
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')



            ax=fig.add_subplot(1, 4, 2)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            if i > 5:
                for iii in range(1,i):
                    colorz = cmm(1.*iii/steps)
                    tax.scatter(tern_x_rb_point_list_d[iii-1:iii,:], c=colorz, **tern_model_kwargs_scatter)
            tax.set_title("dual")
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')




            ax=fig.add_subplot(1, 4, 3)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            if i > 5:
                for iii in range(1,i):
                    colorz = cmm(1.*iii/steps)
                    tax.scatter(tern_x_rb_point_list_a[iii-1:iii,:], c=colorz, **tern_model_kwargs_scatter)
            tax.set_title("a")
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')



            ax=fig.add_subplot(1, 4, 4)
            fig, tax = ternary.figure(ax=ax,scale=1.0)
            tax.boundary()

            tax.gridlines(multiple=0.2, color="black")
            tax.plot(tern_saponite_mg, label='model phases', **tern_min_kwargs)
            tax.plot(tern_fe_celadonite, **tern_min_kwargs)
            tax.plot(tern_fe_oxide, **tern_min_kwargs)
            if i > 5:
                for iii in range(1,i):
                    colorz = cmm(1.*iii/steps)
                    tax.scatter(tern_x_rb_point_list_b[iii-1:iii,:], c=colorz, **tern_model_kwargs_scatter)
            tax.set_title("b")
            tax.clear_matplotlib_ticks()

            tax.get_axes().axis('off')




            #plt.subplots_adjust(hspace=0.15, bottom=0.05, top=0.9, left=0.1, right=0.95)
            plt.subplots_adjust(bottom=0.07, top=0.90, left=0.03, right=0.975)
            plt.savefig(outpath+"jdf_tern_rainbow_X/jdf_tern_rainbow_X_"+letter+"_"+str(i+restart)+".png")

            if i == 50 or i == 49:
                plt.savefig(outpath+"jdf_tern_rainbow_X/jdf_tern_rainbow_X_"+letter+"_"+str(i+restart+1)+".eps")






            if not os.path.exists(outpath+'jdf_alt/'):
                os.makedirs(outpath+'jdf_alt/')

            #todo: FIGURE: jdf_alt_plot, NXF
            print "jdf_alt_plot plot"

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

            alt_col_mean_d = np.zeros(len(xCell))
            alt_col_mean_s = np.zeros(len(xCell))

            alt_col_mean_top_half_d = np.zeros(len(xCell))
            alt_col_mean_top_half_s = np.zeros(len(xCell))

            alt_col_mean_top_cell_d = np.zeros(len(xCell))
            alt_col_mean_top_cell_s = np.zeros(len(xCell))

            for j in range(len(xCell)):
                # full column average
                above_zero = alt_vol[:,j]*100.0
                above_zero = above_zero[above_zero>0.0]
                alt_col_mean_s[j] = np.mean(above_zero)

                above_zero = alt_vol_d[:,j]*100.0
                above_zero = above_zero[above_zero>0.0]
                alt_col_mean_d[j] = np.mean(above_zero)


                # top half of column average
                above_zero = alt_vol[:,j]*100.0
                above_zero = above_zero[above_zero>0.0]
                alt_col_mean_top_half_s[j] = np.mean(above_zero[len(above_zero)/2:])

                above_zero = alt_vol_d[:,j]*100.0
                above_zero = above_zero[above_zero>0.0]
                alt_col_mean_top_half_d[j] = np.mean(above_zero[len(above_zero)/2:])


                # top cell of column
                above_zero = alt_vol[:,j]*100.0
                above_zero = above_zero[above_zero>0.0]
                alt_col_mean_top_cell_s[j] = np.mean(above_zero[-1:])

                above_zero = alt_vol_d[:,j]*100.0
                above_zero = above_zero[above_zero>0.0]
                alt_col_mean_top_cell_d[j] = np.mean(above_zero[-1:])



            fig=plt.figure(figsize=(10.0,10.0))

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
            plt.plot(xCell,alt_col_mean_s,color=plot_purple,lw=2, label="column mean solo")
            plt.plot(xCell,alt_col_mean_d,color=plot_blue,lw=2, label="column mean dual")

            # plot model top half mean
            plt.plot(xCell,alt_col_mean_top_half_s,color=plot_purple,lw=2,linestyle='--')
            plt.plot(xCell,alt_col_mean_top_half_d,color=plot_blue,lw=2,linestyle='--')

            # plot model top cell
            plt.plot(xCell,alt_col_mean_top_cell_s,color=plot_purple,lw=2,linestyle=':')
            plt.plot(xCell,alt_col_mean_top_cell_d,color=plot_blue,lw=2,linestyle=':')



            plt.legend(fontsize=10)
            plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=12)
            plt.xlim([0.0, 90000.0])
            plt.xlabel('Distance along transect [km]')

            plt.yticks([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],fontsize=12)
            plt.ylim([0.0, 30.0])
            plt.ylabel('Alteration volume $\%$')




            #todo: FIGURE: FeO / FeOt plot, NXF
            # FeO / FeOt data
            fe_values = np.array([0.7753, 0.7442, 0.7519, 0.7610, 0.6714, 0.7416, 0.7039, 0.6708, 0.6403])
            lower_eb_fe = np.array([0.7753, 0.7442, 0.7208, 0.7409, 0.6240, 0.7260, 0.6584, 0.6299, 0.6084])
            upper_eb_fe = np.array([0.7753, 0.7442, 0.7519, 0.7812, 0.7110, 0.7610, 0.7396, 0.7104, 0.7026])

            # calculate model FeO/FeOt column mean
            fe_col_mean_d = np.zeros(len(xCell))
            fe_col_mean_s = np.zeros(len(xCell))

            fe_col_mean_d_pri = np.zeros(len(xCell))
            fe_col_mean_s_pri = np.zeros(len(xCell))


            fe_col_mean_top_half_d = np.zeros(len(xCell))
            fe_col_mean_top_half_s = np.zeros(len(xCell))

            fe_col_mean_top_cell_d = np.zeros(len(xCell))
            fe_col_mean_top_cell_s = np.zeros(len(xCell))



            feo_col_mean_temp = np.zeros(len(xCell))
            feot_col_mean_temp = np.zeros(len(xCell))
            for j in range(len(xCell)):

                # solo, purple

                secStep_temp = secStep
                alt_vol_temp = pri_total
                glass_temp = glass
                ol_temp = ol

                above_zero = alt_vol_temp[:,j]*100.0
                above_zero = above_zero[above_zero>0.0]
                above_zero_ind = np.nonzero(alt_vol_temp[:,j])


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



                fe_col_mean_s[j] = feo_col_mean_temp[j] / (feo_col_mean_temp[j] + feot_col_mean_temp[j])


                fe_col_mean_s_pri[j] = 1.0*np.mean(ol_temp[above_zero_ind,j])*(density_pri[2]/molar_pri[2]) / ((1.0*np.mean(ol_temp[above_zero_ind,j])*(density_pri[2]/molar_pri[2])) + (0.149*2.0*0.8998*np.mean(glass_temp[above_zero_ind,j])*(density_pri[3]/molar_pri[3])))


            # print "fe_col_mean_s" , fe_col_mean_s


            feo_col_mean_temp = np.zeros(len(xCell))
            feot_col_mean_temp = np.zeros(len(xCell))
            for j in range(len(xCell)):
                # dual, blue

                secStep_temp = secStep_d
                alt_vol_temp = pri_total_d
                glass_temp = glass_d
                ol_temp = ol_d

                above_zero = alt_vol_temp[:,j]*100.0
                above_zero = above_zero[above_zero>0.0]
                above_zero_ind = np.nonzero(alt_vol_temp[:,j])


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

                fe_col_mean_d[j] = feo_col_mean_temp[j] / (feo_col_mean_temp[j] + feot_col_mean_temp[j])

                fe_col_mean_d_pri[j] = 1.0*np.mean(ol_temp[above_zero_ind,j])*(density_pri[2]/molar_pri[2]) / ((1.0*np.mean(ol_temp[above_zero_ind,j])*(density_pri[2]/molar_pri[2])) + (0.149*2.0*0.8998*np.mean(glass_temp[above_zero_ind,j])*(density_pri[3]/molar_pri[3])))

            # print "fe_col_mean_d" , fe_col_mean_d



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
            plt.plot(xCell,fe_col_mean_s,color=plot_purple,lw=2, label="column mean solo")
            plt.plot(xCell,fe_col_mean_d,color=plot_blue,lw=2, label="column mean dual")


            # plt.plot(xCell,fe_col_mean_s_pri,color=plot_purple,lw=1, label="column mean solo pri")
            # plt.plot(xCell,fe_col_mean_d_pri,color=plot_blue,lw=1, label="column mean dual pri")

            # # plot model top half mean
            # plt.plot(xCell,alt_col_mean_top_half_s,color=plot_purple,lw=2,linestyle='--')
            # plt.plot(xCell,alt_col_mean_top_half_d,color=plot_blue,lw=2,linestyle='--')
            #
            # # plot model top cell
            # plt.plot(xCell,alt_col_mean_top_cell_s,color=plot_purple,lw=2,linestyle=':')
            # plt.plot(xCell,alt_col_mean_top_cell_d,color=plot_blue,lw=2,linestyle=':')






            #plt.legend(fontsize=10)
            plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=12)
            plt.xlim([0.0, 90000.0])
            plt.xlabel('Distance along transect [km]')

            plt.yticks([0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80])
            #plt.ylim([0.6, 0.8])
            plt.ylim([0.6, 0.8])
            plt.ylabel('FeO / FeOt')


            #plt.subplots_adjust( wspace=0.05 , bottom=0.2, top=0.95, left=0.03, right=0.975)
            plt.savefig(outpath+"jdf_alt/jdf_"+letter+"_alt_"+str(i+restart)+".png")




            # #tod0: FIG: speed
            # print "jdf_Speed"
            # fig=plt.figure(figsize=(10.0,14.0))
            #
            #
            # if i == 0:
            #     vvmin = np.min([speed3_0[speed3_0>0.0], speed3_0_a[speed3_0_a>0.0], speed3_0_b[speed3_0_b>0.0]])
            #     vvmax = np.max([speed3_0, speed3_0_a, speed3_0_b])
            # print "vvmax" , vvmax , "vvmin" , vvmin
            #
            # ax=fig.add_subplot(5, 3, 1, frameon=True, aspect=asp*4.0)
            #
            #
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = speed3
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=vvmin, vmax=vvmax)
            # plt.ylim([-510.0,-320.0])
            # plt.xlim([0,90000])
            # plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=8)
            #
            #
            #
            #
            # ax=fig.add_subplot(5, 3, 2, frameon=True, aspect=asp*4.0)
            #
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = speed3_a
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=vvmin, vmax=vvmax)
            # plt.ylim([-510.0,-320.0])
            # plt.xlim([0,90000])
            # plt.yticks([])
            # plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=8)
            #
            #
            #
            #
            # ax=fig.add_subplot(5, 3, 3, frameon=True, aspect=asp*4.0)
            #
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = speed3_b
            #
            #
            # pSpeed = plt.pcolor(xCell_t,yCell,varStep_t,vmin=vvmin, vmax=vvmax)
            # plt.ylim([-510.0,-320.0])
            # plt.xlim([0,90000])
            # plt.yticks([])
            # plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=8)
            #
            # bbox = ax.get_position()
            # cax = fig.add_axes([bbox.xmin+bbox.width-0.6, bbox.ymin+0.08, bbox.width*2.0, bbox.height*0.08])
            # cbar = plt.colorbar(pSpeed, cax = cax,orientation='horizontal')
            # #plt.title("ms",fontsize=10)
            # cbar.solids.set_rasterized(True)
            # cbar.solids.set_edgecolor("face")
            #
            #
            #
            #
            #
            #
            #
            #
            # ax=fig.add_subplot(5, 3, 4, frameon=True, aspect=asp*4.0)
            #
            # varSpeed = speed3#np.zeros(speed3.shape)
            # varSpeed[speed1!=3.0] = 0.0
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = varSpeed
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=vvmin, vmax=vvmax)
            # plt.ylim([-510.0,-320.0])
            # plt.xlim([0,90000])
            # plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=8)
            #
            #
            #
            #
            # ax=fig.add_subplot(5, 3, 5, frameon=True, aspect=asp*4.0)
            #
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = speed3_a
            #
            # plt.pcolor(xCell_t,yCell,varStep_t,vmin=vvmin, vmax=vvmax)
            # plt.ylim([-510.0,-320.0])
            # plt.xlim([0,90000])
            # plt.yticks([])
            # plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=8)
            #
            #
            #
            #
            # ax=fig.add_subplot(5, 3, 6, frameon=True, aspect=asp*4.0)
            #
            # xCell_t = []
            # xCell_t = np.append(xCell, [xCell[-1]+xCell[-1]-xCell[-2]],axis=0)
            # varStep_t = np.zeros([len(yCell),len(xCell)+1])
            # varStep_t[:,:len(xCell)] = speed3_b
            #
            #
            # pSpeed = plt.pcolor(xCell_t,yCell,varStep_t,vmin=vvmin, vmax=vvmax)
            # plt.ylim([-510.0,-320.0])
            # plt.xlim([0,90000])
            # plt.yticks([])
            # plt.xticks([0.0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000],['0', '10', '20', '30', '40','50','60','70','80','90'],fontsize=8)
            #
            # # bbox = ax.get_position()
            # # cax = fig.add_axes([bbox.xmin+bbox.width-0.6, bbox.ymin+0.08, bbox.width*2.0, bbox.height*0.08])
            # # cbar = plt.colorbar(pSpeed, cax = cax,orientation='horizontal')
            # # plt.title("ms",fontsize=10)
            # # cbar.solids.set_rasterized(True)
            # # cbar.solids.set_edgecolor("face")
            #
            #
            #
            #
            #
            #
            # plt.subplots_adjust( wspace=0.1 , bottom=0.05, top=0.99, left=0.03, right=0.975)
            # plt.savefig(outpath+"jdf_Speed_"+str(i+restart)+".png")






            # #todoo: FIGURE: jdf_alt_plot, DSA
            # fig=plt.figure(figsize=(4.5,9.0))
            #
            # ax=fig.add_subplot(2, 1, 1, frameon=True,aspect='equal')
            #
            # xCell_90 = np.linspace(0.0, 90000.0, xli)
            # alt_values_interp = np.interp(xCell_90,site_locations,alt_values)
            #
            # # column mean
            # alt_model_interp_s = np.ones(xli)
            # alt_model_interp_s[:xli] = alt_col_mean_s
            # alt_model_interp_s[alt_model_interp_s==1.0] = None
            #
            # # sites interpolated
            # # alt_values_individual = np.ones(len(site_locations))
            # alt_model_individual = np.interp(site_locations,xCell_90,alt_model_interp_s)
            #
            #
            # plt.plot(alt_values_interp,alt_model_interp_s,c=dark_red)
            # plt.plot([0.0, 30.0],[0.0,30.0],c='k',linestyle='--')
            # plt.scatter(alt_values,alt_model_individual,s=40,c=dark_red,edgecolor=dark_red)
            #
            #
            # # sites error bars
            # ebw = 0.5
            # for j in range(nsites):
            #     # error bar height
            #     plt.plot([lower_eb[j],upper_eb[j]],[alt_model_individual[j],alt_model_individual[j]],c=dark_red)
            #     # lower error bar
            #     plt.plot([lower_eb[j],lower_eb[j]],[alt_model_individual[j]-ebw,alt_model_individual[j]+ebw],c=dark_red)
            #     # upper error bar
            #     plt.plot([upper_eb[j],upper_eb[j]],[alt_model_individual[j]-ebw,alt_model_individual[j]+ebw],c=dark_red)
            #
            #
            # plt.xlabel('Observed alteration fraction')
            # plt.ylabel('Model-predicted alteration fraction')
            #
            # plt.savefig(outpath+"jdf_alt_plot_dsa_"+str(i+restart)+".png")



            if not os.path.exists(outpath+'jdf_amount_sec_x/'):
                os.makedirs(outpath+'jdf_amount_sec_x/')

            #todo: NON-FINAL FIG: sec_amount_x
            fig=plt.figure(figsize=(12.0,3.5))

            amt_lw = 1.25

            norm_amount = np.zeros([steps,minNum+1])
            norm_amount_d = np.zeros([steps,minNum+1])
            norm_amount_a = np.zeros([steps,minNum+1])
            norm_amount_b = np.zeros([steps,minNum+1])

            norm_growth_rate2 = np.zeros([steps,minNum+1])
            norm_growth_rate2_d = np.zeros([steps,minNum+1])
            norm_growth_rate2_a = np.zeros([steps,minNum+1])
            norm_growth_rate2_b = np.zeros([steps,minNum+1])

            #cutoff end!
            co = -1
            x_secStep_ts[co:] = 0.0
            x_secStep_ts_a[co:] = 0.0
            x_secStep_ts_b[co:] = 0.0
            x_secStep_ts_d[co:] = 0.0

            secStep_ts[co:] = 0.0
            secStep_ts_a[co:] = 0.0
            secStep_ts_b[co:] = 0.0
            secStep_ts_d[co:] = 0.0

            # # COLUMN 1
            ng0 = 2



            ax1=fig.add_subplot(1,4,1, frameon=True)

            for j in range(len(any_min)):

                if np.max(x_secStep_ts[ng0:,any_min[j]]) > 0.0:
                    norm_amount[ng0:,any_min[j]] = x_secStep_ts[ng0:,any_min[j]]/np.max(x_secStep_ts[ng0:,any_min[j]])
                plt.plot(np.arange(1+ng0,steps+1),norm_amount[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
                plt.xlim([0,steps+co])
                plt.ylim([-.05,1.05])
                plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
                plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                #plt.xticks([])
                plt.title('min amounts, solo',fontsize=10)

            #plt.legend(fontsize=12,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(1.8, 1.35))
            plt.legend(fontsize=8,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(3.1, 1.3))


            ax1=fig.add_subplot(1,4,2, frameon=True)

            for j in range(len(any_min)):
                if np.max(x_secStep_ts_d[ng0:,any_min[j]]) > 0.0:
                    norm_amount_d[ng0:,any_min[j]] = x_secStep_ts_d[ng0:,any_min[j]]/np.max(x_secStep_ts_d[ng0:,any_min[j]])
                plt.plot(np.arange(1+ng0,steps+1),norm_amount_d[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
                plt.xlim([0,steps+co])
                plt.ylim([-.05,1.05])
                plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
                plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                #plt.xticks([])
                plt.title('min amounts, dual',fontsize=10)


            ax1=fig.add_subplot(1,4,3, frameon=True)

            for j in range(len(any_min)):
                if np.max(x_secStep_ts_a[ng0:,any_min[j]]) > 0.0:
                    norm_amount_a[ng0:,any_min[j]] = x_secStep_ts_a[ng0:,any_min[j]]/np.max(x_secStep_ts_a[ng0:,any_min[j]])
                plt.plot(np.arange(1+ng0,steps+1),norm_amount_a[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
                plt.xlim([0,steps+co])

                plt.ylim([-.05,1.05])
                plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
                plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                #plt.xticks([])
                plt.title('min amounts, a',fontsize=10)


            ax1=fig.add_subplot(1,4,4, frameon=True)

            for j in range(len(any_min)):
                if np.max(x_secStep_ts_b[ng0:,any_min[j]]) > 0.0:
                    norm_amount_b[ng0:,any_min[j]] = x_secStep_ts_b[ng0:,any_min[j]]/np.max(x_secStep_ts_b[ng0:,any_min[j]])
                plt.plot(np.arange(1+ng0,steps+1),norm_amount_b[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
                plt.xlim([0,steps+co])
                plt.ylim([-.05,1.05])
                plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
                plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                plt.xlabel('time',fontsize=8)
                plt.title('min amounts, b',fontsize=10)



            #plt.subplots_adjust(top=0.84, bottom=0.06,hspace=0.15,left=0.05,right=0.95)
            plt.subplots_adjust(top=0.78, bottom=0.12,left=0.05,right=0.95)
            plt.savefig(outpath+'jdf_amount_sec_x/jdf_'+letter+'_amount_sec_X_'+str(i+restart)+'.png')

            if i == 50 or i == 49:
                plt.savefig(outpath+'jdf_amount_sec_x/jdf_'+letter+'_amount_sec_X_'+str(i+restart+1)+'.eps')








            #todo: NON-FINAL FIG: net element uptake in column X

            # elements_sec[,5] = 0.0 # Ca
            # elements_sec[,6] = 0.0 # Mg
            # elements_sec[,7] = 0.0 # Na
            # elements_sec[,8] = 0.0 # K
            # elements_sec[,9] = 0.0 # Fe
            # elements_sec[,10] = 0.0 # S
            # elements_sec[,11] = 0.0 # Si
            # elements_sec[,12] = 0.0 # Cl
            # elements_sec[,13] = 0.0 # Al

            if not os.path.exists(outpath+'jdf_net_uptake_x/'):
                os.makedirs(outpath+'jdf_net_uptake_x/')

            #todo: NON-FINAL FIG: sec_amount_x
            fig=plt.figure(figsize=(14.0,7.0))

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

                    x_elements[i,jj] = x_elements[i,jj] + elements_sec[any_min[j],jj]*x_dsecStep_ts[i,any_min[j]]*(density[any_min[j]]/molar[any_min[j]])
                    x_elements_d[i,jj] = x_elements_d[i,jj] + elements_sec[any_min[j],jj]*x_dsecStep_ts_d[i,any_min[j]]*(density[any_min[j]]/molar[any_min[j]])
                    x_elements_a[i,jj] = x_elements_a[i,jj] + elements_sec[any_min[j],jj]*x_dsecStep_ts_a[i,any_min[j]]*(density[any_min[j]]/molar[any_min[j]])
                    x_elements_b[i,jj] = x_elements_b[i,jj] + elements_sec[any_min[j],jj]*x_dsecStep_ts_b[i,any_min[j]]*(density[any_min[j]]/molar[any_min[j]])

                    # if j == 0:
                    #     x_elements[i,jj] = x_elements[i,jj] + elements_pri[0,jj]*x_dpriStep_ts[i,5]#*(density_pri[3]/molar_pri[3])
                    #     x_elements_d[i,jj] = x_elements_d[i,jj] + elements_pri[0,jj]*x_dpriStep_ts_d[i,5]#*(density_pri[3]/molar_pri[3])
                    #     x_elements_a[i,jj] = x_elements_a[i,jj] + elements_pri[0,jj]*x_dpriStep_ts_a[i,5]#*(density_pri[3]/molar_pri[3])
                    #     x_elements_b[i,jj] = x_elements_b[i,jj] + elements_pri[0,jj]*x_dpriStep_ts_b[i,5]#*(density_pri[3]/molar_pri[3])

                    if j == 0:
                        x_pri_elements[i,jj] = elements_pri[0,jj]*x_dpriStep_ts[i,5]*(density_pri[3]/molar_pri[3])
                        x_pri_elements_d[i,jj] = elements_pri[0,jj]*x_dpriStep_ts_d[i,5]*(density_pri[3]/molar_pri[3])
                        x_pri_elements_a[i,jj] = elements_pri[0,jj]*x_dpriStep_ts_a[i,5]*(density_pri[3]/molar_pri[3])
                        x_pri_elements_b[i,jj] = elements_pri[0,jj]*x_dpriStep_ts_b[i,5]*(density_pri[3]/molar_pri[3])




            net_uptake_kwargs = dict(lw=1.8)
            xt_fs = 8

            ax=fig.add_subplot(2, 4, 1, frameon=True)
            plt.plot(np.arange(steps),x_elements[:,6],color='#bd3706',label='solo', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_d[:,6],color='#88cf23',label='dual', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_a[:,6],color='#23bacf',label='a only', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_b[:,6],color='#6a37e6',label='b only', **net_uptake_kwargs)
            plt.legend(fontsize=9,loc='best',ncol=2,labelspacing=0.0,columnspacing=0.0)
            plt.xticks(fontsize=xt_fs)
            plt.yticks(fontsize=xt_fs)
            plt.title('Mg uptake in column')


            ax=fig.add_subplot(2, 4, 2, frameon=True)
            plt.plot(np.arange(steps),x_elements[:,5],color='#bd3706',label='solo', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_d[:,5],color='#88cf23',label='dual', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_a[:,5],color='#23bacf',label='a only', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_b[:,5],color='#6a37e6',label='b only', **net_uptake_kwargs)
            plt.xticks(fontsize=xt_fs)
            plt.yticks(fontsize=xt_fs)
            plt.title('Ca uptake in column')


            ax=fig.add_subplot(2, 4, 3, frameon=True)
            plt.plot(np.arange(steps),x_elements[:,8],color='#bd3706',label='solo', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_d[:,8],color='#88cf23',label='dual', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_a[:,8],color='#23bacf',label='a only', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_b[:,8],color='#6a37e6',label='b only', **net_uptake_kwargs)
            plt.xticks(fontsize=xt_fs)
            plt.yticks(fontsize=xt_fs)
            plt.title('K uptake in column')


            ax=fig.add_subplot(2, 4, 4, frameon=True)
            plt.plot(np.arange(steps),x_elements[:,9],color='#bd3706',label='solo', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_d[:,9],color='#88cf23',label='dual', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_a[:,9],color='#23bacf',label='a only', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_elements_b[:,9],color='#6a37e6',label='b only', **net_uptake_kwargs)
            plt.xticks(fontsize=xt_fs)
            plt.yticks(fontsize=xt_fs)
            plt.title('Fe uptake in column')




            ax=fig.add_subplot(2, 4, 5, frameon=True)
            plt.plot(np.arange(steps),x_pri_elements[:,6],color='#bd3706',label='solo', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_d[:,6],color='#88cf23',label='dual', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_a[:,6],color='#23bacf',label='a only', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_b[:,6],color='#6a37e6',label='b only', **net_uptake_kwargs)
            plt.legend(fontsize=9,loc='best',ncol=2,labelspacing=0.0,columnspacing=0.0)
            plt.xticks(fontsize=xt_fs)
            plt.yticks(fontsize=xt_fs)
            plt.title('Mg loss pri')

            ax=fig.add_subplot(2, 4, 6, frameon=True)
            plt.plot(np.arange(steps),x_pri_elements[:,5],color='#bd3706',label='solo', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_d[:,5],color='#88cf23',label='dual', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_a[:,5],color='#23bacf',label='a only', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_b[:,5],color='#6a37e6',label='b only', **net_uptake_kwargs)
            plt.xticks(fontsize=xt_fs)
            plt.yticks(fontsize=xt_fs)
            plt.title('Ca loss pri')

            ax=fig.add_subplot(2, 4, 7, frameon=True)
            plt.plot(np.arange(steps),x_pri_elements[:,8],color='#bd3706',label='solo', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_d[:,8],color='#88cf23',label='dual', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_a[:,8],color='#23bacf',label='a only', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_b[:,8],color='#6a37e6',label='b only', **net_uptake_kwargs)
            plt.xticks(fontsize=xt_fs)
            plt.yticks(fontsize=xt_fs)
            plt.title('K loss pri')

            ax=fig.add_subplot(2, 4, 8, frameon=True)
            plt.plot(np.arange(steps),x_pri_elements[:,9],color='#bd3706',label='solo', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_d[:,9],color='#88cf23',label='dual', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_a[:,9],color='#23bacf',label='a only', **net_uptake_kwargs)
            plt.plot(np.arange(steps),x_pri_elements_b[:,9],color='#6a37e6',label='b only', **net_uptake_kwargs)
            plt.xticks(fontsize=xt_fs)
            plt.yticks(fontsize=xt_fs)
            plt.title('Fe loss pri')





            plt.savefig(outpath+'jdf_net_uptake_x/jdf_'+letter+'_net_uptake_X_'+str(i+restart)+'.png',bbox_inches='tight')











            if not os.path.exists(outpath+'jdf_amount_pri_x/'):
                os.makedirs(outpath+'jdf_amount_pri_x/')

            #todo: NON-FINAL FIG: pri_amount_x
            fig=plt.figure(figsize=(12.0,3.5))


            norm_loss_rate = np.zeros([steps,6])
            norm_loss_rate_d = np.zeros([steps,6])
            norm_loss_rate_a = np.zeros([steps,6])
            norm_loss_rate_b = np.zeros([steps,6])

            pri_col = ['', '', '#855d05', '#224e8f', '#156b2d', '#797979']

            # # COLUMN 1

            ax1=fig.add_subplot(1,4,1, frameon=True)
            ax1.grid(True)

            for j in [2, 3, 4, 5]:
                if pri_toggle[j] == 1:
                    plt.plot(np.arange(1,steps+1),x_priStep_ts[:,j]/np.max(x_priStep_ts[:,j]),label=primary[j],c=pri_col[j],lw=1.5)
            plt.xlim([1,steps])
            plt.ylim([0.0,1.05])
            plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
            #plt.yticks(np.linspace(0,1.0,11))
            plt.title('primary amount, solo',fontsize=10)

            #plt.legend(fontsize=10,ncol=2,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(1.0, 1.45))


            ax1=fig.add_subplot(1,4,2, frameon=True)
            ax1.grid(True)

            for j in [2, 3, 4, 5]:
                if pri_toggle_d[j] == 1:
                    plt.plot(np.arange(1,steps+1),x_priStep_ts_d[:,j]/np.max(x_priStep_ts_d[:,j]),label=primary[j],c=pri_col[j],lw=1.5)
            plt.xlim([1,steps])
            plt.ylim([0.0,1.05])
            plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
            plt.title('primary amount, dual',fontsize=10)


            ax1=fig.add_subplot(1,4,3, frameon=True)
            ax1.grid(True)

            for j in [2, 3, 4, 5]:
                if pri_toggle_a[j] == 1:
                    plt.plot(np.arange(1,steps+1),x_priStep_ts_a[:,j]/np.max(x_priStep_ts_a[:,j]),label=primary[j],c=pri_col[j],lw=1.5)
            plt.xlim([1,steps])
            plt.ylim([0.0,1.05])
            plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
            plt.title('primary amount, a',fontsize=10)


            ax1=fig.add_subplot(1,4,4, frameon=True)
            ax1.grid(True)

            for j in [2, 3, 4, 5]:
                if pri_toggle_b[j] == 1:
                    plt.plot(np.arange(1,steps+1),x_priStep_ts_b[:,j]/np.max(x_priStep_ts_b[:,j]),label=primary[j],c=pri_col[j],lw=1.5)
            plt.xlim([1,steps])
            plt.ylim([0.0,1.05])
            plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
            plt.title('primary amount, b',fontsize=10)


            plt.subplots_adjust(top=0.8, bottom=0.1,left=0.05,right=0.95)
            plt.savefig(outpath+'jdf_amount_pri_x/jdf_'+letter+'_amount_pri_X_'+str(i+restart)+'.png')







            if not os.path.exists(outpath+'jdf_corr_x/'):
                os.makedirs(outpath+'jdf_corr_x/')

            #todo: NON-FINAL FIG: jdf_corr_x
            fig=plt.figure(figsize=(12.0,3.5))
            print "jdf_corr_x plot"

            amt_lw = 1.5


            xlim_b = 0.05
            ylim_b = 0.05


            # COLUMN 1 - GLASS AMOUNT


            #plt.scatter(((x_priStep_ts[:,5]-min_maybe)/(np.max(x_priStep_ts[:,5])-min_maybe)),corr_glass_amount[:,any_min[j]],c=col[j], marker='o',lw=0)

            corr_glass_amount = np.zeros([steps,minNum+1,2])
            corr_glass_amount_d = np.zeros([steps,minNum+1,2])
            corr_glass_amount_a = np.zeros([steps,minNum+1,2])
            corr_glass_amount_b = np.zeros([steps,minNum+1,2])
            corr_glass_amount[corr_glass_amount==0.0] = -1.0
            corr_glass_amount_d[corr_glass_amount_d==0.0] = -1.0
            corr_glass_amount_a[corr_glass_amount_a==0.0] = -1.0
            corr_glass_amount_b[corr_glass_amount_b==0.0] = -1.0

            ax1=fig.add_subplot(1,4,1, frameon=True, aspect='equal')

            for j in range(len(any_min)):
                for iii in range(1,i+1):
                    if x_secStep_ts[iii,any_min[j]] > 0.0 and x_priStep_ts[iii,5] > 0.0:
                        corr_glass_amount[iii,any_min[j],1] = (x_secStep_ts[iii,any_min[j]]/np.max(x_secStep_ts[:,any_min[j]])) / 1.0
                min_maybe = -1.0
                if np.max(x_priStep_ts[:,5]) > 0.0:
                    min_array = x_priStep_ts[:,5]
                    min_maybe = np.min(min_array[min_array>0.0])
                corr_glass_amount[:,any_min[j],0] = ((x_priStep_ts[:,5]-min_maybe)/(np.max(x_priStep_ts[:,5])-min_maybe))
                plt.plot(corr_glass_amount[:,any_min[j],0],corr_glass_amount[:,any_min[j],1],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
            # plt.plot([0.0-xlim_b,1.0+xlim_b],[1.0+ylim_b,0.0-ylim_b],lw=20,linestyle='-',c='#eeeeee',zorder=-12)
            # plt.plot([0.0-xlim_b,1.0+xlim_b],[1.0+ylim_b,0.0-ylim_b],lw=50,linestyle='-',c='#f1f1f1',zorder=-13)
            # plt.plot([0.0-xlim_b,1.1+xlim_b],[1.1+ylim_b,0.0-ylim_b],lw=0.5,linestyle='-',c='#a7a7a7')
            # plt.plot([0.0-xlim_b,0.9+xlim_b],[0.9+ylim_b,0.0-ylim_b],lw=0.5,linestyle='-',c='#a7a7a7')

            plt.xlim([0.0-xlim_b,1.0+xlim_b])
            plt.ylim([0.0-ylim_b,1.0+ylim_b])
            plt.xlabel('amount of primary phase (normalized)',fontsize=9)
            plt.ylabel('amount of secondary phase (normalized)',fontsize=9)
            plt.title('corr glass amount, solo',fontsize=10)

            plt.legend(fontsize=8,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(3.1, 1.3))


            ax1=fig.add_subplot(1,4,2, frameon=True, aspect='equal')

            for j in range(len(any_min)):
                for iii in range(1,i+1):
                    if x_secStep_ts_d[iii,any_min[j]] > 0.0 and x_priStep_ts_d[iii,5] > 0.0:
                        corr_glass_amount_d[iii,any_min[j],1] = (x_secStep_ts_d[iii,any_min[j]]/np.max(x_secStep_ts_d[:,any_min[j]])) / 1.0
                min_maybe = -1.0
                if np.max(x_priStep_ts_d[:,5]) > 0.0:
                    min_array = x_priStep_ts_d[:,5]
                    min_maybe = np.min(min_array[min_array>0.0])
                corr_glass_amount_d[:,any_min[j],0] = ((x_priStep_ts_d[:,5]-min_maybe)/(np.max(x_priStep_ts_d[:,5])-min_maybe))
                plt.plot(corr_glass_amount_d[:,any_min[j],0],corr_glass_amount_d[:,any_min[j],1],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
            plt.xlim([-0.05,1.05])
            plt.ylim([-0.05,1.05])
            plt.title('corr glass amount, d',fontsize=10)


            ax1=fig.add_subplot(1,4,3, frameon=True, aspect='equal')

            for j in range(len(any_min)):
                for iii in range(1,i+1):
                    if x_secStep_ts_a[iii,any_min[j]] > 0.0 and x_priStep_ts_a[iii,5] > 0.0:
                        corr_glass_amount_a[iii,any_min[j],1] = (x_secStep_ts_a[iii,any_min[j]]/np.max(x_secStep_ts_a[:,any_min[j]])) / 1.0
                min_maybe = -1.0
                if np.max(x_priStep_ts_a[:,5]) > 0.0:
                    min_array = x_priStep_ts_a[:,5]
                    min_maybe = np.min(min_array[min_array>0.0])
                corr_glass_amount_a[:,any_min[j],0] = ((x_priStep_ts_a[:,5]-min_maybe)/(np.max(x_priStep_ts_a[:,5])-min_maybe))
                plt.plot(corr_glass_amount_a[:,any_min[j],0],corr_glass_amount_a[:,any_min[j],1],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
            plt.xlim([-0.05,1.05])
            plt.ylim([-0.05,1.05])
            plt.title('corr glass amount, a',fontsize=10)


            ax1=fig.add_subplot(1,4,4, frameon=True, aspect='equal')

            for j in range(len(any_min)):
                for iii in range(1,i+1):
                    if x_secStep_ts_b[iii,any_min[j]] > 0.0 and x_priStep_ts_b[iii,5] > 0.0:
                        corr_glass_amount_b[iii,any_min[j],1] = (x_secStep_ts_b[iii,any_min[j]]/np.max(x_secStep_ts_b[:,any_min[j]])) / 1.0
                min_maybe = -1.0
                if np.max(x_priStep_ts_b[:,5]) > 0.0:
                    min_array = x_priStep_ts_b[:,5]
                    min_maybe = np.min(min_array[min_array>0.0])
                corr_glass_amount_b[:,any_min[j],0] = ((x_priStep_ts_b[:,5]-min_maybe)/(np.max(x_priStep_ts_b[:,5])-min_maybe))
                plt.plot(corr_glass_amount_b[:,any_min[j],0],corr_glass_amount_b[:,any_min[j],1],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
            plt.xlim([-0.05,1.05])
            plt.ylim([-0.05,1.05])
            plt.title('corr glass amount, b',fontsize=10)



            plt.subplots_adjust(top=0.8, bottom=0.1,left=0.05,right=0.95)
            plt.savefig(outpath+'jdf_corr_x/jdf_'+letter+'_corr_X_'+str(i+restart)+'.png')

















            #todo: FIG: jdf_corr_min_x
            #
            # if not os.path.exists(outpath+'jdf_corr_min_x/'):
            #     os.makedirs(outpath+'jdf_corr_min_x/')
            #
            #
            # fig=plt.figure(figsize=(12.0,7.0))
            # print "jdf_corr_min_x plot"
            #
            # amt_lw = 1.0
            # ls2 = '--'
            #
            # xlim_b = 0.05
            # ylim_b = 0.05
            #
            # any_min_sel = [14, 16, 17]
            #
            #
            # # COLUMN 1 - GLASS AMOUNT
            #
            # # corr_glass_amount = np.zeros([steps,minNum+1])
            # # corr_glass_amount_d = np.zeros([steps,minNum+1])
            # # corr_glass_amount_a = np.zeros([steps,minNum+1])
            # # corr_glass_amount_b = np.zeros([steps,minNum+1])
            # # corr_glass_amount[corr_glass_amount==0.0] = -1.0
            # # corr_glass_amount_d[corr_glass_amount_d==0.0] = -1.0
            # # corr_glass_amount_a[corr_glass_amount_a==0.0] = -1.0
            # # corr_glass_amount_b[corr_glass_amount_b==0.0] = -1.0
            #
            # ax1=fig.add_subplot(2,4,1, frameon=True, aspect='equal')
            #
            # for j in range(len(any_min_sel)):
            #     plt.plot(corr_glass_amount[:,any_min_sel[j],0],corr_glass_amount[:,any_min_sel[j],1],label=secondary[any_min_sel[j]],c=col_sel[j],lw=amt_lw)
            #     plt.plot(corr_ol_amount[:,any_min_sel[j],0],corr_ol_amount[:,any_min_sel[j],1],c=col_sel[j],lw=amt_lw,linestyle=ls2)
            # # plt.plot([0.0-xlim_b,1.1+xlim_b],[1.1+ylim_b,0.0-ylim_b],lw=0.5,linestyle='-',c='#a7a7a7')
            # # plt.plot([0.0-xlim_b,0.9+xlim_b],[0.9+ylim_b,0.0-ylim_b],lw=0.5,linestyle='-',c='#a7a7a7')
            #
            # plt.xlim([0.0-xlim_b,1.0+xlim_b])
            # plt.ylim([0.0-ylim_b,1.0+ylim_b])
            # plt.xlabel('amount of primary phase (normalized)',fontsize=9)
            # plt.ylabel('amount of secondary phase (normalized)',fontsize=9)
            # plt.title('corr glass amount, solo',fontsize=10)
            #
            # plt.legend(fontsize=10,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(3.1, 1.35))
            #
            #
            # ax1=fig.add_subplot(2,4,2, frameon=True, aspect='equal')
            #
            # for j in range(len(any_min_sel)):
            #     plt.plot(corr_glass_amount_d[:,any_min_sel[j],0],corr_glass_amount_d[:,any_min_sel[j],1],label=secondary[any_min_sel[j]],c=col_sel[j],lw=amt_lw)
            #     plt.plot(corr_ol_amount_d[:,any_min_sel[j],0],corr_ol_amount_d[:,any_min_sel[j],1],c=col_sel[j],lw=amt_lw,linestyle=ls2)
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.title('corr glass amount, d',fontsize=10)
            #
            #
            # ax1=fig.add_subplot(2,4,3, frameon=True, aspect='equal')
            #
            # for j in range(len(any_min_sel)):
            #     plt.plot(corr_glass_amount_a[:,any_min_sel[j],0],corr_glass_amount_a[:,any_min_sel[j],1],label=secondary[any_min_sel[j]],c=col_sel[j],lw=amt_lw)
            #     plt.plot(corr_ol_amount_a[:,any_min_sel[j],0],corr_ol_amount_a[:,any_min_sel[j],1],c=col_sel[j],lw=amt_lw,linestyle=ls2)
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.title('corr glass amount, a',fontsize=10)
            #
            #
            # ax1=fig.add_subplot(2,4,4, frameon=True, aspect='equal')
            #
            # for j in range(len(any_min_sel)):
            #     plt.plot(corr_glass_amount_b[:,any_min_sel[j],0],corr_glass_amount_b[:,any_min_sel[j],1],label=secondary[any_min_sel[j]],c=col_sel[j],lw=amt_lw)
            #     plt.plot(corr_ol_amount_b[:,any_min_sel[j],0],corr_ol_amount_b[:,any_min_sel[j],1],c=col_sel[j],lw=amt_lw,linestyle=ls2)
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.title('corr glass amount, b',fontsize=10)
            #
            #
            #
            #
            #
            # # COLUMN 2 - OLIVINE AMOUNT
            #
            #
            # ax1=fig.add_subplot(2,4,5, frameon=True, aspect='equal')
            #
            # for j in range(len(any_min_sel)):
            #     plt.plot(corr_ol_amount[:,any_min_sel[j],0],corr_ol_amount[:,any_min_sel[j],1],label=secondary[any_min_sel[j]],c=col_sel[j],lw=amt_lw)
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.title('corr ol amount, solo',fontsize=10)
            #
            # #plt.legend(fontsize=10,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(2.1, 1.45))
            #
            #
            # ax1=fig.add_subplot(2,4,6, frameon=True, aspect='equal')
            #
            # for j in range(len(any_min_sel)):
            #     plt.plot(corr_ol_amount_d[:,any_min_sel[j],0],corr_ol_amount_d[:,any_min_sel[j],1],label=secondary[any_min_sel[j]],c=col_sel[j],lw=amt_lw)
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.title('corr ol amount, d',fontsize=10)
            #
            #
            # ax1=fig.add_subplot(2,4,7, frameon=True, aspect='equal')
            #
            # for j in range(len(any_min_sel)):
            #     plt.plot(corr_ol_amount_a[:,any_min_sel[j],0],corr_ol_amount_a[:,any_min_sel[j],1],label=secondary[any_min_sel[j]],c=col_sel[j],lw=amt_lw)
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.title('corr ol amount, a',fontsize=10)
            #
            #
            # ax1=fig.add_subplot(2,4,8, frameon=True, aspect='equal')
            #
            # for j in range(len(any_min_sel)):
            #     plt.plot(corr_ol_amount_b[:,any_min_sel[j],0],corr_ol_amount_b[:,any_min_sel[j],1],label=secondary[any_min_sel[j]],c=col_sel[j],lw=amt_lw)
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.title('corr ol amount, b',fontsize=10)
            #
            #
            #
            # plt.subplots_adjust(top=0.90, bottom=0.03,hspace=0.15,left=0.05,right=0.95)
            # plt.savefig(outpath+'jdf_corr_min_x/jdf_'+letter+'_corr_min_x'+str(i+restart)+'.png')
            #
            # if i == 50 or i == 49:
            #     plt.savefig(outpath+'jdf_corr_min_x/jdf_'+letter+'_corr_min_X_'+str(i+restart)+'.eps')









            if not os.path.exists(outpath+'jdf_deriv2_sim/'):
                os.makedirs(outpath+'jdf_deriv2_sim/')

            #todo: FIG: jdf_deriv2_sim
            fig=plt.figure(figsize=(6.0,6.0))
            print "jdf_deriv2_sim"


            deriv2 = np.zeros([steps,minNum+1])
            deriv2_d = np.zeros([steps,minNum+1])
            deriv2_a = np.zeros([steps,minNum+1])
            deriv2_b = np.zeros([steps,minNum+1])



            for j in range(len(any_min)):
                for iii in range(6,i-1):
                    # deriv2[iii,any_min[j]] = (x_secStep_ts[iii-1,any_min[j]] - 2.0*x_secStep_ts[iii,any_min[j]] + x_secStep_ts[iii+1,any_min[j]])/(1.0*1.0)
                    # deriv2_d[iii,any_min[j]] = (x_secStep_ts_d[iii-1,any_min[j]] - 2.0*x_secStep_ts_d[iii,any_min[j]] + x_secStep_ts_d[iii+1,any_min[j]])/(1.0*1.0)
                    # deriv2_a[iii,any_min[j]] = (x_secStep_ts_a[iii-1,any_min[j]] - 2.0*x_secStep_ts_a[iii,any_min[j]] + x_secStep_ts_a[iii+1,any_min[j]])/(1.0*1.0)
                    # deriv2_b[iii,any_min[j]] = (x_secStep_ts_b[iii-1,any_min[j]] - 2.0*x_secStep_ts_b[iii,any_min[j]] + x_secStep_ts_b[iii+1,any_min[j]])/(1.0*1.0)

                    deriv2[iii,any_min[j]] = (norm_amount[iii-1,any_min[j]] - 2.0*norm_amount[iii,any_min[j]] + norm_amount[iii+1,any_min[j]])/(1.0*1.0)
                    deriv2_d[iii,any_min[j]] = (norm_amount_d[iii-1,any_min[j]] - 2.0*norm_amount_d[iii,any_min[j]] + norm_amount_d[iii+1,any_min[j]])/(1.0*1.0)
                    deriv2_a[iii,any_min[j]] = (norm_amount_a[iii-1,any_min[j]] - 2.0*norm_amount_a[iii,any_min[j]] + norm_amount_a[iii+1,any_min[j]])/(1.0*1.0)
                    deriv2_b[iii,any_min[j]] = (norm_amount_b[iii-1,any_min[j]] - 2.0*norm_amount_b[iii,any_min[j]] + norm_amount_b[iii+1,any_min[j]])/(1.0*1.0)




            ax=fig.add_subplot(2, 1, 1, frameon=True)
            plt.plot([-0.5, len(any_min)+1], [0.0, 0.0], c='#5d5d5d', lw=0.5,zorder=-10)
            for j in range(len(any_min)):
                plt.scatter(j,np.mean(deriv2[2:i-2,any_min[j]]), marker='o', s=50, edgecolor=col[j], facecolor='none', lw=1.5, label=secondary[any_min[j]])
                plt.scatter(j,np.mean(deriv2_d[2:i-2,any_min[j]]), marker='s', s=20, edgecolor=col[j], facecolor=col[j], lw=1.5)
                if np.abs(np.mean(deriv2_a[2:i-2,any_min[j]])) > 0.0:
                    plt.scatter(j,np.mean(deriv2_a[2:i-2,any_min[j]]), marker='^', s=20, edgecolor='k', facecolor='none', lw=1)
                if np.abs(np.mean(deriv2_b[2:i-2,any_min[j]])) > 0.0:
                    plt.scatter(j,np.mean(deriv2_b[2:i-2,any_min[j]]), marker='v', s=20, edgecolor='k', facecolor='none', lw=1)
            plt.xticks(np.arange(len(any_min)),secondary[any_min.astype(int)],fontsize=7)
            plt.xlim([-0.5,len(any_min)-0.5])

            #if np.max(np.abs(deriv2)) > 0.0:
                #plt.legend(fontsize=10, ncol=4, labelspacing=0.0, columnspacing=0.0, bbox_to_anchor=(1.0, 1.15))





            ax=fig.add_subplot(2, 1, 2, frameon=True)

            # print deriv2[:,any_min[j]].shape
            # for j in range(len(any_min)):
            #     plt.plot(deriv2[:,any_min[j]],c=col[j],label=secondary[any_min[j]])
            #
            # if np.max(np.abs(deriv2)) > 0.0:
            #     plt.legend(fontsize=10, ncol=4, labelspacing=0.0, columnspacing=0.0, bbox_to_anchor=(1.0, 1.05))

            plt.plot([-0.5, len(any_min)+1], [0.0, 0.0], c='#5d5d5d', lw=0.5,zorder=-10)
            for j in range(len(any_min)):
                if np.mean(deriv2[2:i-2,any_min[j]]) > 0.0:
                    plt.scatter(j,1.0, marker='o', s=50, edgecolor=col[j], facecolor='none', lw=1.5)
                if np.mean(deriv2[2:i-2,any_min[j]]) < 0.0:
                    plt.scatter(j,-1.0, marker='o', s=50, edgecolor=col[j], facecolor='none', lw=1.5)

                if np.mean(deriv2_d[2:i-2,any_min[j]]) > 0.0:
                    plt.scatter(j,1.0, marker='s', s=20, edgecolor=col[j], facecolor=col[j], lw=1.5)
                if np.mean(deriv2_d[2:i-2,any_min[j]]) < 0.0:
                    plt.scatter(j,-1.0, marker='s', s=20, edgecolor=col[j], facecolor=col[j], lw=1.5)

                if np.mean(deriv2_a[2:i-2,any_min[j]]) > 0.0:
                    plt.scatter(j,1.0, marker='^', s=20, edgecolor='k', facecolor='none', lw=1)
                if np.mean(deriv2_a[2:i-2,any_min[j]]) < 0.0:
                    plt.scatter(j,-1.0, marker='^', s=20, edgecolor='k', facecolor='none', lw=1)


                if np.mean(deriv2_b[2:i-2,any_min[j]]) > 0.0:
                    plt.scatter(j,1.0, marker='v', s=20, edgecolor='k', facecolor='none', lw=1)
                if np.mean(deriv2_b[2:i-2,any_min[j]]) < 0.0:
                    plt.scatter(j,-1.0, marker='v', s=20, edgecolor='k', facecolor='none', lw=1)


            plt.xticks(np.arange(len(any_min)),secondary[any_min.astype(int)],fontsize=7)
            plt.xlim([-0.5,len(any_min)-0.5])


            plt.subplots_adjust(hspace=0.15, bottom=0.05, top=0.9, left=0.1, right=0.95)
            plt.savefig(outpath+"jdf_deriv2_sim/jdf_deriv2_sim_"+str(i+restart)+".png")









    plt.close('all')



#todo: FINAL FIG: secondary_rate
fig=plt.figure(figsize=(9.5,9.5))

norm_growth_rate = np.zeros([steps,minNum+1])
norm_growth_rate_d = np.zeros([steps,minNum+1])
norm_growth_rate_a = np.zeros([steps,minNum+1])
norm_growth_rate_b = np.zeros([steps,minNum+1])

norm_growth_rate2 = np.zeros([steps,minNum+1])
norm_growth_rate2_d = np.zeros([steps,minNum+1])
norm_growth_rate2_a = np.zeros([steps,minNum+1])
norm_growth_rate2_b = np.zeros([steps,minNum+1])

# # COLUMN 1
ng0 = 1

amt_lw = 2

tsteps = np.arange(1+ng0,steps+1)


ax1=fig.add_subplot(2,2,1, frameon=True)

for j in range(len(any_min)):
    if np.max(dsecStep_ts[ng0:,any_min[j]]) > 0.0:
        norm_growth_rate[ng0:,any_min[j]] = dsecStep_ts[ng0:,any_min[j]]
        norm_growth_rate2[ng0:,any_min[j]] = dsecStep_ts[ng0:,any_min[j]]/np.max(dsecStep_ts[ng0:,any_min[j]])

        norm_growth_rate2[30:-1,any_min[j]] = norm_growth_rate2[31:,any_min[j]]
    plt.plot(tsteps[::2],norm_growth_rate2[ng0::2,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    #plt.xlim([5,steps])
    plt.ylim([.5,1.05])
    #plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([])
    plt.title('min growth rate, solo',fontsize=10)

plt.legend(fontsize=12,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(1.8, 1.35))


ax1=fig.add_subplot(2,2,2, frameon=True)

for j in range(len(any_min)):
    if np.max(dsecStep_ts_d[ng0:,any_min[j]]) > 0.0:
        norm_growth_rate_d[ng0:,any_min[j]] = dsecStep_ts_d[ng0:,any_min[j]]
        norm_growth_rate2_d[ng0:,any_min[j]] = dsecStep_ts_d[ng0:,any_min[j]]/np.max(dsecStep_ts_d[ng0:,any_min[j]])

        norm_growth_rate2_d[30:-1,any_min[j]] = norm_growth_rate2_d[31:,any_min[j]]
    plt.plot(tsteps[::2],norm_growth_rate2_d[ng0::2,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    #plt.xlim([5,steps])
    plt.ylim([.5,1.05])
    #plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([])
    plt.title('min growth rate, dual',fontsize=10)


ax1=fig.add_subplot(2,2,3, frameon=True)

for j in range(len(any_min)):
    if np.max(dsecStep_ts_a[ng0:,any_min[j]]) > 0.0:
        norm_growth_rate_a[ng0:,any_min[j]] = dsecStep_ts_a[ng0:,any_min[j]]
        norm_growth_rate2_a[ng0:,any_min[j]] = dsecStep_ts_a[ng0:,any_min[j]]/np.max(dsecStep_ts_a[ng0:,any_min[j]])

        norm_growth_rate2_a[30:-1,any_min[j]] = norm_growth_rate2_a[31:,any_min[j]]

    plt.plot(tsteps[::2],norm_growth_rate2_a[ng0::2,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    #plt.xlim([5,steps])
    plt.ylim([.5,1.05])
    #plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([])
    plt.title('min growth rate, a',fontsize=10)


ax1=fig.add_subplot(2,2,4, frameon=True)

for j in range(len(any_min)):
    if np.max(dsecStep_ts_b[ng0:,any_min[j]]) > 0.0:
        norm_growth_rate_b[ng0:,any_min[j]] = dsecStep_ts_b[ng0:,any_min[j]]
        norm_growth_rate2_b[ng0:,any_min[j]] = dsecStep_ts_b[ng0:,any_min[j]]/np.max(dsecStep_ts_b[ng0:,any_min[j]])
        if any_min[j] == 3:
            print norm_growth_rate2_b[:,3]

        norm_growth_rate2_b[30:-1,any_min[j]] = norm_growth_rate2_b[31:,any_min[j]]

    plt.plot(tsteps[::2],norm_growth_rate2_b[ng0::2,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    #plt.xlim([5,steps])
    plt.ylim([.5,1.05])
    #plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel('time',fontsize=8)
    plt.title('min growth rate, b',fontsize=10)

# ax1=fig.add_subplot(2,2,2, frameon=True)
# for j in range(len(any_min)):
#     plt.plot(dsecStep_ts_d[ng0:,any_min[j]]/dsecStep_ts[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j])


#
# # COLUMN 2 , BIG?
#
# ax1=fig.add_subplot(2,2,2, frameon=True)
#
# plt.plot([0.0,1.0],[0.0, 1.0], lw=1.0, linestyle='--', c='#cccccc',zorder=-1)
# for j in range(len(any_min)):
#     print secondary[any_min[j]]
#     #print " "
#     # ax1_solo = 0.0
#     # ax2_dual = 0.0
#     # print any_min[j]
#     # if np.sum(secStep_ts[-1,any_min[j]]) > 0.0:
#     #     ax1_solo = np.max(dsecStep_ts[2:,any_min[j]])/np.sum(secStep_ts[-1,any_min[j]])
#     # if np.sum(secStep_ts_d[-1,any_min[j]]) > 0.0:
#     #     ax2_dual = np.max(dsecStep_ts_d[2:,any_min[j]])/np.sum(secStep_ts_d[-1,any_min[j]])
#     # print ax1_solo, ax2_dual
#     # print " "
#     max_both = 2.0*np.mean([np.max(norm_growth_rate_d[:,any_min[j]]), np.max(norm_growth_rate[:,any_min[j]])])
#     max_both = np.sum(norm_growth_rate[:,any_min[j]]) + np.sum(norm_growth_rate_d[:,any_min[j]])
#     print max_both
#     plt.scatter(np.sum(norm_growth_rate[2:,any_min[j]])/max_both,np.sum(norm_growth_rate_d[2:,any_min[j]])/max_both,marker='o',label=secondary[any_min[j]],s=40,facecolors=col[j],edgecolor=col[j],lw=2.0)
#     # plt.scatter(norm_growth_rate[-1,any_min[j]],norm_growth_rate_d[-1,any_min[j]],marker='o',label=secondary[any_min[j]],facecolors=col[j],edgecolor=col[j])
#     #plt.scatter(ax1_solo,ax2_dual,marker='o',label=secondary[any_min[j]],facecolors=col[j],edgecolor=col[j])
#
#     plt.xlabel('solo growth rate',fontsize=8)
#     plt.ylabel('dual growth rate',fontsize=8)
#     # plt.xlim([-1.0,4.0])
#     # plt.ylim([-1.0,4.0])
#     plt.title('steady-state growth rate',fontsize=10)
#
#
#
# ax1=fig.add_subplot(2,2,4, frameon=True)
#
# plt.plot([0.0,steps],[1.0, 1.0], lw=40.0, c='#cccccc')
# plt.plot([0.0,steps],[1.0, 1.0], lw=20.0, c='#aaaaaa')
# for j in range(len(any_min)):
#     d_to_s = norm_growth_rate_d[:,any_min[j]]/norm_growth_rate[:,any_min[j]]
#     #d_to_s = d_to_s/np.max(d_to_s)
#     plt.plot(np.arange(1,steps+1),d_to_s,label=secondary[any_min[j]],c=col[j])
#     plt.xlim([10,steps])
#     #plt.ylim([0.95,1.05])
#     #plt.scatter(norm_growth_rate[:,any_min[j]],norm_growth_rate_d[:,any_min[j]],marker='o',label=secondary[any_min[j]],facecolors='none',edgecolor=col[j])
#     # for i in range(0,steps):
#     #     if norm_growth_rate[i,any_min[j]] == 0.0:
#     #         norm_growth_rate[i,any_min[j]] = None
#     #     if norm_growth_rate_d[i,any_min[j]] == 0.0:
#     #         norm_growth_rate_d[i,any_min[j]] = None
#     # plt.plot(norm_growth_rate[:,any_min[j]],norm_growth_rate_d[:,any_min[j]],label=secondary[any_min[j]])
#     #plt.ylim([0.95,1.05])
#     plt.xlabel('time',fontsize=8)
#     plt.title('dual:solo growth rate over time',fontsize=10)


plt.subplots_adjust(top=0.84, bottom=0.06,hspace=0.15,left=0.05,right=0.95)
plt.savefig(outpath+'jdf_secondary_rate_'+letter+'.png')



# plt.savefig(outpath+'all_secondary.eps')









#todo: FINAL FIG: ternary_rainbow_x
fig=plt.figure(figsize=(12.0,3.5))





















#todo: FINAL FIG: secondary_rate_x
fig=plt.figure(figsize=(12.0,3.5))

norm_amount = np.zeros([steps,minNum+1])
norm_amount_d = np.zeros([steps,minNum+1])
norm_amount_a = np.zeros([steps,minNum+1])
norm_amount_b = np.zeros([steps,minNum+1])

# # COLUMN 1
ng0 = 2

ax1=fig.add_subplot(1,4,1, frameon=True)

for j in range(len(any_min)):
    #norm_growth_rate[:,any_min[j]] = 1.0
    if np.max(x_dsecStep_ts[ng0:,any_min[j]]) > 0.0:
        norm_amount[:,any_min[j]] = x_dsecStep_ts[:,any_min[j]]/np.max(x_dsecStep_ts[:,any_min[j]])
    plt.plot(np.arange(1,steps+1),norm_amount[:,any_min[j]],label=secondary[any_min[j]],c=col[j])
    plt.xlim([5,steps])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('min growth rate, solo',fontsize=10)

#plt.legend(fontsize=12,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(1.8, 1.35))
plt.legend(fontsize=8,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(3.1, 1.3))

ax1=fig.add_subplot(1,4,2, frameon=True)

for j in range(len(any_min)):
    #norm_growth_rate_d[:,any_min[j]] = 1.0
    if np.max(x_dsecStep_ts_d[ng0:,any_min[j]]) > 0.0:
        norm_amount_d[ng0:,any_min[j]] = x_dsecStep_ts_d[ng0:,any_min[j]]/np.max(x_dsecStep_ts_d[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_amount_d[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j])
    plt.xlim([5,steps])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('min growth rate, dual',fontsize=10)

ax1=fig.add_subplot(1,4,3, frameon=True)

for j in range(len(any_min)):
    if np.max(x_dsecStep_ts_a[ng0:,any_min[j]]) > 0.0:
        norm_amount_a[ng0:,any_min[j]] = x_dsecStep_ts_a[ng0:,any_min[j]]/np.max(x_dsecStep_ts_a[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_amount_a[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j])
    plt.xlim([5,steps])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('min growth rate, a',fontsize=10)

ax1=fig.add_subplot(1,4,4, frameon=True)

for j in range(len(any_min)):
    if np.max(x_dsecStep_ts_b[ng0:,any_min[j]]) > 0.0:
        norm_amount_b[ng0:,any_min[j]] = x_dsecStep_ts_b[ng0:,any_min[j]]/np.max(x_dsecStep_ts_b[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_amount_b[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j])
    plt.xlim([5,steps])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.xlabel('time',fontsize=8)
    plt.title('min growth rate, b',fontsize=10)


plt.subplots_adjust(top=0.78, bottom=0.12,left=0.05,right=0.95)
#plt.subplots_adjust(top=0.84, bottom=0.06,hspace=0.15,left=0.05,right=0.95)
plt.savefig(outpath+'jdf_secondary_rate_X_'+letter+'.png')
plt.savefig(outpath+'jdf_secondary_rate_X_'+letter+'.eps')










#todo: FINAL FIG: secondary_amount_x
fig=plt.figure(figsize=(12.0,3.5))

amt_lw = 1.25

norm_amount = np.zeros([steps,minNum+1])
norm_amount_d = np.zeros([steps,minNum+1])
norm_amount_a = np.zeros([steps,minNum+1])
norm_amount_b = np.zeros([steps,minNum+1])

norm_amount2 = np.zeros([steps,minNum+1])
norm_amount2_d = np.zeros([steps,minNum+1])
norm_amount2_a = np.zeros([steps,minNum+1])
norm_amount2_b = np.zeros([steps,minNum+1])

#cutoff end!
co = -1
x_secStep_ts[co:] = 0.0
x_secStep_ts_a[co:] = 0.0
x_secStep_ts_b[co:] = 0.0
x_secStep_ts_d[co:] = 0.0

secStep_ts[co:] = 0.0
secStep_ts_a[co:] = 0.0
secStep_ts_b[co:] = 0.0
secStep_ts_d[co:] = 0.0

# # COLUMN 1
ng0 = 2

ax1=fig.add_subplot(1,4,1, frameon=True)

for j in range(len(any_min)):

    if np.max(x_secStep_ts[ng0:,any_min[j]]) > 0.0:
        norm_growth_rate[ng0:,any_min[j]] = x_secStep_ts[ng0:,any_min[j]]
        norm_growth_rate2[ng0:,any_min[j]] = x_secStep_ts[ng0:,any_min[j]]/np.max(x_secStep_ts[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_growth_rate2[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    plt.xlim([0,steps+co])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('min amounts, solo',fontsize=10)

plt.legend(fontsize=8,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(3.1, 1.3))

ax1=fig.add_subplot(1,4,2, frameon=True)

for j in range(len(any_min)):
    if np.max(x_secStep_ts_d[ng0:,any_min[j]]) > 0.0:
        norm_growth_rate_d[ng0:,any_min[j]] = x_secStep_ts_d[ng0:,any_min[j]]
        norm_growth_rate2_d[ng0:,any_min[j]] = x_secStep_ts_d[ng0:,any_min[j]]/np.max(x_secStep_ts_d[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_growth_rate2_d[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    plt.xlim([0,steps+co])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('min amounts, dual',fontsize=10)

ax1=fig.add_subplot(1,4,3, frameon=True)

for j in range(len(any_min)):
    if np.max(x_secStep_ts_a[ng0:,any_min[j]]) > 0.0:

        norm_growth_rate_a[ng0:,any_min[j]] = x_secStep_ts_a[ng0:,any_min[j]]
        norm_growth_rate2_a[ng0:,any_min[j]] = x_secStep_ts_a[ng0:,any_min[j]]/np.max(x_secStep_ts_a[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_growth_rate2_a[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    plt.xlim([0,steps+co])

    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('min amounts, a',fontsize=10)

ax1=fig.add_subplot(1,4,4, frameon=True)

for j in range(len(any_min)):
    if np.max(x_secStep_ts_b[ng0:,any_min[j]]) > 0.0:
        norm_growth_rate_b[ng0:,any_min[j]] = x_secStep_ts_b[ng0:,any_min[j]]
        norm_growth_rate2_b[ng0:,any_min[j]] = x_secStep_ts_b[ng0:,any_min[j]]/np.max(x_secStep_ts_b[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_growth_rate2_b[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    plt.xlim([0,steps+co])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.xlabel('time',fontsize=8)
    plt.title('min amounts, b',fontsize=10)



plt.subplots_adjust(top=0.78, bottom=0.12,left=0.05,right=0.95)
plt.savefig(outpath+'jdf_secondary_amount_X_'+letter+'.png')









#todo: FINAL FIG: secondary_amount
fig=plt.figure(figsize=(12.0,3.5))

norm_amount = np.zeros([steps,minNum+1])
norm_amount_d = np.zeros([steps,minNum+1])
norm_amount_a = np.zeros([steps,minNum+1])
norm_amount_b = np.zeros([steps,minNum+1])

norm_amount2 = np.zeros([steps,minNum+1])
norm_amount2_d = np.zeros([steps,minNum+1])
norm_amount2_a = np.zeros([steps,minNum+1])
norm_amount2_b = np.zeros([steps,minNum+1])

# # COLUMN 1
ng0 = 2

ax1=fig.add_subplot(1,4,1, frameon=True)

for j in range(len(any_min)):

    if np.max(secStep_ts[ng0:,any_min[j]]) > 0.0:
        norm_growth_rate[ng0:,any_min[j]] = secStep_ts[ng0:,any_min[j]]
        norm_growth_rate2[ng0:,any_min[j]] = secStep_ts[ng0:,any_min[j]]/np.max(secStep_ts[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_growth_rate2[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    plt.xlim([0,steps+co])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('min amounts, solo',fontsize=10)

plt.legend(fontsize=8,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(3.1, 1.3))

ax1=fig.add_subplot(1,4,2, frameon=True)

for j in range(len(any_min)):
    if np.max(secStep_ts_d[ng0:,any_min[j]]) > 0.0:
        norm_growth_rate_d[ng0:,any_min[j]] = secStep_ts_d[ng0:,any_min[j]]
        norm_growth_rate2_d[ng0:,any_min[j]] = secStep_ts_d[ng0:,any_min[j]]/np.max(secStep_ts_d[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_growth_rate2_d[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    plt.xlim([0,steps+co])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('min amounts, dual',fontsize=10)

ax1=fig.add_subplot(1,4,3, frameon=True)

for j in range(len(any_min)):
    if np.max(secStep_ts_a[ng0:,any_min[j]]) > 0.0:

        norm_growth_rate_a[ng0:,any_min[j]] = secStep_ts_a[ng0:,any_min[j]]
        norm_growth_rate2_a[ng0:,any_min[j]] = secStep_ts_a[ng0:,any_min[j]]/np.max(secStep_ts_a[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_growth_rate2_a[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    plt.xlim([0,steps+co])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('min amounts, a',fontsize=10)

ax1=fig.add_subplot(1,4,4, frameon=True)

for j in range(len(any_min)):
    if np.max(secStep_ts_b[ng0:,any_min[j]]) > 0.0:
        norm_growth_rate_b[ng0:,any_min[j]] = secStep_ts_b[ng0:,any_min[j]]
        norm_growth_rate2_b[ng0:,any_min[j]] = secStep_ts_b[ng0:,any_min[j]]/np.max(secStep_ts_b[ng0:,any_min[j]])
    plt.plot(np.arange(1+ng0,steps+1),norm_growth_rate2_b[ng0:,any_min[j]],label=secondary[any_min[j]],c=col[j],lw=amt_lw)
    plt.xlim([0,steps+co])
    plt.ylim([-.05,1.05])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.xlabel('time',fontsize=8)
    plt.title('min amounts, b',fontsize=10)



plt.subplots_adjust(top=0.8, bottom=0.1,left=0.05,right=0.95)
plt.savefig(outpath+'jdf_secondary_amount_'+letter+'.png')











#todo: FINAL FIG: primary_amount
fig=plt.figure(figsize=(12.0,3.5))


norm_loss_rate = np.zeros([steps,6])
norm_loss_rate_d = np.zeros([steps,6])
norm_loss_rate_a = np.zeros([steps,6])
norm_loss_rate_b = np.zeros([steps,6])

pri_col = ['', '', 'darkorange', 'grey', 'g', 'k']

# # COLUMN 1

ax1=fig.add_subplot(1,4,1, frameon=True)

for j in [2, 3, 4, 5]:
    if np.max(np.abs(dpriStep_ts[:,j])) > 0.0:
        # norm_loss_rate[:,j] = np.abs(dpriStep_ts[:,j])/np.max(np.abs(dpriStep_ts[:,j]))
        norm_loss_rate[:,j] = np.abs(dpriStep_ts[:,j])/np.max(np.abs(dpriStep_ts[-1,j]))
    plt.plot(np.arange(1,steps+1),priStep_ts[:,5],label=primary[j],c=pri_col[j])
    plt.xlim([1,steps])
    #plt.ylim([0.75,1.05])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('primary amount, solo',fontsize=10)

#plt.legend(fontsize=8,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(1.8, 1.65))


ax1=fig.add_subplot(1,4,2, frameon=True)

for j in [2, 3, 4, 5]:
    if np.max(np.abs(dpriStep_ts_d[:,j])) > 0.0:
        # norm_loss_rate_d[:,j] = np.abs(dpriStep_ts_d[:,j])/np.max(np.abs(dpriStep_ts_d[:,j]))
        norm_loss_rate_d[:,j] = np.abs(dpriStep_ts_d[:,j])/np.max(np.abs(dpriStep_ts_d[-1,j]))
    plt.plot(np.arange(1,steps+1),priStep_ts_d[:,5],label=primary[j],c=pri_col[j])
    plt.xlim([1,steps])
    #plt.ylim([0.75,1.05])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('primary amount, dual',fontsize=10)


ax1=fig.add_subplot(1,4,3, frameon=True)

for j in [2, 3, 4, 5]:
    if np.max(np.abs(dpriStep_ts_a[:,j])) > 0.0:
        # norm_loss_rate_a[:,j] = np.abs(dpriStep_ts_a[:,j])/np.max(np.abs(dpriStep_ts_a[:,j]))
        norm_loss_rate_a[:,j] = np.abs(dpriStep_ts_a[:,j])/np.max(np.abs(dpriStep_ts_a[-1,j]))
    plt.plot(np.arange(1,steps+1),priStep_ts_a[:,5],label=primary[j],c=pri_col[j])
    plt.xlim([1,steps])
    #plt.ylim([0.75,1.05])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('primary amount, a',fontsize=10)


ax1=fig.add_subplot(1,4,4, frameon=True)

for j in [2, 3, 4, 5]:
    if np.max(np.abs(dpriStep_ts_b[:,j])) > 0.0:
        # norm_loss_rate_b[:,j] = np.abs(dpriStep_ts_b[:,j])/np.max(np.abs(dpriStep_ts_b[:,j]))
        norm_loss_rate_b[:,j] = np.abs(dpriStep_ts_b[:,j])/np.max(np.abs(dpriStep_ts_b[-1,j]))
    plt.plot(np.arange(1,steps+1),priStep_ts_b[:,5],label=primary[j],c=pri_col[j])
    plt.xlim([1,steps])
    #plt.ylim([0.75,1.05])
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],fontsize=8)
    plt.title('primary amount, b',fontsize=10)


plt.subplots_adjust(top=0.8, bottom=0.1,left=0.05,right=0.95)
plt.savefig(outpath+'jdf_primary_amount_'+letter+'.png')









#todo: FINAL FIG: primary_amount_x
fig=plt.figure(figsize=(12.0,3.5))


norm_loss_rate = np.zeros([steps,6])
norm_loss_rate_d = np.zeros([steps,6])
norm_loss_rate_a = np.zeros([steps,6])
norm_loss_rate_b = np.zeros([steps,6])

pri_col = ['', '', '#855d05', '#224e8f', '#156b2d', '#797979']

# # COLUMN 1

ax1=fig.add_subplot(1,4,1, frameon=True)
ax1.grid(True)

for j in [2, 3, 4, 5]:
    if pri_toggle[j] == 1:
        plt.plot(np.arange(1,steps+1),x_priStep_ts[:,j]/np.max(x_priStep_ts[:,j]),label=primary[j],c=pri_col[j],lw=2)
plt.xlim([1,steps])
plt.ylim([0.0,1.05])
plt.xticks(np.linspace(0,50,11))
#plt.yticks(np.linspace(0,1.0,11))
plt.title('primary amount, solo',fontsize=10)

#plt.legend(fontsize=10,ncol=2,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(1.0, 1.65))


ax1=fig.add_subplot(1,4,2, frameon=True)
ax1.grid(True)

for j in [2, 3, 4, 5]:
    if pri_toggle_d[j] == 1:
        plt.plot(np.arange(1,steps+1),x_priStep_ts_d[:,j]/np.max(x_priStep_ts_d[:,j]),label=primary[j],c=pri_col[j],lw=2)
plt.xlim([1,steps])
plt.ylim([0.0,1.05])
plt.xticks(np.linspace(0,50,11))
plt.title('primary amount, dual',fontsize=10)


ax1=fig.add_subplot(1,4,3, frameon=True)
ax1.grid(True)

for j in [2, 3, 4, 5]:
    if pri_toggle_a[j] == 1:
        plt.plot(np.arange(1,steps+1),x_priStep_ts_a[:,j]/np.max(x_priStep_ts_a[:,j]),label=primary[j],c=pri_col[j],lw=2)
plt.xlim([1,steps])
plt.ylim([0.0,1.05])
plt.xticks(np.linspace(0,50,11))
plt.title('primary amount, a',fontsize=10)


ax1=fig.add_subplot(1,4,4, frameon=True)
ax1.grid(True)

for j in [2, 3, 4, 5]:
    if pri_toggle_b[j] == 1:
        plt.plot(np.arange(1,steps+1),x_priStep_ts_b[:,j]/np.max(x_priStep_ts_b[:,j]),label=primary[j],c=pri_col[j],lw=2)
plt.xlim([1,steps])
plt.ylim([0.0,1.05])
plt.xticks(np.linspace(0,50,11))
plt.title('primary amount, b',fontsize=10)


plt.subplots_adjust(top=0.8, bottom=0.1,left=0.05,right=0.95)
plt.savefig(outpath+'jdf_primary_amount_X_'+letter+'.png')
