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
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rcParams['axes.titlesize'] = 10
#plt.rcParams['hatch.linewidth'] = 0.1

plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

# col = ['maroon', 'r', 'darkorange', 'gold', 'lawngreen', 'g', 'darkcyan', 'c', 'b', 'navy','purple', 'm', 'hotpink', 'gray', 'k', 'sienna', 'saddlebrown']
col = ['maroon', 'r', 'darkorange', 'lawngreen', 'g', 'c', 'b', 'navy','purple', 'hotpink', 'gray', 'k', 'sienna', 'saddlebrown']

# secondary = np.array(['', 'stilbite', 'aragonite', 'kaolinite', 'albite', 'saponite_mg', 'celadonite', 7
# 'clinoptilolite', 'pyrite', 'mont_na', 'goethite', 'dolomite', 'smectite', 'saponite_k',7
# 'anhydrite', 'siderite', 'calcite', 'quartz', 'kspar', 'saponite_na', 'nont_na', 'nont_mg',8
# 'nont_k', 'nont_h', 'nont_ca', 'muscovite', 'mesolite', 'hematite', 'mont_ca', 'verm_ca',8
# 'analcime', 'phillipsite', 'diopside', 'epidote', 'gismondine', 'hedenbergite', 'chalcedony',7
# 'verm_mg', 'ferrihydrite', 'natrolite', 'talc', 'smectite_low', 'prehnite', 'chlorite',7
# 'scolecite', 'chamosite7a', 'clinochlore14a', 'clinochlore7a', 'saponite_ca', 'verm_na',6
# 'pyrrhotite', 'magnetite', 'lepidocrocite', 'daphnite_7a', 'daphnite_14a', 'verm_k',6
# 'mont_k', 'mont_mg'])2

secondary = np.array(['', 'kaolinite', 'saponite_mg', 'celadonite', 'clinoptilolite', 'pyrite', 'mont_na', 'goethite',
'smectite', 'calcite', 'kspar', 'saponite_na', 'nont_na', 'nont_mg', 'fe_celad', 'nont_ca',
'mesolite', 'hematite', 'mont_ca', 'verm_ca', 'analcime', 'philipsite', 'diopside', 'gismondine',
'verm_mg', 'natrolite', 'talc', 'smectite_low', 'prehnite', 'chlorite', 'scolecite', 'clinochlorte14a',
'clinochlore7a', 'saponite_ca', 'verm_na', 'pyrrhotite', 'daphnite_7a', 'daphnite14a'])

primary = np.array(['', '', 'plagioclase', 'pyroxene', 'olivine', 'basaltic glass'])

density = 2.5*np.ones(37)
molar = 200.0*np.ones(37)

# density = np.array([0, 2.63, 2.3, 3.0,
# 2.15, 5.02, 2.01, 4.27,2.01,
# 2.71, 2.56, 2.3, 2.3, 2.3,
# 2.3, 2.29, 5.3, 2.01, 2.5,
# 2.27, 2.2, 3.41, 2.26,
# 2.5, 2.23, 2.75, 2.01, 2.87, 2.468,
# 2.27, 3.0, 3.0, 3.0, 2.3,2.5,
# 4.62, 3.2, 3.2])
#
#
# density = np.array([0, 2.15-, 2.93-, 2.63, 2.62-, 2.3, 3.0,
# 2.15, 5.02, 2.01, 4.27, 2.84-, 2.01, 2.3-,
# 2.97-, 3.96-, 2.71, 2.65-, 2.56, 2.3, 2.3, 2.3,
# 2.3-, 2.3-, 2.3, 2.81-, 2.29, 5.3, 2.01, 2.5,
# 2.27, 2.2, 3.3-, 3.41, 2.26, 3.56-, 2.65-,
# 2.5, 3.8-, 2.23, 2.75, 2.01, 2.87, 2.468,
# 2.27, 3.0, 3.0, 3.0, 2.3,2.5,
# 4.62, 5.15-, 4.08-, 3.2, 3.2, 2.5-,
# 2.01-, 2.01-])
#
# molar = np.array([0, 480.19, 100.19, 258.16, 263.02, 480.19, 429.02,
# 2742.13, 119.98, 549.07, 88.85, 180.4, 540.46, 480.19,
# 136.14, 115.86, 100.19, 60.08, 278.33, 480.19, 495.9, 495.9,
# 495.9, 495.9, 495.9, 398.71, 1164.9, 159.69, 549.07, 504.19,
# 220.15, 704.93, 216.55, 519.3, 718.55, 248.08, 60.08,
# 504.19, 169.7, 380.22, 379.27, 540.46, 395.38, 67.4,
# 392.34, 664.18, 595.22, 595.22, 480.19, 504.19,
# 85.12, 231.53, 88.85, 664.18, 664.18, 504.19,
# 49.07, 549.07])

# density = np.array([0, 2.15, 2.93, 2.63, 2.62, 2.3, 3.0,
# 2.15, 5.02, 2.01, 4.27, 2.84, 2.01, 2.3,
# 2.97, 3.96, 2.71, 2.65, 2.56, 2.3, 2.3, 2.3,
# 2.3, 2.3, 2.3, 2.81, 2.29, 5.3, 2.01, 2.5,
# 2.27, 2.2, 3.3, 3.41, 2.26, 3.56, 2.65,
# 2.5, 3.8, 2.23, 2.75, 2.01, 2.87, 2.468,
# 2.27, 3.0, 3.0, 3.0, 2.3,2.5,
# 4.62, 5.15, 4.08, 3.2, 3.2, 2.5,
# 2.01, 2.01])
#
# molar = np.array([0, 480.19, 100.19, 258.16, 263.02, 480.19, 429.02,
# 2742.13, 119.98, 549.07, 88.85, 180.4, 540.46, 480.19,
# 136.14, 115.86, 100.19, 60.08, 278.33, 480.19, 495.9, 495.9,
# 495.9, 495.9, 495.9, 398.71, 1164.9, 159.69, 549.07, 504.19,
# 220.15, 704.93, 216.55, 519.3, 718.55, 248.08, 60.08,
# 504.19, 169.7, 380.22, 379.27, 540.46, 395.38, 67.4,
# 392.34, 664.18, 595.22, 595.22, 480.19, 504.19,
# 85.12, 231.53, 88.85, 664.18, 664.18, 504.19,
# 49.07, 549.07])

print secondary.shape
print density.shape
print molar.shape

molar_pri = np.array([110.0, 153.0, 236.0, 277.0])

density_pri = np.array([2.7, 3.0, 3.0, 3.0])

print secondary.shape
print density.shape
print molar.shape

##############
# INITIALIZE #
##############

# #steps = 400
# steps = 50
# # corr = 20 
# corr = 5
# minNum = 57
# ison=10000
# trace = 0
# chem = 1
# iso = 0
# cell = 5
# cellx = 10
# celly = 1

# #steps = 400
# steps = 20
# # corr = 20
# corr = 2
# minNum = 37
# ison=10000
# trace = 0
# chem = 1
# iso = 0
# cell = 1
# cellx = 10
# celly = 1

steps = 25
corr = 2
minNum = 37
ison=10000
trace = 0
chem = 1
iso = 0
cell = 1
cellx = 10
celly = 1

#-LOAD PATH-#
outpath = "../output/revival/coarse_grid/bi_2/"
path = outpath
param_w = 300.0
param_w_rhs = 200.0 


# load output
x0 = np.loadtxt(path + 'x.txt',delimiter='\n')
y0 = np.loadtxt(path + 'y.txt',delimiter='\n')

#-BOOP-#

# format plotting geometry
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
# xCell = xCell[:-1]
# xCell = np.append(xCell, np.max(xCell))
# yCell = y0[1::celly]
yCell = y0[0::celly]
# yCell = np.append(yCell, np.max(yCell))
print "xcell, ycell len: " , len(xCell) , len(yCell)

xg, yg = np.meshgrid(x[:],y[:])
xgh, ygh = np.meshgrid(x[:],y[:])
xgCell, ygCell = np.meshgrid(xCell[:],yCell[:])

mask = np.loadtxt(path + 'mask.txt')
maskP = np.loadtxt(path + 'maskP.txt')
mask_coarse = np.loadtxt(path + 'mask_coarse.txt')
u1 = np.loadtxt(path + 'u.txt')
#print u1[:,29]
#print u1.shape
v1 = np.loadtxt(path + 'v.txt')

u_coarse = np.loadtxt(path + 'u_coarse.txt')
v_coarse = np.loadtxt(path + 'v_coarse.txt')
psi_coarse = np.loadtxt(path + 'psi_coarse.txt')
# psi0 = np.loadtxt(path + 'psiMat.txt')
# rho0 = np.loadtxt(path + 'rhoMat.txt')
# perm0 = np.loadtxt(path + 'permMat.txt')
perm = np.loadtxt(path + 'permeability.txt')
# perm_kx = np.loadtxt(path + 'perm_kx.txt')
# perm_ky = np.loadtxt(path + 'perm_ky.txt')

perm = np.log10(perm)

# temp0 = np.loadtxt(path + 'hMat.txt')
# temp0 = temp0 - 273.0
# u0 = np.loadtxt(path + 'uMat.txt')
# v0 = np.loadtxt(path + 'vMat.txt')
# lambdaMat = np.loadtxt(path + 'lambdaMat.txt')

u_ts = np.zeros([steps])

# lam = np.loadtxt(path + 'lambdaMat.txt')


fig=plt.figure()

grd_msh = np.ones(u_coarse[len(y):,:].shape)

ax1=fig.add_subplot(2,2,1,frameon=False)
pgp = plt.pcolor(mask_coarse[:,:],cmap=cm.rainbow,zorder=-10)
# plt.ylim([-550.0,-250.0])
#msh = ax1.pcolor(xCell,yCell,grd_msh, cmap=cm.Greys_r, facecolor='none', edgecolor='k', zorder=2)
plt.title('mask_coarse')
plt.colorbar(pgp,orientation='horizontal')

ax1=fig.add_subplot(2,2,2,frameon=False)
pgp = plt.pcolor(psi_coarse,cmap=cm.rainbow,zorder=-10)
# plt.ylim([-550.0,-250.0])
# msh = ax1.pcolor(xCell,yCell,grd_msh, cmap=cm.Greys_r, facecolor='none', edgecolor='k', zorder=2)
plt.title('psi_coarse')
plt.colorbar(pgp,orientation='horizontal')

ax1=fig.add_subplot(2,2,3, frameon=False)
#u_coarse = np.abs(u_coarse)
#u_coarse[u_coarse==0.0] = 1.0e-15
#u_coarse = np.log10(u_coarse)
u_coarse = u_coarse*3.14e7
pgu = plt.pcolor(u_coarse,cmap=cm.rainbow,zorder=-10)
#plt.ylim([-550.0,-250.0])
#msh = ax1.pcolor(xCell,yCell,grd_msh, cmap=cm.Greys_r, facecolor='none', edgecolor='k', zorder=2)
plt.title('u_coarse')
plt.colorbar(pgu,orientation='horizontal')

ax1=fig.add_subplot(2,2,4, frameon=False)
v_coarse = np.abs(v_coarse)
v_coarse[v_coarse==0.0] = 1.0e-15
v_coarse = np.log10(v_coarse)
pgv = plt.pcolor(v_coarse,cmap=cm.rainbow,zorder=-10)
#plt.ylim([-550.0,-250.0])
#msh = ax1.pcolor(xCell,yCell,grd_msh, cmap=cm.Greys_r, facecolor='none', edgecolor='k', zorder=2)
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



def chemplot(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, xtix=1, ytix=1, cb=1, cb_title='', cb_min=-10.0, cb_max=10.0):
    #print cb_title
    #print np.abs(np.abs(np.max(varStep)) - np.abs(np.min(varStep[varStep>0.0]))) 
    if np.abs(np.abs(np.max(varStep)) - np.abs(np.min(varStep))) > 0.0:
        #cb_min=0.0
        if cb_min==-10.0 and cb_max==10.0:
            contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),5)
        if cb_max!=10.0:
            contours = np.linspace(cb_min,cb_max,5)
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4,frameon=False)
    
        #pGlass = plt.contourf(xCell,yCell,varStep,contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0,antialiased=True)
        #print contours
        # pGlass = plt.pcolor(xCell,yCell,np.round(varStep,7),cmap=cm.rainbow,vmin=contours[0], vmax=contours[-1])
        pGlass = plt.pcolor(xCell,yCell,varStep,cmap=cm.rainbow,vmin=contours[0], vmax=contours[-1])
    
        #p = plt.contour(xgh,ygh,perm[:,:],[-14.9],colors='black',linewidths=np.array([1.5]))
        plt.yticks([])
        if ytix==1:
            plt.yticks([-450, -400, -350])
        if xtix==0:
            plt.xticks([])
        plt.ylim([np.min(yCell),0.])
        #cMask = plt.contourf(xg,yg,maskP,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
        plt.title(cp_title,fontsize=8)
        plt.ylim([-500,-325.0])
        #plt.ylim([-500,-200.0])
        pGlass.set_edgecolor("face")
        if cb==1:
            #cbaxes = fig.add_axes([0.5, 0.5, 0.3, 0.03]) 
            #cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
            bbox = ax1.get_position()
            #print bbox
            cax = fig.add_axes([bbox.xmin+bbox.width/10.0, bbox.ymin-0.28, bbox.width*0.8, bbox.height*0.13])
            cbar = plt.colorbar(pGlass, cax = cax,orientation='horizontal',ticks=contours[::contour_interval])
            plt.title(cb_title,fontsize=10)
            #cbar = plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::contour_interval],shrink=0.9, pad = 0.5)
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")
        #fig.set_tight_layout(True)
    return chemplot
    

def chemcont(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, xtix=1, ytix=1, perm_lines=1, frame_lines=1, min_color='r',to_hatch=0,hatching='*'):
    varStep[varStep>0.0] = 1.0
    if frame_lines==1:
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4,frameon=True)
    if frame_lines==0:
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4,frameon=False)
    if to_hatch==1:
        pGlass = plt.contourf(xCell,yCell,varStep,[0.5,1.0],colors=[min_color], alpha=0.0, edgecolors=[min_color],hatches=[hatching])
        #pGlass.set_linewidth(0.25)
    if to_hatch==0:
        pGlass = plt.contourf(xCell,yCell,varStep,[0.5,1.0],colors=[min_color], alpha=0.5)
    #pGlass = plt.contour(xCell,yCell,varStep,[0.5,1.0],colors=min_color, alpha=1.0)
    if perm_lines==1:
        p = plt.contour(xgh,ygh,perm[:,:],[-14.9],colors='black',linewidths=np.array([1.0]),zorder=-3)
    plt.yticks([])
    if ytix==1:
        plt.yticks([-450, -400, -350, -300])
    if xtix==0:
        plt.xticks([])
    # plt.ylim([np.min(yCell),0.])
    plt.ylim([-475.0,-325.0])
    plt.title(cp_title,fontsize=10)
    return chemcont
    
    
def chemcont_l(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, xtix=1, ytix=1, perm_lines=1, frame_lines=1, min_cmap=cm.coolwarm, cb_min=-10.0, cb_max=10.0):
    #varStep[varStep>0.0] = 1.0
    if frame_lines==1:
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4*1.5,frameon=True)
    if frame_lines==0:
        ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4*1.5,frameon=False)
    # if to_hatch==1:
    #     pGlass = plt.contourf(xCell,yCell,varStep,5,cmap=min_cmap, alpha=0.0,hatches=[hatching])
        #pGlass.set_linewidth(0.25)
    if np.max(varStep) > 0.0:
        #contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varStep),10)
        contours = np.linspace(cb_min,cb_max+cb_max/10.0,10)
        pGlass = plt.contourf(xCell,yCell,varStep,contours[:],cmap=min_cmap, alpha=1.0)
    #pGlass = plt.contour(xCell,yCell,varStep,[0.5,1.0],colors=min_color, alpha=1.0)
    if perm_lines==1:
        p = plt.contour(xgh,ygh,perm[:,:],[-14.9],colors='black',linewidths=np.array([1.0]),zorder=-3)
    plt.yticks([])
    if ytix==1:
        plt.yticks([-500, -400, -300, -200, -100, 0])
    if xtix==0:
        plt.xticks([])
    # plt.ylim([np.min(yCell),0.])
    #plt.ylim([-500.0,-300.0])
    plt.ylim([-450.0,-350.0])
    plt.title(cp_title,fontsize=10)


    #pGlass.set_edgecolor("face")
    return chemcont_l


#secMat = np.zeros([len(yCell),len(xCell)*steps+1,minNum+1])
secMat = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])
secMat_a = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])
secMat_b = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])
secMat_d = np.zeros([len(yCell),len(xCell)*steps+corr,minNum+1])



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

secStep_ts = np.zeros([steps,minNum+1])
secStep_ts_a = np.zeros([steps,minNum+1])
secStep_ts_b = np.zeros([steps,minNum+1])
secStep_ts_d = np.zeros([steps,minNum+1])

dsecStep_ts = np.zeros([steps,minNum+1])
dsecStep_ts_a = np.zeros([steps,minNum+1])
dsecStep_ts_b = np.zeros([steps,minNum+1])
dsecStep_ts_d = np.zeros([steps,minNum+1])

priStep_ts = np.zeros([steps,6])
priStep_ts_a = np.zeros([steps,6])
priStep_ts_b = np.zeros([steps,6])
priStep_ts_d = np.zeros([steps,6])


dpriStep_ts = np.zeros([steps,6])
dpriStep_ts_a = np.zeros([steps,6])
dpriStep_ts_b = np.zeros([steps,6])
dpriStep_ts_d = np.zeros([steps,6])


any_min = []

if chem == 1:
    # IMPORT MINERALS
    print " "
    ch_path = path + 'ch_s/'
    print "ch_s/:"
    for j in range(1,minNum):
        #if j % 5 ==0:
            #print 'loading minerals', str(j-5), "-", str(j)
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            print j , secondary[j] ,
            secMat[:,:,j] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat[:,:,j] = secMat[:,:,j]*molar[j]/density[j]
            dsecMat[:,2*len(xCell):,j] = secMat[:,len(xCell):-len(xCell),j] - secMat[:,2*len(xCell):,j]
        #satMat[:,:,j] = np.loadtxt(path + 'sat' + str(j) + '.txt')
    inert0 = np.loadtxt(ch_path + 'z_sol_inert.txt')
    dic0 = np.loadtxt(ch_path + 'z_sol_c.txt')
    ca0 = np.loadtxt(ch_path + 'z_sol_ca.txt')
    mg0 = np.loadtxt(ch_path + 'z_sol_mg.txt')
    na0 = np.loadtxt(ch_path + 'z_sol_na.txt')
    k0 = np.loadtxt(ch_path + 'z_sol_k.txt')
    fe0 = np.loadtxt(ch_path + 'z_sol_fe.txt')
    si0 = np.loadtxt(ch_path + 'z_sol_si.txt')
    al0 = np.loadtxt(ch_path + 'z_sol_al.txt')
    ph0 = np.loadtxt(ch_path + 'z_sol_ph.txt')
    alk0 = np.loadtxt(ch_path + 'z_sol_alk.txt')
    glass0 = np.loadtxt(ch_path + 'z_pri_glass.txt')*molar_pri[0]/density_pri[0]
    glass0_p = glass0/(np.max(glass0))
    ol0 = np.loadtxt(ch_path + 'z_pri_ol.txt')*molar_pri[1]/density_pri[1]
    pyr0 = np.loadtxt(ch_path + 'z_pri_pyr.txt')*molar_pri[2]/density_pri[2]
    plag0 = np.loadtxt(ch_path + 'z_pri_plag.txt')*molar_pri[3]/density_pri[3]
    togg0 = np.loadtxt(ch_path + 'z_med_cell_toggle.txt')
    # precip0 = np.loadtxt(ch_path + 'z_med_precip.txt')
    pri_total0 = glass0 + ol0 + pyr0 + plag0
    #pri_total0 = pri_total0/np.max(pri_total0)
    
    
    
    print " "
    ch_path = path + 'ch_a/'
    print "ch_a/:"
    for j in range(1,minNum):
        #if j % 5 ==0:
            #print 'loading minerals', str(j-5), "-", str(j)
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            print j , secondary[j] ,
            secMat_a[:,:,j] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat_a[:,:,j] = secMat_a[:,:,j]*molar[j]/density[j]
            dsecMat_a[:,2*len(xCell):,j] = secMat_a[:,len(xCell):-len(xCell),j] - secMat_a[:,2*len(xCell):,j]
        #satMat[:,:,j] = np.loadtxt(path + 'sat' + str(j) + '.txt')
    inert0_a = np.loadtxt(ch_path + 'z_sol_inert.txt')
    dic0_a = np.loadtxt(ch_path + 'z_sol_c.txt')
    ca0_a = np.loadtxt(ch_path + 'z_sol_ca.txt')
    mg0_a = np.loadtxt(ch_path + 'z_sol_mg.txt')
    na0_a = np.loadtxt(ch_path + 'z_sol_na.txt')
    k0_a = np.loadtxt(ch_path + 'z_sol_k.txt')
    fe0_a = np.loadtxt(ch_path + 'z_sol_fe.txt')
    si0_a = np.loadtxt(ch_path + 'z_sol_si.txt')
    al0_a = np.loadtxt(ch_path + 'z_sol_al.txt')
    ph0_a = np.loadtxt(ch_path + 'z_sol_ph.txt')
    alk0_a = np.loadtxt(ch_path + 'z_sol_alk.txt')
    glass0_a = np.loadtxt(ch_path + 'z_pri_glass.txt')*molar_pri[0]/density_pri[0]
    glass0_p_a = glass0_a/(np.max(glass0_a))
    ol0_a = np.loadtxt(ch_path + 'z_pri_ol.txt')*molar_pri[1]/density_pri[1]
    pyr0_a = np.loadtxt(ch_path + 'z_pri_pyr.txt')*molar_pri[2]/density_pri[2]
    plag0_a = np.loadtxt(ch_path + 'z_pri_plag.txt')*molar_pri[3]/density_pri[3]
    # water0_a = np.loadtxt(ch_path + 'z_med_cell_toggle.txt')
    # precip0_a = np.loadtxt(ch_path + 'z_med_precip.txt')
    pri_total0_a = glass0_a + ol0_a + pyr0_a + plag0_a
    #pri_total0_a = pri_total0_a/np.max(pri_total0_a)
    

    
    
    print " "
    ch_path = path + 'ch_b/'
    print "ch_b/:"
    for j in range(1,minNum):
        #if j % 5 ==0:
            #print 'loading minerals', str(j-5), "-", str(j)
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            print j , secondary[j] ,
            secMat_b[:,:,j] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat_b[:,:,j] = secMat_b[:,:,j]*molar[j]/density[j]
            dsecMat_b[:,2*len(xCell):,j] = secMat_b[:,len(xCell):-len(xCell),j] - secMat_b[:,2*len(xCell):,j]
        #satMat[:,:,j] = np.loadtxt(path + 'sat' + str(j) + '.txt')
    inert0_b = np.loadtxt(ch_path + 'z_sol_inert.txt')
    dic0_b = np.loadtxt(ch_path + 'z_sol_c.txt')
    ca0_b = np.loadtxt(ch_path + 'z_sol_ca.txt')
    mg0_b = np.loadtxt(ch_path + 'z_sol_mg.txt')
    na0_b = np.loadtxt(ch_path + 'z_sol_na.txt')
    k0_b = np.loadtxt(ch_path + 'z_sol_k.txt')
    fe0_b = np.loadtxt(ch_path + 'z_sol_fe.txt')
    si0_b = np.loadtxt(ch_path + 'z_sol_si.txt')
    al0_b = np.loadtxt(ch_path + 'z_sol_al.txt')
    ph0_b = np.loadtxt(ch_path + 'z_sol_ph.txt')
    alk0_b = np.loadtxt(ch_path + 'z_sol_alk.txt')
    glass0_b = np.loadtxt(ch_path + 'z_pri_glass.txt')*molar_pri[0]/density_pri[0]
    glass0_p_b = glass0_b#/(np.max(glass0_b))
    ol0_b = np.loadtxt(ch_path + 'z_pri_ol.txt')*molar_pri[1]/density_pri[1]
    pyr0_b = np.loadtxt(ch_path + 'z_pri_pyr.txt')*molar_pri[2]/density_pri[2]
    plag0_b = np.loadtxt(ch_path + 'z_pri_plag.txt')*molar_pri[3]/density_pri[3]
    # water0_b = np.loadtxt(ch_path + 'z_med_cell_toggle.txt')
    # precip0_b = np.loadtxt(ch_path + 'z_med_precip.txt')
    pri_total0_b = glass0_b + ol0_b + pyr0_b + plag0_b
    #pri_total0_b = pri_total0_b/np.max(pri_total0_b)
    
    
    print " "
    ch_path = path + 'ch_d/'
    print "ch_d/:"
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
        #satMat[:,:,j] = np.loadtxt(path + 'sat' + str(j) + '.txt')
    inert0_d = np.loadtxt(ch_path + 'z_sol_inert.txt')
    dic0_d = np.loadtxt(ch_path + 'z_sol_c.txt')
    ca0_d = np.loadtxt(ch_path + 'z_sol_ca.txt')
    mg0_d = np.loadtxt(ch_path + 'z_sol_mg.txt')
    na0_d = np.loadtxt(ch_path + 'z_sol_na.txt')
    k0_d = np.loadtxt(ch_path + 'z_sol_k.txt')
    fe0_d = np.loadtxt(ch_path + 'z_sol_fe.txt')
    si0_d = np.loadtxt(ch_path + 'z_sol_si.txt')
    al0_d = np.loadtxt(ch_path + 'z_sol_al.txt')
    ph0_d = np.loadtxt(ch_path + 'z_sol_ph.txt')
    alk0_d = np.loadtxt(ch_path + 'z_sol_alk.txt')
    glass0_d = np.loadtxt(ch_path + 'z_pri_glass.txt')*molar_pri[0]/density_pri[0]
    glass0_p_d = glass0_d/(np.max(glass0_d))
    ol0_d = np.loadtxt(ch_path + 'z_pri_ol.txt')*molar_pri[1]/density_pri[1]
    pyr0_d = np.loadtxt(ch_path + 'z_pri_pyr.txt')*molar_pri[2]/density_pri[2]
    plag0_d = np.loadtxt(ch_path + 'z_pri_plag.txt')*molar_pri[3]/density_pri[3]
    #water0_d = np.loadtxt(ch_path + 'z_med_cell_toggle.txt')
    #precip0_d = np.loadtxt(ch_path + 'z_med_precip.txt')
    pri_total0_d = glass0_d + ol0_d + pyr0_d + plag0_d
    #pri_total0_d = pri_total0_d/np.max(pri_total0_d)

    print " "
    print " "
    print "any_min" , any_min
    
    
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

    # lots of this stuff not done for _dual
    smectites0 = np.zeros(secMat[:,:,1].shape)
    zeolites0 = np.zeros(secMat[:,:,1].shape)
    chlorites0 = np.zeros(secMat[:,:,1].shape)
    alt_vol0 = np.zeros(secMat[:,:,1].shape)
    
    smectites0_a = np.zeros(secMat[:,:,1].shape)
    zeolites0_a = np.zeros(secMat[:,:,1].shape)
    chlorites0_a = np.zeros(secMat[:,:,1].shape)
    alt_vol0_a = np.zeros(secMat[:,:,1].shape)
    
    smectites0_b = np.zeros(secMat[:,:,1].shape)
    zeolites0_b = np.zeros(secMat[:,:,1].shape)
    chlorites0_b = np.zeros(secMat[:,:,1].shape)
    alt_vol0_b = np.zeros(secMat[:,:,1].shape)

    smectites0_d = np.zeros(secMat[:,:,1].shape)
    zeolites0_d = np.zeros(secMat[:,:,1].shape)
    chlorites0_d = np.zeros(secMat[:,:,1].shape)
    alt_vol0_d = np.zeros(secMat[:,:,1].shape)
    #
    # smec_list = [9, 28, 56, 57, 5, 13, 19, 20, 21, 22, 23, 24, 48, 12, 41]
    # zeo_list = [6, 26, 30, 34, 38, 44, 39, 31]
    # chlor_list = [43, 45, 46, 47, 53, 54]
    #
    smec_list = [2, 6, 8, 11, 12, 13, 15, 18, 27, 33]
    zeo_list = [4, 16, 20, 21, 23, 25, 30]
    chlor_list = [29, 31, 32, 36, 37]
    
    for j in range(len(smec_list)):
        smectites0 = smectites0 + secMat[:,:,smec_list[j]]
        smectites0_a = smectites0_a + secMat_a[:,:,smec_list[j]]
        smectites0_b = smectites0_b + secMat_b[:,:,smec_list[j]]
        smectites0_d = smectites0_d + secMat_d[:,:,smec_list[j]]
    for j in range(len(zeo_list)):
        zeolites0 = zeolites0 + secMat[:,:,zeo_list[j]]
        zeolites0_a = zeolites0_a + secMat_a[:,:,zeo_list[j]]
        zeolites0_b = zeolites0_b + secMat_b[:,:,zeo_list[j]]
        zeolites0_d = zeolites0_d + secMat_d[:,:,zeo_list[j]]
    for j in range(len(chlor_list)):
        chlorites0 = chlorites0 + secMat[:,:,chlor_list[j]]
        chlorites0_a = chlorites0_a + secMat_a[:,:,chlor_list[j]]
        chlorites0_b = chlorites0_b + secMat_b[:,:,chlor_list[j]]
        chlorites0_d = chlorites0_d + secMat_d[:,:,chlor_list[j]]

    for j in range(len(xCell)*steps):
        for jj in range(len(yCell)):
            if pri_total0[jj,j] > 0.0:
                alt_vol0[jj,j] = np.sum(secMat[jj,j,:])
            if pri_total0[jj,j] + smectites0[jj,j] > 0.0:
                smectites0[jj,j] = smectites0[jj,j] / (pri_total0[jj,j] + smectites0[jj,j])
            if pri_total0[jj,j] + zeolites0[jj,j] > 0.0:
                zeolites0[jj,j] = zeolites0[jj,j] / (pri_total0[jj,j] + zeolites0[jj,j])
            if pri_total0[jj,j] + chlorites0[jj,j] > 0.0:
                chlorites0[jj,j] = chlorites0[jj,j] / (pri_total0[jj,j] + chlorites0[jj,j])
            # if pri_total0[jj,j] + alt_vol0[jj,j] > 0.0:
            #     alt_vol0[jj,j] = alt_vol0[jj,j] #/ (pri_total0[jj,j] + alt_vol0[jj,j])
                
    for j in range(len(xCell)*steps):
        for jj in range(len(yCell)):
            if pri_total0_a[jj,j] > 0.0:
                alt_vol0_a[jj,j] = np.sum(secMat_a[jj,j,:])
            if pri_total0_a[jj,j] + smectites0_a[jj,j] > 0.0:
                smectites0_a[jj,j] = smectites0_a[jj,j] / (pri_total0_a[jj,j] + smectites0_a[jj,j])
            if pri_total0_a[jj,j] + zeolites0_a[jj,j] > 0.0:
                zeolites0_a[jj,j] = zeolites0_a[jj,j] / (pri_total0_a[jj,j] + zeolites0_a[jj,j])
            if pri_total0_a[jj,j] + chlorites0_a[jj,j] > 0.0:
                chlorites0_a[jj,j] = chlorites0_a[jj,j] / (pri_total0_a[jj,j] + chlorites0_a[jj,j])
            # if pri_total0_a[jj,j] + alt_vol0_a[jj,j] > 0.0:
            #     alt_vol0_a[jj,j] = alt_vol0_a[jj,j] #/ (pri_total0_a[jj,j] + alt_vol0_a[jj,j])
                
                
    for j in range(len(xCell)*steps):
        for jj in range(len(yCell)):
            if pri_total0_b[jj,j] > 0.0:
                alt_vol0_b[jj,j] = np.sum(secMat_b[jj,j,:])
            if pri_total0_b[jj,j] + smectites0_b[jj,j] > 0.0:
                smectites0_b[jj,j] = smectites0_b[jj,j] / (pri_total0_b[jj,j] + smectites0_b[jj,j])
            if pri_total0_b[jj,j] + zeolites0_b[jj,j] > 0.0:
                zeolites0_b[jj,j] = zeolites0_b[jj,j] / (pri_total0_b[jj,j] + zeolites0_b[jj,j])
            if pri_total0_b[jj,j] + chlorites0_b[jj,j] > 0.0:
                chlorites0_b[jj,j] = chlorites0_b[jj,j] / (pri_total0_b[jj,j] + chlorites0_b[jj,j])
            # if pri_total0_b[jj,j] + alt_vol0_b[jj,j] > 0.0:
            #     alt_vol0_b[jj,j] = alt_vol0_b[jj,j] #/ (pri_total0_b[jj,j] + alt_vol0_b[jj,j])
                
                
                
    for j in range(len(xCell)*steps):
            for jj in range(len(yCell)):
                if pri_total0_d[jj,j] > 0.0:
                    alt_vol0_d[jj,j] = np.sum(secMat_d[jj,j,:])
                if pri_total0_d[jj,j] + smectites0_d[jj,j] > 0.0:
                    smectites0_d[jj,j] = smectites0_d[jj,j] / (pri_total0_d[jj,j] + smectites0_d[jj,j])
                if pri_total0_d[jj,j] + zeolites0_d[jj,j] > 0.0:
                    zeolites0_d[jj,j] = zeolites0_d[jj,j] / (pri_total0_d[jj,j] + zeolites0_d[jj,j])
                if pri_total0_d[jj,j] + chlorites0_d[jj,j] > 0.0:
                    chlorites0_d[jj,j] = chlorites0_d[jj,j] / (pri_total0_d[jj,j] + chlorites0_d[jj,j])
                # if pri_total0_d[jj,j] + alt_vol0_d[jj,j] > 0.0:
                #     alt_vol0_d[jj,j] = alt_vol0_d[jj,j] #/ (pri_total0_d[jj,j] + alt_vol0_d[jj,j])


    
conv_mean_qu = 0.0
conv_max_qu = 0.0
conv_mean_psi = 0.0
conv_max_psi = 0.0
conv_tot_hf = 0.0
conv_count = 0



#for i in range(4,steps,5):
for i in range(0,steps,1): 
    #print " "
    #print " "
    print "step =", i
    
#-TOG PLOT #
    if i == 1:
        fig=plt.figure()
        ax1=fig.add_subplot(1,1,1, frameon=False)
        plt.pcolor(togg)
        plt.pcolor(togg, cmap=cm.Greys_r, facecolor='none', edgecolor='w', zorder=2)
        fig.savefig(outpath+'togg_plot.png')



    if chem == 1:
        for j in range(len(any_min)):
            secStep[:,:,any_min[j]] = cut_chem(secMat[:,:,any_min[j]],i)
            dsecStep[:,:,any_min[j]] = cut_chem(dsecMat[:,:,any_min[j]],i)
            secStep_ts[i,any_min[j]] = np.sum(secStep[:,:,any_min[j]])
            # print any_min[j]
            if i > 0:
                dsecStep_ts[i,any_min[j]] = secStep_ts[i,any_min[j]] - secStep_ts[i-1,any_min[j]]
        inert = cut_chem(inert0,i)
        dic = cut_chem(dic0,i)
        ca = cut_chem(ca0,i)
        ph = cut_chem(ph0,i)
        alk = cut_chem(alk0,i)
        mg = cut_chem(mg0,i)
        fe = cut_chem(fe0,i)
        si = cut_chem(si0,i)
        k1 = cut_chem(k0,i)
        na = cut_chem(na0,i)
        al = cut_chem(al0,i)
        glass = cut_chem(glass0,i)
        glass_p = cut_chem(glass0_p,i)
        togg = cut_chem(togg0,i)
        smectites = cut_chem(smectites0,i)
        zeolites = cut_chem(zeolites0,i)
        chlorites = cut_chem(chlorites0,i)
        alt_vol = cut_chem(alt_vol0,i)
        # precip = cut_chem(precip0,i)
        pri_total = cut_chem(pri_total0,i)
        ol = cut_chem(ol0,i)
        pyr = cut_chem(pyr0,i)
        plag = cut_chem(plag0,i)
        
        priStep_ts[i,5] = np.sum(glass)
        priStep_ts[i,4] = np.sum(ol)
        priStep_ts[i,3] = np.sum(pyr)
        priStep_ts[i,2] = np.sum(plag)
        if i > 0:
            dpriStep_ts[i,5] = priStep_ts[i,5] - priStep_ts[i-1,5]
            dpriStep_ts[i,4] = priStep_ts[i,4] - priStep_ts[i-1,4]
            dpriStep_ts[i,3] = priStep_ts[i,3] - priStep_ts[i-1,3]
            dpriStep_ts[i,2] = priStep_ts[i,2] - priStep_ts[i-1,2]
        #print dpriStep_ts[i,2]
        
        
        
        
        for j in range(len(any_min)):
            secStep_a[:,:,any_min[j]] = cut_chem(secMat_a[:,:,any_min[j]],i)
            dsecStep_a[:,:,any_min[j]] = cut_chem(dsecMat_a[:,:,any_min[j]],i)
            secStep_ts_a[i,any_min[j]] = np.sum(secStep_a[:,:,any_min[j]])
            if i > 0:
                dsecStep_ts_a[i,any_min[j]] = secStep_ts_a[i,any_min[j]] - secStep_ts_a[i-1,any_min[j]]
        inert_a = cut_chem(inert0_a,i)
        dic_a = cut_chem(dic0_a,i)
        ca_a = cut_chem(ca0_a,i)
        ph_a = cut_chem(ph0_a,i)
        alk_a = cut_chem(alk0_a,i)
        mg_a = cut_chem(mg0_a,i)
        fe_a = cut_chem(fe0_a,i)
        si_a = cut_chem(si0_a,i)
        k1_a = cut_chem(k0_a,i)
        na_a = cut_chem(na0_a,i)
        al_a = cut_chem(al0_a,i)
        glass_a = cut_chem(glass0_a,i)
        glass_p_a = cut_chem(glass0_p_a,i)
        # water_a = cut_chem(water0_a,i)
        smectites_a = cut_chem(smectites0_a,i)
        zeolites_a = cut_chem(zeolites0_a,i)
        chlorites_a = cut_chem(chlorites0_a,i)
        alt_vol_a = cut_chem(alt_vol0_a,i)
        # precip_a = cut_chem(precip0_a,i)
        pri_total_a = cut_chem(pri_total0_a,i)
        ol_a = cut_chem(ol0_a,i)
        pyr_a = cut_chem(pyr0_a,i)
        plag_a = cut_chem(plag0_a,i)
        
        priStep_ts_a[i,5] = np.sum(glass_a)
        priStep_ts_a[i,4] = np.sum(ol_a)
        priStep_ts_a[i,3] = np.sum(pyr_a)
        priStep_ts_a[i,2] = np.sum(plag_a)
        if i > 0:
            dpriStep_ts_a[i,5] = priStep_ts_a[i,5] - priStep_ts_a[i-1,5]
            dpriStep_ts_a[i,4] = priStep_ts_a[i,4] - priStep_ts_a[i-1,4]
            dpriStep_ts_a[i,3] = priStep_ts_a[i,3] - priStep_ts_a[i-1,3]
            dpriStep_ts_a[i,2] = priStep_ts_a[i,2] - priStep_ts_a[i-1,2]
        
        
        
        
        for j in range(len(any_min)):
            secStep_b[:,:,any_min[j]] = cut_chem(secMat_b[:,:,any_min[j]],i)
            dsecStep_b[:,:,any_min[j]] = cut_chem(dsecMat_b[:,:,any_min[j]],i)
            secStep_ts_b[i,any_min[j]] = np.sum(secStep_b[:,:,any_min[j]])
            if i > 0:
                dsecStep_ts_b[i,any_min[j]] = secStep_ts_b[i,any_min[j]] - secStep_ts_b[i-1,any_min[j]]
        inert_b = cut_chem(inert0_b,i)
        dic_b = cut_chem(dic0_b,i)
        ca_b = cut_chem(ca0_b,i)
        ph_b = cut_chem(ph0_b,i)
        alk_b = cut_chem(alk0_b,i)
        mg_b = cut_chem(mg0_b,i)
        fe_b = cut_chem(fe0_b,i)
        si_b = cut_chem(si0_b,i)
        k1_b = cut_chem(k0_b,i)
        na_b = cut_chem(na0_b,i)
        al_b = cut_chem(al0_b,i)
        glass_b = cut_chem(glass0_b,i)
        glass_p_b = cut_chem(glass0_p_b,i)
        # water_b = cut_chem(water0_b,i)
        smectites_b = cut_chem(smectites0_b,i)
        zeolites_b = cut_chem(zeolites0_b,i)
        chlorites_b = cut_chem(chlorites0_b,i)
        alt_vol_b = cut_chem(alt_vol0_b,i)
        # precip_b = cut_chem(precip0_b,i)
        pri_total_b = cut_chem(pri_total0_b,i)
        ol_b = cut_chem(ol0_b,i)
        pyr_b = cut_chem(pyr0_b,i)
        plag_b = cut_chem(plag0_b,i)
        
        priStep_ts_b[i,5] = np.sum(glass_b)
        priStep_ts_b[i,4] = np.sum(ol_b)
        priStep_ts_b[i,3] = np.sum(pyr_b)
        priStep_ts_b[i,2] = np.sum(plag_b)
        
        if i > 0:
            dpriStep_ts_b[i,5] = priStep_ts_b[i,5] - priStep_ts_b[i-1,5]
            dpriStep_ts_b[i,4] = priStep_ts_b[i,4] - priStep_ts_b[i-1,4]
            dpriStep_ts_b[i,3] = priStep_ts_b[i,3] - priStep_ts_b[i-1,3]
            dpriStep_ts_b[i,2] = priStep_ts_b[i,2] - priStep_ts_b[i-1,2]
        #print dpriStep_ts_b[i,2]
        
        # for j in range(1,minNum):
        #     secStep_d[:,:,j] = cut_chem(secMat_d[:,:,j],i)
        #     dsecStep_d[:,:,j] = cut_chem(dsecMat_d[:,:,j],i)
        for j in range(len(any_min)):
            secStep_d[:,:,any_min[j]] = cut_chem(secMat_d[:,:,any_min[j]],i)
            dsecStep_d[:,:,any_min[j]] = cut_chem(dsecMat_d[:,:,any_min[j]],i)
            secStep_ts_d[i,any_min[j]] = np.sum(secStep_d[:,:,any_min[j]])
            #print secStep_ts_d[i,any_min[j]]
            if i > 0:
                dsecStep_ts_d[i,any_min[j]] = secStep_ts_d[i,any_min[j]] - secStep_ts_d[i-1,any_min[j]]
        inert_d = cut_chem(inert0_d,i)
        dic_d = cut_chem(dic0_d,i)
        ca_d = cut_chem(ca0_d,i)
        ph_d = cut_chem(ph0_d,i)
        alk_d = cut_chem(alk0_d,i)
        mg_d = cut_chem(mg0_d,i)
        fe_d = cut_chem(fe0_d,i)
        si_d = cut_chem(si0_d,i)
        k1_d = cut_chem(k0_d,i)
        na_d = cut_chem(na0_d,i)
        al_d = cut_chem(al0_d,i)
        glass_d = cut_chem(glass0_d,i)
        glass_p_d = cut_chem(glass0_p_d,i)
        #water_d = cut_chem(water0_d,i)
        smectites_d = cut_chem(smectites0_d,i)
        zeolites_d = cut_chem(zeolites0_d,i)
        chlorites_d = cut_chem(chlorites0_d,i)
        alt_vol_d = cut_chem(alt_vol0_d,i)
        #precip_d = cut_chem(precip0_d,i)
        pri_total_d = cut_chem(pri_total0_d,i)
        ol_d = cut_chem(ol0_d,i)
        pyr_d = cut_chem(pyr0_d,i)
        plag_d = cut_chem(plag0_d,i)
        
        priStep_ts_d[i,5] = np.sum(glass_d)
        priStep_ts_d[i,4] = np.sum(ol_d)
        priStep_ts_d[i,3] = np.sum(pyr_d)
        priStep_ts_d[i,2] = np.sum(plag_d)
        if i > 0:
            dpriStep_ts_d[i,5] = priStep_ts_d[i,5] - priStep_ts_d[i-1,5]
            dpriStep_ts_d[i,4] = priStep_ts_d[i,4] - priStep_ts_d[i-1,4]
            dpriStep_ts_d[i,3] = priStep_ts_d[i,3] - priStep_ts_d[i-1,3]
            dpriStep_ts_d[i,2] = priStep_ts_d[i,2] - priStep_ts_d[i-1,2]
        
        

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


    

        
    # ##########################
    # #        HEAT PLOT       #
    # ##########################
    #
    # fig=plt.figure()
    #
    #
    # # temp plot
    # varStep = temp
    # varMat = varStep
    # contours = np.linspace(np.min(varMat),np.max(varMat),30)
    #
    # ax1=fig.add_subplot(2,1,2, aspect=asp,frameon=False)
    # pGlass = plt.contourf(x, y, varStep, contours, cmap=cm.rainbow, alpha=1.0,color='#444444',antialiased=True)
    # p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    # CS = plt.contour(xg, yg, psi, 8, colors='black',linewidths=np.array([0.5]))
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    #
    # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::2])
    # cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
    #
    #
    #
    # # v velocity plot
    # varMat = v*(3.14e7)#dic0
    # varStep = v*(3.14e7)#dic
    # contours = np.linspace(np.min(varMat),np.max(varMat),10)
    #
    # ax1=fig.add_subplot(2,2,2, aspect=asp,frameon=False)
    # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    #
    # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    # cbar.ax.set_xlabel('v [m/yr]')
    #
    #
    # # u velocity plot
    # varMat = u*(3.14e7)#ca0
    # varStep = u*(3.14e7)#ca
    # contours = np.linspace(np.min(varMat),np.max(varMat),10)
    #
    # ax1=fig.add_subplot(2,2,1, aspect=asp,frameon=False)
    # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    #
    #
    # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    # cbar.ax.set_xlabel('u [m/yr]')
    #
    #
    #
    # plt.savefig(outpath+'jdf_'+str(i+restart)+'.png')
    

      

    
    chem6 = 6
    
    if chem == 1:
        
########################
#-CHEM 0 PLOT-#        
########################

        fig=plt.figure(figsize=(13.0,7.0))
        plt.subplots_adjust( wspace=0.03, bottom=0.1, top=0.97, left=0.01, right=0.99)

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
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])
        #c_max = 0.015
        
        chemplot(plt_s, plt_ss, 7, 5, 1, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Ca] concentration')
        chemplot(plt_d, plt_dd, 7, 5, 6, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 7, 10, 21, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_bb, plt_bb, 7, 10, 22, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

        plt_s = si0
        plt_a = si0_a
        plt_b = si0_b
        plt_d = si0_d
        plt_ss = si
        plt_aa = si_a
        plt_bb = si_b
        plt_dd = si_d
        all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])

        chemplot(plt_s, plt_ss, 7, 5, 21, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Si] concentration')
        chemplot(plt_d, plt_dd, 7, 5, 26, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 7, 10, 61, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_bb, plt_bb, 7, 10, 62, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

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
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])

        chemplot(plt_s, plt_ss, 7, 5, 2, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Mg] concentration')
        chemplot(plt_d, plt_dd, 7, 5, 7, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 7, 10, 23, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_b, plt_bb, 7, 10, 24, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

        plt_s = fe0
        plt_a = fe0_a
        plt_b = fe0_b
        plt_d = fe0_d
        plt_ss = fe
        plt_aa = fe_a
        plt_bb = fe_b
        plt_dd = fe_d
        all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])

        chemplot(plt_s, plt_ss, 7, 5, 22, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Fe] concentration')
        chemplot(plt_d, plt_dd, 7, 5, 27, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 7, 10, 63, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_bb, plt_bb, 7, 10, 64, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

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
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])

        chemplot(plt_s, plt_ss, 7, 5, 3, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[K] concentration')
        chemplot(plt_d, plt_dd, 7, 5, 8, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 7, 10, 25, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_b, plt_bb, 7, 10, 26, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

        plt_s = alk0
        plt_a = alk0_a
        plt_b = alk0_b
        plt_d = alk0_d
        plt_ss = alk
        plt_aa = alk_a
        plt_bb = alk_b
        plt_dd = alk_d
        all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])

        chemplot(plt_s, plt_ss, 7, 5, 23, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='Alkalinity')
        chemplot(plt_d, plt_dd, 7, 5, 28, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 7, 10, 65, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_bb, plt_bb, 7, 10, 66, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

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
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])
        chemplot(plt_s, plt_ss, 7, 5, 4, 1, 'ph_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='pH')
        chemplot(plt_d, plt_dd, 7, 5, 9, 1, 'ph_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 7, 10, 27, 1, 'ph_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_b, plt_bb, 7, 10, 28, 1, 'ph_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

        plt_s = alt_vol0
        plt_a = alt_vol0_a
        plt_b = alt_vol0_b
        plt_d = alt_vol0_d
        plt_ss = alt_vol
        plt_aa = alt_vol_a
        plt_bb = alt_vol_b
        plt_dd = alt_vol_d
        all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])

        chemplot(plt_s, plt_ss, 7, 5, 24, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='Alteration volume')
        chemplot(plt_d, plt_dd, 7, 5, 29, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 7, 10, 67, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_bb, plt_bb, 7, 10, 68, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

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
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])
        chemplot(plt_s, plt_ss, 7, 5, 5, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Al]')
        chemplot(plt_s, plt_dd, 7, 5, 10, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 7, 10, 29, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_b, plt_bb, 7, 10, 30, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        
        plt_s = pri_total0
        plt_a = pri_total0_a
        plt_b = pri_total0_b
        plt_d = pri_total0_d
        plt_ss = pri_total
        plt_aa = pri_total_a
        plt_bb = pri_total_b
        plt_dd = pri_total_d
        all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])

        chemplot(plt_s, plt_ss, 7, 5, 25, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='Fraction basalt remaining')
        chemplot(plt_d, plt_dd, 7, 5, 30, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 7, 10, 69, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_bb, plt_bb, 7, 10, 70, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

        plt.savefig(outpath+'jdfSol0_'+str(i+restart)+'.png')
        
########################
#-CHEM 2 SECONDARIES-#
########################

        fig=plt.figure(figsize=(13.0,7.0))
        plt.subplots_adjust( wspace=0.03, bottom=0.1, top=0.97, left=0.01, right=0.99)
        
        for am in range(len(any_min)):
            
            if am < 5:
                am_p = am
                am_pp = 2*am+1
                am_ppp = 2*(am-1)+1
            if am >= 5:
                am_p = 20+(am-5)
                am_pp = 40 + 2*(am-5)+1
                am_ppp = 50 + 2*(am-4)
            
            print any_min[am]
            # # col 1
            plt_s = secMat[:,:,any_min[am]]
            plt_a = secMat_a[:,:,any_min[am]]
            plt_b = secMat_b[:,:,any_min[am]]
            plt_d = secMat_d[:,:,any_min[am]]
            plt_ss = secStep[:,:,any_min[am]]
            plt_aa = secStep_a[:,:,any_min[am]]
            plt_bb = secStep_b[:,:,any_min[am]]
            plt_dd = secStep_d[:,:,any_min[am]]
            #all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
            c_min = 0.0#np.amin(all_ch[all_ch>0.0])
            #print "c_min" , c_min
            all_ch = [np.max(plt_s), np.max(plt_a), np.max(plt_b), np.max(plt_d)]
            # c_max = np.amax(all_ch[all_ch>0.0])
            c_max = np.max(all_ch)
            #print "c_max" , c_max
            #c_max = 0.015
        
            chemplot(plt_s, plt_ss, 7, 5, 1+am_p, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title=secondary[any_min[am]])
            chemplot(plt_d, plt_dd, 7, 5, 6+am_p, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_a, plt_aa, 7, 10, 20+am_pp, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
            chemplot(plt_bb, plt_bb, 7, 10, 21+am_pp, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
    
        
        plt.savefig(outpath+'jdfChem2_'+str(i+restart)+'.png')
        

########################
#-CHEM PRI 0 PRIMARIES-# 
########################


        fig=plt.figure(figsize=(13.0,7.0))
        #plt.subplots_adjust( wspace=0.03, bottom=0.15, top=0.97, left=0.01, right=0.99)
        plt.subplots_adjust( wspace=0.03, bottom=0.1, top=0.97, left=0.01, right=0.99)

        # # col 1
        plt_s = pri_total0
        plt_a = pri_total0_d
        plt_b = pri_total0_d
        plt_d = pri_total0_d 
        plt_ss = pri_total
        plt_aa = pri_total_d
        plt_bb = pri_total_d
        plt_dd = pri_total_d
        all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])

        chemplot(plt_s, plt_ss, 7, 5, 1, 1, 'basalt remaining s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='basalt remaining')
        chemplot(plt_d, plt_dd, 7, 5, 6, 1, 'basalt remaining d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 21, 1, 'basalt remaining a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 22, 1, 'basalt remaining b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

        # # col 2
        plt_s = glass0
        plt_a = glass0_a
        plt_b = glass0_a
        plt_d = glass0_a
        plt_ss = glass
        plt_aa = glass_a
        plt_bb = glass_a
        plt_dd = glass_a
        all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])

        chemplot(plt_s, plt_ss, 7, 5, 2, 1, 'glass_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='glass')
        chemplot(plt_d, plt_dd, 7, 5, 7, 1, 'glass_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        #  chemplot(plt_a, plt_aa, 3, 10, 23, 1, 'glass_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_bb, plt_bb, 3, 10, 24, 1, 'glass_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

        # # col 3
        plt_s = ol0
        plt_a = ol0_b
        plt_b = ol0_b
        plt_d = ol0_b
        plt_ss = ol
        plt_aa = ol_b
        plt_bb = ol_b
        plt_dd = ol_b
        all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])

        chemplot(plt_s, plt_ss, 7, 5, 3, 1, 'ol_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='olivine')
        chemplot(plt_d, plt_dd, 7, 5, 8, 1, 'ol_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 25, 1, 'ol_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 26, 1, 'ol_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

        # # col 4
        plt_s = pyr0
        plt_a = pyr0_b
        plt_b = pyr0_b
        plt_d = pyr0_b
        plt_ss = pyr
        plt_aa = pyr_b
        plt_bb = pyr_b
        plt_dd = pyr_b
        all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])
        chemplot(plt_s, plt_ss, 7, 5, 4, 1, 'pyr_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='pyroxene')
        chemplot(plt_d, plt_dd, 7, 5, 9, 1, 'pyr_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 27, 1, 'pyr_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 28, 1, 'pyr_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

        # # col 5
        plt_s = plag0
        plt_a = plag0_b
        plt_b = plag0_b
        plt_d = plag0_b
        plt_ss = plag
        plt_aa = plag_b
        plt_bb = plag_b
        plt_dd = plag_b
        all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        c_min = np.amin(all_ch[all_ch>0.0])
        all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        c_max = np.amax(all_ch[all_ch>0.0])
        chemplot(plt_s, plt_ss, 7, 5, 5, 1, 'plag_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='plagioclase')
        chemplot(plt_d, plt_dd, 7, 5, 10, 1, 'plag_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 29, 1, 'plag_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 30, 1, 'plag_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)

        #fig.set_tight_layout(True)
        # plt.subplots_adjust( wspace=0.05 , bottom=0.04, top=0.97, left=0.03, right=0.975)
        plt.savefig(outpath+'jdfPri0_'+str(i+restart)+'.png')


##############################
#-CHEM 1 BINARY SECONDARIES-#
##############################

        fig=plt.figure(figsize=(11.0,4.25))

        chemcont(smectites0, smectites, 3, 3, 1, 1, 'mineral distribution in solo chamber', xtix=0, ytix=1,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        chemcont(zeolites0, zeolites, 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        chemcont(chlorites0, chlorites, 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        chemcont(secMat[:,:,26], secStep[:,:,26], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='gold',to_hatch=0)
        chemcont(secMat[:,:,7], secStep[:,:,7], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        chemcont(secMat[:,:,5], secStep[:,:,5], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='O')
        chemcont(secMat[:,:,3], secStep[:,:,3], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=0)
        #chemcont(secMat[:,:,16], secStep[:,:,16], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='grey',to_hatch=0)


        chemcont(smectites0_d, smectites_d, 3, 3, 4, 1, 'mineral distribution in dual chamber (a + b)', xtix=0, ytix=1,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        chemcont(zeolites0_d, zeolites_d, 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        chemcont(chlorites0_d, chlorites_d, 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        chemcont(secMat_d[:,:,26], secStep_d[:,:,26], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='gold',to_hatch=0)
        chemcont(secMat_d[:,:,7], secStep_d[:,:,7], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        chemcont(secMat_d[:,:,5], secStep_d[:,:,5], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='O')
        chemcont(secMat_d[:,:,3], secStep_d[:,:,3], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=0)
        #chemcont(secMat_d[:,:,16], secStep_d[:,:,16], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='grey',to_hatch=0)


        chemcont(smectites0_a, smectites_a, 3, 6, 13, 1, 'chamber a only', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        chemcont(zeolites0_a, zeolites_a, 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        chemcont(chlorites0_a, chlorites_a, 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        chemcont(secMat_a[:,:,26], secStep_a[:,:,26], 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='gold',to_hatch=0)
        chemcont(secMat_a[:,:,7], secStep_a[:,:,7], 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        chemcont(secMat_a[:,:,5], secStep_a[:,:,5], 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='O')
        chemcont(secMat_a[:,:,3], secStep_a[:,:,3], 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=0)
        #chemcont(secMat_a[:,:,16], secStep_a[:,:,16], 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='grey',to_hatch=0)


        chemcont(smectites0_b, smectites_b, 3, 6, 14, 1, 'chamber b only', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        chemcont(zeolites0_b, zeolites_b, 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        chemcont(chlorites0_b, chlorites_b, 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        chemcont(secMat_b[:,:,26], secStep_b[:,:,26], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='gold',to_hatch=0)
        chemcont(secMat_b[:,:,7], secStep_b[:,:,7], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        chemcont(secMat_b[:,:,5], secStep_b[:,:,5], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='O')
        chemcont(secMat_b[:,:,3], secStep_b[:,:,3], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=0)
        #chemcont(secMat_b[:,:,16], secStep_b[:,:,16], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='grey',to_hatch=0)


        lg1 = Patch(facecolor='r', label='smectites', alpha=0.5)
        lg2 = Patch(facecolor='g', label='zeolites', alpha=0.5)
        lg3 = Patch(facecolor='gold', label='talc', alpha=0.5)
        lg4 = Patch(facecolor='w', label='chlorites', hatch ='////')
        lg5 = Patch(facecolor='w', label='goethite', hatch ='\\')
        lg6 = Patch(facecolor='w', label='pyrite', hatch ='O')
        lg7 = Patch(facecolor='b', label='celadonite', alpha=0.5)
        lg8 = Patch(facecolor='grey', label='caco3', alpha=0.5)

        plt.legend([lg1,lg2,lg3,lg4,lg5,lg6,lg7,lg8],['smectites','zeolites','talc', 'chlorites','goethite','pyrite','celadonite','caco3'],fontsize=10,ncol=3,bbox_to_anchor=(0.0, -2.0),loc=8)


        if np.max(dsecStep[:,:,16]) != 0.0:
            plt_s = dsecStep[:,:,16]
            plt_a = dsecStep_a[:,:,16]
            plt_b = dsecStep_b[:,:,16]
            all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0])]
            c_min = np.amin(all_ch[all_ch>0.0])
            all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0])]
            c_max = np.amax(all_ch[all_ch>0.0])

            chemcont_l(dsecMat[:,:,16], dsecStep[:,:,16], 3, 3, 2, 1, 'calcite growth rate in solo chamber', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_cmap=cm.Blues, cb_min=c_min, cb_max=c_max)
            chemcont_l(dsecMat_a[:,:,16]+dsecMat_b[:,:,16], dsecStep_a[:,:,16]+dsecStep_b[:,:,16], 3, 3, 5, 1, 'dual chamber (a+b growth rate)', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_cmap=cm.Blues, cb_min=c_min, cb_max=c_max)
            chemcont_l(dsecMat_a[:,:,16], dsecStep_a[:,:,16], 3, 6, 15, 1, 'chamber a only', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_cmap=cm.Blues, cb_min=c_min, cb_max=c_max)
            chemcont_l(dsecMat_b[:,:,16], dsecStep_b[:,:,16], 3, 6, 16, 1, 'chamber b only', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_cmap=cm.Blues, cb_min=c_min, cb_max=c_max)



        # if np.max(dsecStep[:,:,16]) != 0.0:
#             plt_s = dsecStep[:,:,16]
#             plt_a = dsecStep_a[:,:,16]
#             plt_b = dsecStep_b[:,:,16]
#             all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0])]
#             c_min = np.amin(all_ch[all_ch>0.0])
#             all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0])]
#             c_max = np.amax(all_ch[all_ch>0.0])
#
#             chemcont_l(dsecMat[:,:,16], dsecStep[:,:,16], 3, 3, 2, 1, 'calcite growth rate in solo chamber', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_cmap=cm.Blues, cb_min=c_min, cb_max=c_max)
#             chemcont_l(dsecMat_a[:,:,16]+dsecMat_b[:,:,16], dsecStep_a[:,:,16]+dsecStep_b[:,:,16], 3, 3, 5, 1, 'dual chamber (a+b growth rate)', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_cmap=cm.Blues, cb_min=c_min, cb_max=c_max)
#             chemcont_l(dsecMat_a[:,:,16], dsecStep_a[:,:,16], 3, 6, 15, 1, 'chamber a only', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_cmap=cm.Blues, cb_min=c_min, cb_max=c_max)
#             chemcont_l(dsecMat_b[:,:,16], dsecStep_b[:,:,16], 3, 6, 16, 1, 'chamber b only', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_cmap=cm.Blues, cb_min=c_min, cb_max=c_max)
#

        # chemcont_l(dsecMat_a[:,:,16]+dsecMat_b[:,:,16], dsecStep_a[:,:,16]+dsecStep_b[:,:,16], 3, 3, 5, 1, 'chamber a+b minerals', xtix=1, ytix=0,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        # chemcont_l(dsecMat_a[:,:,46]+dsecMat_b[:,:,46], dsecStep_a[:,:,46]+dsecStep_b[:,:,46], 3, 3, 5, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        # chemcont_l(dsecMat_b[:,:,40], dsecStep_b[:,:,40], 3, 3, 5, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='gold',to_hatch=0)
        # chemcont_l(dsecMat_a[:,:,5]+dsecMat_b[:,:,5], dsecStep_a[:,:,5]+dsecStep_b[:,:,5], 3, 3, 5, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        # chemcont_l(dsecMat_a[:,:,10]+dsecMat_b[:,:,10], dsecStep_a[:,:,10]+dsecStep_b[:,:,10], 3, 3, 5, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        #
        #
        #
        #

        # chemcont_l(secMat_a[:,:,16], secStep_a[:,:,16], 3, 6, 15, 1, 'chamber a minerals', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        # chemcont_l(secMat_a[:,:,46], secStep_a[:,:,46], 3, 6, 15, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        # chemcont_l(secMat_a[:,:,5], secStep_a[:,:,5], 3, 6, 15, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        # chemcont_l(secMat_a[:,:,10], secStep_a[:,:,10], 3, 6, 15, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        #
        # chemcont_l(secMat_b[:,:,16], secStep_b[:,:,16], 3, 6, 16, 1, 'chamber b minerals', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        # chemcont_l(secMat_b[:,:,46], secStep_b[:,:,46], 3, 6, 16, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        # chemcont_l(secMat_b[:,:,40], secStep_b[:,:,40], 3, 6, 16, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='gold',to_hatch=0)
        # chemcont_l(secMat_b[:,:,5], secStep_b[:,:,5], 3, 6, 16, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        # chemcont_l(secMat_b[:,:,10], secStep_b[:,:,10], 3, 6, 16, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        #

        # # print np.max(ca0)
        # # print np.min(ca0)
        # chemplot(ca0, ca, 3, 2, 1, 1, '[Ca]')
        # chemplot(glass0_p, glass_p, 3, 2, 2, 1, 'BASLATIC GLASS [fraction remaining]')
        # chemplot(ph0_a, ph_a, 3, 2, 3, 2, 'pH')
        # chemplot(alk0, alk, 3, 2, 4, 1, 'alkalinity')
        # chemplot(dic0, dic, 3, 2, 5, 1, 'DIC')
        # chemplot(secMat[:,:,16], secStep[:,:,16], 3, 2, 6, 1, 'CALCITE [cm$^3$]')

        #fig.set_tight_layout(True)
        plt.subplots_adjust( wspace=0.05 , bottom=0.12, top=0.95, left=0.03, right=0.975)
        plt.savefig(outpath+'jdfChem1_'+str(i+restart)+'.png')

 
    plt.close('all')
    

#-FULL SECONDARY SUMMARY-# 

fig=plt.figure(figsize=(6.5,6.5))

# ax1=fig.add_subplot(4,2,1, frameon=True)
#
# for j in range(len(any_min)):
#     # print np.arange(1,steps).shape
#     # print secStep_ts[:,any_min[j]].shape
#     plt.plot(np.arange(1,steps+1),secStep_ts[:,any_min[j]],label=secondary[any_min[j]],c=col[j])
#     plt.title('all mins, solo',fontsize=10)
#
#

norm_growth_rate = np.zeros([steps,minNum+1])
norm_growth_rate_d = np.zeros([steps,minNum+1])
norm_growth_rate_a = np.zeros([steps,minNum+1])
norm_growth_rate_b = np.zeros([steps,minNum+1])

norm_growth_rate2 = np.zeros([steps,minNum+1])
norm_growth_rate2_d = np.zeros([steps,minNum+1])
norm_growth_rate2_a = np.zeros([steps,minNum+1])
norm_growth_rate2_b = np.zeros([steps,minNum+1])

# # COLUMN 1

print " "
ax1=fig.add_subplot(4,2,1, frameon=True)

for j in range(len(any_min)):
    
    #norm_growth_rate[:,any_min[j]] = 1.0
    if np.max(dsecStep_ts[:,any_min[j]]) > 0.0:
        #print any_min[j]
        # norm_growth_rate[:,any_min[j]] = dsecStep_ts[:,any_min[j]]/np.max(dsecStep_ts[:,any_min[j]])
        norm_growth_rate[:,any_min[j]] = dsecStep_ts[:,any_min[j]]#/dsecStep_ts[-1,any_min[j]]
        norm_growth_rate2[:,any_min[j]] = dsecStep_ts[:,any_min[j]]/np.max(dsecStep_ts[:,any_min[j]])
    plt.plot(np.arange(1,steps+1),norm_growth_rate2[:,any_min[j]],label=secondary[any_min[j]],c=col[j])
    #plt.plot(np.arange(1,steps+1),norm_growth_rate[:,any_min[j]]/np.max(norm_growth_rate[:,any_min[j]]),label=secondary[any_min[j]],c=col[j])
    plt.xlim([5,steps])
    #plt.ylim([0.95,1.05])
    plt.ylim([0.0,1.05])
    plt.xticks([])
    plt.title('min growth rate, solo',fontsize=10)
    
plt.legend(fontsize=8,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(1.8, 1.65))

print " "
ax1=fig.add_subplot(4,2,3, frameon=True)

for j in range(len(any_min)):
    #norm_growth_rate_d[:,any_min[j]] = 1.0
    if np.max(dsecStep_ts_d[:,any_min[j]]) > 0.0:
        #print any_min[j]
        # norm_growth_rate_d[:,any_min[j]] = dsecStep_ts_d[:,any_min[j]]/np.max(dsecStep_ts_d[:,any_min[j]])
        norm_growth_rate_d[:,any_min[j]] = dsecStep_ts_d[:,any_min[j]]#/dsecStep_ts_d[-1,any_min[j]]
        norm_growth_rate2_d[:,any_min[j]] = dsecStep_ts_d[:,any_min[j]]/np.max(dsecStep_ts_d[:,any_min[j]])
    plt.plot(np.arange(1,steps+1),norm_growth_rate2_d[:,any_min[j]],label=secondary[any_min[j]],c=col[j])
    #plt.plot(np.arange(1,steps+1),norm_growth_rate_d[:,any_min[j]]/np.max(norm_growth_rate_d[:,any_min[j]]),label=secondary[any_min[j]],c=col[j])
    plt.xlim([5,steps])
    #plt.ylim([0.95,1.05])
    plt.ylim([0.0,1.05])
    plt.xticks([])
    plt.title('min growth rate, dual',fontsize=10)

print " "
ax1=fig.add_subplot(4,2,5, frameon=True)

for j in range(len(any_min)):
    if np.max(secStep_ts_a[:,any_min[j]]) > 0.0:
        #print any_min[j]
        # norm_growth_rate_a[:,any_min[j]] = dsecStep_ts_a[:,any_min[j]]/np.max(dsecStep_ts_a[:,any_min[j]])
        norm_growth_rate_a[:,any_min[j]] = dsecStep_ts_a[:,any_min[j]]#/dsecStep_ts_a[-1,any_min[j]]
        norm_growth_rate2_a[:,any_min[j]] = dsecStep_ts_a[:,any_min[j]]/np.max(dsecStep_ts_a[:,any_min[j]])
    plt.plot(np.arange(1,steps+1),norm_growth_rate2_a[:,any_min[j]],label=secondary[any_min[j]],c=col[j])
    #plt.plot(np.arange(1,steps+1),norm_growth_rate_a[:,any_min[j]]/np.max(norm_growth_rate_a[:,any_min[j]]),label=secondary[any_min[j]],c=col[j])
    plt.xlim([5,steps])
    #plt.ylim([0.95,1.05])
    plt.ylim([0.0,1.05])
    plt.xticks([])
    plt.title('min growth rate, a',fontsize=10)
   
print " " 
ax1=fig.add_subplot(4,2,7, frameon=True)

for j in range(len(any_min)):
    if np.max(secStep_ts_b[:,any_min[j]]) > 0.0:
        #print any_min[j]
        #norm_growth_rate_b[:,any_min[j]] = dsecStep_ts_b[:,any_min[j]]/np.max(dsecStep_ts_b[:,any_min[j]])
        norm_growth_rate_b[:,any_min[j]] = dsecStep_ts_b[:,any_min[j]]#/dsecStep_ts_b[-1,any_min[j]]
        norm_growth_rate2_b[:,any_min[j]] = dsecStep_ts_b[:,any_min[j]]/np.max(dsecStep_ts_a[:,any_min[j]])
    plt.plot(np.arange(1,steps+1),norm_growth_rate2_b[:,any_min[j]],label=secondary[any_min[j]],c=col[j])
    #plt.plot(np.arange(1,steps+1),norm_growth_rate_b[:,any_min[j]]/np.max(norm_growth_rate_b[:,any_min[j]]),label=secondary[any_min[j]],c=col[j])
    plt.xlim([5,steps])
    #plt.ylim([0.95,1.05])
    plt.ylim([0.0,1.05])
    plt.xlabel('time',fontsize=8)
    plt.title('min growth rate, b',fontsize=10)
    
    
    
# COLUMN 2 , BIG?
    
ax1=fig.add_subplot(2,2,2, frameon=True)

plt.plot([0.0,1.0],[0.0, 1.0], lw=1.0, linestyle='--', c='#cccccc',zorder=-1)
for j in range(len(any_min)):
    print secondary[any_min[j]]
    #print " "
    # ax1_solo = 0.0
    # ax2_dual = 0.0
    # print any_min[j]
    # if np.sum(secStep_ts[-1,any_min[j]]) > 0.0:
    #     ax1_solo = np.max(dsecStep_ts[2:,any_min[j]])/np.sum(secStep_ts[-1,any_min[j]])
    # if np.sum(secStep_ts_d[-1,any_min[j]]) > 0.0:
    #     ax2_dual = np.max(dsecStep_ts_d[2:,any_min[j]])/np.sum(secStep_ts_d[-1,any_min[j]])
    # print ax1_solo, ax2_dual
    # print " "
    max_both = 2.0*np.mean([np.max(norm_growth_rate_d[:,any_min[j]]), np.max(norm_growth_rate[:,any_min[j]])])
    max_both = np.sum(norm_growth_rate[:,any_min[j]]) + np.sum(norm_growth_rate_d[:,any_min[j]])
    print max_both
    plt.scatter(np.sum(norm_growth_rate[2:,any_min[j]])/max_both,np.sum(norm_growth_rate_d[2:,any_min[j]])/max_both,marker='o',label=secondary[any_min[j]],s=40,facecolors=col[j],edgecolor=col[j],lw=2.0)
    # plt.scatter(norm_growth_rate[-1,any_min[j]],norm_growth_rate_d[-1,any_min[j]],marker='o',label=secondary[any_min[j]],facecolors=col[j],edgecolor=col[j])
    #plt.scatter(ax1_solo,ax2_dual,marker='o',label=secondary[any_min[j]],facecolors=col[j],edgecolor=col[j])
    
    plt.xlabel('solo growth rate',fontsize=8)
    plt.ylabel('dual growth rate',fontsize=8)
    # plt.xlim([-1.0,4.0])
    # plt.ylim([-1.0,4.0])
    plt.title('steady-state growth rate',fontsize=10)
    
    
    
ax1=fig.add_subplot(2,2,4, frameon=True)

plt.plot([0.0,steps],[1.0, 1.0], lw=40.0, c='#cccccc')
plt.plot([0.0,steps],[1.0, 1.0], lw=20.0, c='#aaaaaa')
for j in range(len(any_min)):
    d_to_s = norm_growth_rate_d[:,any_min[j]]/norm_growth_rate[:,any_min[j]]
    #d_to_s = d_to_s/np.max(d_to_s)
    plt.plot(np.arange(1,steps+1),d_to_s,label=secondary[any_min[j]],c=col[j])
    plt.xlim([10,steps])
    #plt.ylim([0.95,1.05])
    #plt.scatter(norm_growth_rate[:,any_min[j]],norm_growth_rate_d[:,any_min[j]],marker='o',label=secondary[any_min[j]],facecolors='none',edgecolor=col[j])
    # for i in range(0,steps):
    #     if norm_growth_rate[i,any_min[j]] == 0.0:
    #         norm_growth_rate[i,any_min[j]] = None
    #     if norm_growth_rate_d[i,any_min[j]] == 0.0:
    #         norm_growth_rate_d[i,any_min[j]] = None
    # plt.plot(norm_growth_rate[:,any_min[j]],norm_growth_rate_d[:,any_min[j]],label=secondary[any_min[j]])
    #plt.ylim([0.95,1.05])
    plt.xlabel('time',fontsize=8)
    plt.title('dual:solo growth rate over time',fontsize=10)


plt.subplots_adjust(top=0.88, bottom=0.06,hspace=0.25,left=0.05,right=0.95)
plt.savefig(outpath+'all_ts_sec.png')
















#-FULL PRIMARY SUMMARY-# 

fig=plt.figure(figsize=(6.5,6.5))


norm_loss_rate = np.zeros([steps,6])
norm_loss_rate_d = np.zeros([steps,6])
norm_loss_rate_a = np.zeros([steps,6])
norm_loss_rate_b = np.zeros([steps,6])

pri_col = ['', '', 'darkorange', 'grey', 'g', 'k']

# # COLUMN 1

ax1=fig.add_subplot(4,2,1, frameon=True)

for j in [2, 3, 4, 5]:
    if np.max(np.abs(dpriStep_ts[:,j])) > 0.0:
        # norm_loss_rate[:,j] = np.abs(dpriStep_ts[:,j])/np.max(np.abs(dpriStep_ts[:,j]))
        norm_loss_rate[:,j] = np.abs(dpriStep_ts[:,j])/np.max(np.abs(dpriStep_ts[-1,j]))
    plt.plot(np.arange(1,steps+1),norm_loss_rate[:,j],label=primary[j],c=pri_col[j])
    plt.xlim([10,steps])
    #plt.ylim([0.75,1.05])
    plt.xticks([])
    plt.title('primary loss rate, solo',fontsize=10)
    
plt.legend(fontsize=8,ncol=4,labelspacing=0.0,columnspacing=0.0,bbox_to_anchor=(1.8, 1.65))


ax1=fig.add_subplot(4,2,3, frameon=True)

for j in [2, 3, 4, 5]:
    if np.max(np.abs(dpriStep_ts_d[:,j])) > 0.0:
        # norm_loss_rate_d[:,j] = np.abs(dpriStep_ts_d[:,j])/np.max(np.abs(dpriStep_ts_d[:,j]))
        norm_loss_rate_d[:,j] = np.abs(dpriStep_ts_d[:,j])/np.max(np.abs(dpriStep_ts_d[-1,j]))
    plt.plot(np.arange(1,steps+1),norm_loss_rate_d[:,j],label=primary[j],c=pri_col[j])
    plt.xlim([10,steps])
    #plt.ylim([0.75,1.05])
    plt.xticks([])
    plt.title('primary loss rate, dual',fontsize=10)
    
    
ax1=fig.add_subplot(4,2,5, frameon=True)

for j in [2, 3, 4, 5]:
    if np.max(np.abs(dpriStep_ts_a[:,j])) > 0.0:
        # norm_loss_rate_a[:,j] = np.abs(dpriStep_ts_a[:,j])/np.max(np.abs(dpriStep_ts_a[:,j]))
        norm_loss_rate_a[:,j] = np.abs(dpriStep_ts_a[:,j])/np.max(np.abs(dpriStep_ts_a[-1,j]))
    plt.plot(np.arange(1,steps+1),norm_loss_rate_a[:,j],label=primary[j],c=pri_col[j])
    plt.xlim([10,steps])
    #plt.ylim([0.75,1.05])
    plt.xticks([])
    plt.title('primary loss rate, a',fontsize=10)
    
    
ax1=fig.add_subplot(4,2,7, frameon=True)

for j in [2, 3, 4, 5]:
    if np.max(np.abs(dpriStep_ts_b[:,j])) > 0.0:
        # norm_loss_rate_b[:,j] = np.abs(dpriStep_ts_b[:,j])/np.max(np.abs(dpriStep_ts_b[:,j]))
        norm_loss_rate_b[:,j] = np.abs(dpriStep_ts_b[:,j])/np.max(np.abs(dpriStep_ts_b[-1,j]))
    plt.plot(np.arange(1,steps+1),norm_loss_rate_b[:,j],label=primary[j],c=pri_col[j])
    plt.xlim([10,steps])
    #plt.ylim([0.75,1.05])
    #plt.xticks([])
    plt.title('primary loss rate, b',fontsize=10)


plt.subplots_adjust(top=0.88, bottom=0.06,hspace=0.25,left=0.05,right=0.95)
plt.savefig(outpath+'all_ts_pri.png')