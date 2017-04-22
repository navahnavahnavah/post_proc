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
plt.rcParams['axes.titlesize'] = 12
#plt.rcParams['hatch.linewidth'] = 0.1

plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

secondary = np.array(['', 'stilbite', 'aragonite', 'kaolinite', 'albite', 'saponite_mg', 'celadonite',
'clinoptilolite', 'pyrite', 'mont_na', 'goethite', 'dolomite', 'smectite', 'saponite_k',
'anhydrite', 'siderite', 'calcite', 'quartz', 'kspar', 'saponite_na', 'nont_na', 'nont_mg',
'nont_k', 'nont_h', 'nont_ca', 'muscovite', 'mesolite', 'hematite', 'mont_ca', 'verm_ca',
'analcime', 'phillipsite', 'diopside', 'epidote', 'gismondine', 'hedenbergite', 'chalcedony',
'verm_mg', 'ferrihydrite', 'natrolite', 'talc', 'smectite_low', 'prehnite', 'chlorite',
'scolecite', 'chamosite7a', 'clinochlore14a', 'clinochlore7a', 'saponite_ca', 'verm_na',
'pyrrhotite', 'magnetite', 'lepidocrocite', 'daphnite_7a', 'daphnite_14a', 'verm_k',
'mont_k', 'mont_mg'])

density = np.array([0, 2.15, 2.93, 2.63, 2.62, 2.3, 3.0, 
2.15, 5.02, 2.01, 4.27, 2.84, 2.01, 
2.3, 2.97, 3.96, 2.71, 2.65, 2.56, 2.3, 
2.3, 2.3, 2.3, 2.3, 2.3, 2.81, 2.29, 5.3,
2.01, 2.5, 2.27, 2.2, 3.3, 3.41, 2.26, 
3.56, 2.65, 2.5, 3.8, 2.23, 2.75, 2.01, 
2.87, 2.468, 2.27, 3.0, 3.0, 3.0, 2.3,
2.5, 4.62, 5.15, 4.08, 3.2, 3.2, 2.5,
2.01, 2.01])

molar = np.array([0, 480.19, 100.19, 258.16, 263.02, 480.19, 429.02, 
2742.13, 119.98, 549.07, 88.85, 180.4, 540.46, 
480.19, 136.14, 115.86, 100.19, 60.08, 278.33, 480.19, 
495.9, 495.9, 495.9, 495.9, 495.9, 398.71, 1164.9, 159.69,
549.07, 504.19, 220.15, 704.93, 216.55, 519.3, 718.55, 
248.08, 60.08, 504.19, 169.7, 380.22, 379.27, 540.46, 
395.38, 67.4, 392.34, 664.18, 595.22, 595.22, 480.19,
504.19, 85.12, 231.53, 88.85, 664.18, 664.18, 504.19,
49.07, 549.07])

molar_pri = np.array([110.0, 153.0, 236.0, 277.0])

density_pri = np.array([2.7, 3.0, 3.0, 3.0])

print secondary.shape
print density.shape
print molar.shape

##############
# INITIALIZE #
##############

#steps = 400
steps = 10
# corr = 20
corr = 1
minNum = 57
ison=10000
trace = 0
chem = 1
iso = 0
cell = 5
cellx = 10
celly = 1

outpath = "output/revival/coarse_grid/coarse_cons/"
path = outpath
param_w = 300.0
param_w_rhs = 200.0


# load output
x0 = np.loadtxt(path + 'x.txt',delimiter='\n')
y0 = np.loadtxt(path + 'y.txt',delimiter='\n')



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
u1 = np.loadtxt(path + 'u.txt')
print u1[:,29]
print u1.shape
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
#plt.plot(x0)
ax1=fig.add_subplot(2,2,1, aspect=asp,frameon=False)
pgu = plt.pcolor(u1)
plt.colorbar(pgu,orientation='horizontal')

# ax1=fig.add_subplot(2,2,2, aspect=asp,frameon=False)
# pgv = plt.pcolor(v1)
# plt.colorbar(pgv,orientation='horizontal')

ax1=fig.add_subplot(2,2,2, aspect=asp/10.0,frameon=False)
pgp = plt.pcolor(psi_coarse)
plt.colorbar(pgp,orientation='horizontal')

ax1=fig.add_subplot(2,2,3, aspect=asp/10.0,frameon=False)
u_coarse = np.abs(u_coarse)
u_coarse[u_coarse==0.0] = 1.0e-15
#u_coarse = np.log10(u_coarse)
u_coarse = u_coarse*3.14e7
pgu = plt.pcolor(u_coarse)
plt.colorbar(pgu,orientation='horizontal')

ax1=fig.add_subplot(2,2,4, aspect=asp/10.0,frameon=False)
v_coarse = np.abs(v_coarse)
v_coarse[v_coarse==0.0] = 1.0e-15
v_coarse = np.log10(v_coarse)
pgv = plt.pcolor(v_coarse)
plt.colorbar(pgv,orientation='horizontal')

fig.savefig(outpath+'x.png')



def cut(geo0,index):
    #geo_cut = geo0[(index*len(y0)/cell):(index*len(y0)/cell+len(y0)/cell),:]
    geo_cut = geo0[:,(index*len(x0)):(index*len(x0)+len(x0))]
    geo_cut = np.append(geo_cut, geo_cut[-1:,:], axis=0)
    geo_cut = np.append(geo_cut, geo_cut[:,-1:], axis=1)
    return geo_cut
    
def cut_chem(geo0,index):
    geo_cut_chem = geo0[:,(index*len(xCell)):(index*len(xCell)+len(xCell))]
    return geo_cut_chem




#######################
##### CHEM PCOLOR #####
#######################

def chemplot(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, xtix=1, ytix=1, cb=1, cb_title='', cb_min=-10.0, cb_max=10.0):
    #cb_min=0.0
    if cb_min==-10.0 and cb_max==10.0:
        contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),5)
    if cb_max!=10.0:
        contours = np.linspace(cb_min,cb_max,5)
    ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp*4,frameon=False)
    
    #pGlass = plt.contourf(xCell,yCell,varStep,contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0,antialiased=True)
    #print contours
    pGlass = plt.pcolor(xCell,yCell,np.round(varStep,7),cmap=cm.rainbow,vmin=contours[0], vmax=contours[-1])
    
    #p = plt.contour(xgh,ygh,perm[:,:],[-14.9],colors='black',linewidths=np.array([1.5]))
    plt.yticks([])
    if ytix==1:
        plt.yticks([-450, -400, -350, -300])
    if xtix==0:
        plt.xticks([])
    plt.ylim([np.min(yCell),0.])
    #cMask = plt.contourf(xg,yg,maskP,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
    plt.title(cp_title,fontsize=8)
    plt.ylim([-550.0,-250.0])
    pGlass.set_edgecolor("face")
    if cb==1:
        #cbaxes = fig.add_axes([0.5, 0.5, 0.3, 0.03]) 
        #cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
        bbox = ax1.get_position()
        #print bbox
        cax = fig.add_axes([bbox.xmin+bbox.width/10.0, bbox.ymin-0.6, bbox.width*0.8, bbox.height*0.15])
        cbar = plt.colorbar(pGlass, cax = cax,orientation='horizontal',ticks=contours[::contour_interval],label=cb_title)
        #cbar = plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::contour_interval],shrink=0.9, pad = 0.5)
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor("face")
    #fig.set_tight_layout(True)
    return chemplot
    
    
########################
##### CHEM CONTOUR #####
########################
    
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
    
    
    
##########################
##### CHEM CONTOUR L #####
##########################
    
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


if chem == 1:
    # IMPORT MINERALS
    ch_path = path + 'ch_s/'
    print "ch_s/:"
    for j in range(1,minNum):
        #if j % 5 ==0:
            #print 'loading minerals', str(j-5), "-", str(j)
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            print j , secondary[j]
            secMat[:,:,j] = np.loadtxt(ch_path + 'z_sec' + str(j) + '.txt')
            secMat[:,:,j] = secMat[:,:,j]*molar[j]/density[j]
            dsecMat[:,2*len(xCell):,j] = secMat[:,len(xCell):-len(xCell),j] - secMat[:,2*len(xCell):,j]
            
            #geo_cut_chem = geo0[:,(index*len(xCell)):(index*len(xCell)+len(xCell))]
            
            
            
            
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
    water0 = np.loadtxt(ch_path + 'z_med_cell_toggle.txt')
    precip0 = np.loadtxt(ch_path + 'z_med_precip.txt')
    pri_total0 = glass0 + ol0 + pyr0 + plag0
    pri_total0 = pri_total0/np.max(pri_total0)
    
    ch_path = path + 'ch_a/'
    print "ch_a/:"
    for j in range(1,minNum):
        #if j % 5 ==0:
            #print 'loading minerals', str(j-5), "-", str(j)
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            print j , secondary[j]
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
    water0_a = np.loadtxt(ch_path + 'z_med_cell_toggle.txt')
    precip0_a = np.loadtxt(ch_path + 'z_med_precip.txt')
    pri_total0_a = glass0_a + ol0_a + pyr0_a + plag0_a
    pri_total0_a = pri_total0_a/np.max(pri_total0_a)
    
    ch_path = path + 'ch_b/'
    print "ch_b/:"
    for j in range(1,minNum):
        #if j % 5 ==0:
            #print 'loading minerals', str(j-5), "-", str(j)
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            print j , secondary[j]
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
    glass0_p_b = glass0_b/(np.max(glass0_b))
    ol0_b = np.loadtxt(ch_path + 'z_pri_ol.txt')*molar_pri[1]/density_pri[1]
    pyr0_b = np.loadtxt(ch_path + 'z_pri_pyr.txt')*molar_pri[2]/density_pri[2]
    plag0_b = np.loadtxt(ch_path + 'z_pri_plag.txt')*molar_pri[3]/density_pri[3]
    water0_b = np.loadtxt(ch_path + 'z_med_cell_toggle.txt')
    precip0_b = np.loadtxt(ch_path + 'z_med_precip.txt')
    pri_total0_b = glass0_b + ol0_b + pyr0_b + plag0_b
    pri_total0_b = pri_total0_b/np.max(pri_total0_b)
    
    
    ch_path = path + 'ch_d/'
    print "ch_d/:"
    for j in range(1,minNum):
        #if j % 5 ==0:
            #print 'loading minerals', str(j-5), "-", str(j)
        if os.path.isfile(ch_path + 'z_sec' + str(j) + '.txt'):
            print j , secondary[j]
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
    pri_total0_d = pri_total0_d/np.max(pri_total0_d)


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

    smectites0 = secMat[:,:,9] + secMat[:,:,28] + secMat[:,:,56] + secMat[:,:,57] + secMat[:,:,5] + secMat[:,:,13] + secMat[:,:,19] + secMat[:,:,20] + secMat[:,:,21] + secMat[:,:,22] + secMat[:,:,23] + secMat[:,:,24] + secMat[:,:,48] + secMat[:,:,12] + secMat[:,:,41]
    zeolites0 = secMat[:,:,6] + secMat[:,:,26] + secMat[:,:,30] + secMat[:,:,34] + secMat[:,:,38] + secMat[:,:,44] + secMat[:,:,39] + secMat[:,:,31]
    chlorites0 = secMat[:,:,43] + secMat[:,:,45] + secMat[:,:,46] + secMat[:,:,47] + secMat[:,:,53] + secMat[:,:,54]
    
    smectites0 = secMat_a[:,:,9] + secMat_a[:,:,28] + secMat_a[:,:,56] + secMat_a[:,:,57] + secMat_a[:,:,5] + secMat_a[:,:,13] + secMat_a[:,:,19] + secMat_a[:,:,20] + secMat_a[:,:,21] + secMat_a[:,:,22] + secMat_a[:,:,23] + secMat_a[:,:,24] + secMat_a[:,:,48] + secMat_a[:,:,12] + secMat_a[:,:,41]
    zeolites0 = secMat_a[:,:,6] + secMat_a[:,:,26] + secMat_a[:,:,30] + secMat_a[:,:,34] + secMat_a[:,:,38] + secMat_a[:,:,44] + secMat_a[:,:,39] + secMat_a[:,:,31]
    chlorites0 = secMat_a[:,:,43] + secMat_a[:,:,45] + secMat_a[:,:,46] + secMat_a[:,:,47] + secMat_a[:,:,53] + secMat_a[:,:,54]
    
    
    smectites0 = secMat_b[:,:,9] + secMat_b[:,:,28] + secMat_b[:,:,56] + secMat_b[:,:,57] + secMat_b[:,:,5] + secMat_b[:,:,13] + secMat_b[:,:,19] + secMat_b[:,:,20] + secMat_b[:,:,21] + secMat_b[:,:,22] + secMat_b[:,:,23] + secMat_b[:,:,24] + secMat_b[:,:,48] + secMat_b[:,:,12] + secMat_b[:,:,41]
    zeolites0 = secMat_b[:,:,6] + secMat_b[:,:,26] + secMat_b[:,:,30] + secMat_b[:,:,34] + secMat_b[:,:,38] + secMat_b[:,:,44] + secMat_b[:,:,39] + secMat_b[:,:,31]
    chlorites0 = secMat_b[:,:,43] + secMat_b[:,:,45] + secMat_b[:,:,46] + secMat_b[:,:,47] + secMat_b[:,:,53] + secMat_b[:,:,54]
    
    smectites0 = secMat_d[:,:,9] + secMat_d[:,:,28] + secMat_d[:,:,56] + secMat_d[:,:,57] + secMat_d[:,:,5] + secMat_d[:,:,13] + secMat_d[:,:,19] + secMat_d[:,:,20] + secMat_d[:,:,21] + secMat_d[:,:,22] + secMat_d[:,:,23] + secMat_d[:,:,24] + secMat_d[:,:,48] + secMat_d[:,:,12] + secMat_d[:,:,41]
    zeolites0 = secMat_d[:,:,6] + secMat_d[:,:,26] + secMat_d[:,:,30] + secMat_d[:,:,34] + secMat_d[:,:,38] + secMat_d[:,:,44] + secMat_d[:,:,39] + secMat_d[:,:,31]
    chlorites0 = secMat_d[:,:,43] + secMat_d[:,:,45] + secMat_d[:,:,46] + secMat_d[:,:,47] + secMat_d[:,:,53] + secMat_d[:,:,54]


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
    print " "
    print " "
    print "step =", i
    
    # # single time slice matrices
    # psi = psi0[:,i*len(x):((i)*len(x)+len(x))]
    # rho = rho0[:,i*len(x):((i)*len(x)+len(x))]
    # # perm = perm0[:,i*len(x):((i)*len(x)+len(x))]
    # temp = temp0[:,i*len(x):((i)*len(x)+len(x))]
    # u = u0[:,i*len(x):((i)*len(x)+len(x))]
    # v = v0[:,i*len(x):((i)*len(x)+len(x))]

    
    if chem == 1:
        for j in range(1,minNum):
            secStep[:,:,j] = cut_chem(secMat[:,:,j],i)
            dsecStep[:,:,j] = cut_chem(dsecMat[:,:,j],i)
        inert = cut_chem(inert0,i)
        dic = cut_chem(dic0,i)
        ca = cut_chem(ca0,i)
        ph = cut_chem(ph0,i)
        #print ph[:,50]
        print ph.shape
        alk = cut_chem(alk0,i)
        mg = cut_chem(mg0,i)
        fe = cut_chem(fe0,i)
        si = cut_chem(si0,i)
        k1 = cut_chem(k0,i)
        na = cut_chem(na0,i)
        al = cut_chem(al0,i)
        glass = cut_chem(glass0,i)
        glass_p = cut_chem(glass0_p,i)
        water = cut_chem(water0,i)
        smectites = cut_chem(smectites0,i)
        zeolites = cut_chem(zeolites0,i)
        chlorites = cut_chem(chlorites0,i)
        alt_vol = cut_chem(alt_vol0,i)
        precip = cut_chem(precip0,i)
        pri_total = cut_chem(pri_total0,i)
        
        for j in range(1,minNum):
            secStep_a[:,:,j] = cut_chem(secMat_a[:,:,j],i)
            dsecStep_a[:,:,j] = cut_chem(dsecMat_a[:,:,j],i)
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
        water_a = cut_chem(water0_a,i)
        smectites_a = cut_chem(smectites0_a,i)
        zeolites_a = cut_chem(zeolites0_a,i)
        chlorites_a = cut_chem(chlorites0_a,i)
        alt_vol_a = cut_chem(alt_vol0_a,i)
        precip_a = cut_chem(precip0_a,i)
        pri_total_a = cut_chem(pri_total0_a,i)
        
        
        for j in range(1,minNum):
            secStep_b[:,:,j] = cut_chem(secMat_b[:,:,j],i)
            dsecStep_b[:,:,j] = cut_chem(dsecMat_b[:,:,j],i)
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
        water_b = cut_chem(water0_b,i)
        smectites_b = cut_chem(smectites0_b,i)
        zeolites_b = cut_chem(zeolites0_b,i)
        chlorites_b = cut_chem(chlorites0_b,i)
        alt_vol_b = cut_chem(alt_vol0_b,i)
        precip_b = cut_chem(precip0_b,i)
        pri_total_b = cut_chem(pri_total0_b,i)
        
        
        for j in range(1,minNum):
            secStep_d[:,:,j] = cut_chem(secMat_d[:,:,j],i)
            dsecStep_d[:,:,j] = cut_chem(dsecMat_d[:,:,j],i)
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
    

      
    # #######################
    # ##    CHEM 0 PLOT    ##
    # #######################
    #
    # fig=plt.figure()
    #
    # # varMat = temp
    # # varStep = temp
    # # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    # # ax1=fig.add_subplot(3,1,1, aspect=asp*2.5,frameon=False)
    # # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    # # p = plt.contour(xg,yg,perm,[-12.0,-13.5],colors='black',linewidths=np.array([1.0]))
    # # CS = plt.contour(xg, yg, psi, 4, colors='black',linewidths=np.array([0.5]))
    # # cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
    # # plt.title('TEMPERATURE [$^{o}$C]')
    # # plt.ylim([np.min(y)/2.0,0.])
    # # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    #
    # chemplot(ca0, ca, 2, 2, 1, 1, '[Ca]',0)
    # chemplot(dic0, dic, 2, 2, 2, 1, '[DIC]',0)
    # chemplot(glass0, glass, 2, 2, 3, 1, 'BASLATIC GLASS [mol]',1)
    # # chemplot(secMat[:,:,14], secStep[:,:,14], 4, 2, 5, 4, 'ANHYDRITE [mol]')
    # chemplot(secMat[:,:,16], secStep[:,:,16], 2, 2, 4, 1, 'CALCITE [mol]',0)
    # # chemplot(secMat[:,:,3], secStep[:,:,3], 4, 2, 7, 4, 'KAOLINITE [mol]')
    # # chemplot(secMat[:,:,5], secStep[:,:,5], 4, 2, 8, 4, 'SAPONITE MG [mol]')
    # #
    # # chemplot(uCoarseMat, uCoarseStep, 4, 2, 6, 4, 'uCoarse')
    # # chemplot(vCoarseMat, vCoarseStep, 4, 2, 7, 4, 'vCoarse')
    # # chemplot(psiCoarseMat, psiCoarseStep, 4, 2, 8, 4, 'psiCoarse')
    #
    # fig.set_tight_layout(True)
    # plt.savefig(outpath+'jdfChem0_'+str(i+restart)+'.png')
    #
    
    
    
    
    # profile_mg = np.zeros(len(y))
    # profile_mg_y = np.zeros(len(y))
    # for j in range(len(y)):
    #     if y[j] > -sed1[j]:
    #         profile_mg[j] = mg[j,(param_w + 3300.0)/bitsx]
    #         profile_mg_y[j] = y[j]
    #         print profile_mg[j]
    #
    # fig=plt.figure()
    # ax1=fig.add_subplot(2,2,1)
    # plt.scatter(profile_mg,profile_mg_y)
    #
    #
    # plt.savefig(outpath+'jdfProfile_'+str(i+restart)+'.png')
    
    

    
    if chem == 1:
        
        #
        # ########################
        # ##    CHEM 0 EXTRA    ##
        # ########################
        #
        # fig=plt.figure(figsize=(13.0,3.5))
        # plt.subplots_adjust( wspace=0.03, bottom=0.15, top=0.97, left=0.01, right=0.99)
        #
        # # # col 1
        # plt_s = alt_vol0
        # plt_a = alt_vol0_a
        # plt_b = alt_vol0_b
        # plt_d = alt_vol0_d
        # plt_ss = alt_vol
        # plt_aa = alt_vol_a
        # plt_bb = alt_vol_b
        # plt_dd = alt_vol_d
        # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        # c_min = np.amin(all_ch[all_ch>0.0])
        # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        # c_max = np.amax(all_ch[all_ch>0.0])
        #
        # chemplot(plt_s, plt_ss, 3, 5, 1, 1, 'alt_vol_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='fractional alt. vol.')
        # chemplot(plt_d, plt_dd, 3, 5, 6, 1, 'alt_vol_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 21, 1, 'alt_vol_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_bb, plt_bb, 3, 10, 22, 1, 'alt_vol_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        #
        #
        # # # col 2
        # plt_s = pri_total0
        # plt_a = pri_total0_a
        # plt_b = pri_total0_b
        # plt_d = pri_total0_d
        # plt_ss = pri_total
        # plt_aa = pri_total_a
        # plt_bb = pri_total_b
        # plt_dd = pri_total_d
        # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        # c_min = np.amin(all_ch[all_ch>0.0])
        # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        # c_max = np.amax(all_ch[all_ch>0.0])
        #
        # chemplot(plt_s, plt_ss, 3, 5, 2, 1, 'basalt remaining s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='basalt remaining')
        # chemplot(plt_d, plt_dd, 3, 5, 7, 1, 'basalt remaining d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 23, 1, 'basalt remaining a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 24, 1, 'basalt remaining b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        #
        #
        #
        # # # col 3
        #
        # plt_s = alk0
        # plt_a = alk0_a
        # plt_b = alk0_b
        # plt_d = alk0_d
        # plt_ss = alk
        # plt_aa = alk_a
        # plt_bb = alk_b
        # plt_dd = alk_d
        # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        # c_min = np.amin(all_ch[all_ch>0.0])
        # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        # c_max = np.amax(all_ch[all_ch>0.0])
        #
        # chemplot(plt_s, plt_ss, 3, 5, 3, 1, 'alk_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='alkalinity')
        # chemplot(plt_d, plt_dd, 3, 5, 8, 1, 'alk_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 25, 1, 'alk_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 26, 1, 'alk_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        #
        #
        #
        # # # col 4
        #
        # plt_s = ph0
        # plt_a = ph0_a
        # plt_b = ph0_b
        # plt_d = ph0_d
        # plt_ss = ph
        # plt_aa = ph_a
        # plt_bb = ph_b
        # plt_dd = ph_d
        # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        # c_min = np.amin(all_ch[all_ch>0.0])
        # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        # c_max = np.amax(all_ch[all_ch>0.0])
        # chemplot(plt_s, plt_ss, 3, 5, 4, 1, 'ph_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='fluid pH')
        # chemplot(plt_d, plt_dd, 3, 5, 9, 1, 'ph_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 27, 1, 'ph_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 28, 1, 'ph_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        #
        #
        #
        # # # col 5
        #
        # plt_s = dic0
        # plt_a = dic0_a
        # plt_b = dic0_b
        # plt_d = dic0_d
        # plt_ss = dic
        # plt_aa = dic_a
        # plt_bb = dic_b
        # plt_dd = dic_d
        # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        # c_min = np.amin(all_ch[all_ch>0.0])
        # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        # c_max = np.amax(all_ch[all_ch>0.0])
        # chemplot(plt_s, plt_ss, 3, 5, 5, 1, 'dic_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='DIC')
        # chemplot(plt_s, plt_dd, 3, 5, 10, 1, 'dic_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 29, 1, 'dic_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 30, 1, 'dic_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        #
        #
        #
        #
        #
        #
        # #fig.set_tight_layout(True)
        # # plt.subplots_adjust( wspace=0.05 , bottom=0.04, top=0.97, left=0.03, right=0.975)
        # plt.savefig(outpath+'jdfChem0_'+str(i+restart)+'.png')
        #
        
        
        
        ######################
        ##    CHEM 0 SOL    ##
        ######################
    
        fig=plt.figure(figsize=(13.0,3.5))
        plt.subplots_adjust( wspace=0.03, bottom=0.15, top=0.97, left=0.01, right=0.99)

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
        
        chemplot(plt_s, plt_ss, 3, 5, 1, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Ca] concentration')
        chemplot(plt_d, plt_dd, 3, 5, 6, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_a, plt_aa, 3, 10, 21, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        chemplot(plt_bb, plt_bb, 3, 10, 22, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        
        #
        # # # col 2
        # plt_s = mg0
        # plt_a = mg0_a
        # plt_b = mg0_b
        # plt_d = mg0_d
        # plt_ss = mg
        # plt_aa = mg_a
        # plt_bb = mg_b
        # plt_dd = mg_d
        # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        # c_min = np.amin(all_ch[all_ch>0.0])
        # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        # c_max = np.amax(all_ch[all_ch>0.0])
        #
        # chemplot(plt_s, plt_ss, 3, 5, 2, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[Mg] concentration')
        # chemplot(plt_d, plt_dd, 3, 5, 7, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 23, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 24, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        #
        #
        #
        # # # col 3
        #
        # plt_s = k0
        # plt_a = k0_a
        # plt_b = k0_b
        # plt_d = k0_d
        # plt_ss = k1
        # plt_aa = k1_a
        # plt_bb = k1_b
        # plt_dd = k1_d
        # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        # c_min = np.amin(all_ch[all_ch>0.0])
        # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        # c_max = np.amax(all_ch[all_ch>0.0])
        #
        # chemplot(plt_s, plt_ss, 3, 5, 3, 1, 's', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='[K] concentration')
        # chemplot(plt_d, plt_dd, 3, 5, 8, 1, 'd', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 25, 1, 'a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 26, 1, 'b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        #
        #
        #
        # # # col 4
        #
        # plt_s = ph0
        # plt_a = ph0_a
        # plt_b = ph0_b
        # plt_d = ph0_d
        # plt_ss = ph
        # plt_aa = ph_a
        # plt_bb = ph_b
        # plt_dd = ph_d
        # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        # c_min = np.amin(all_ch[all_ch>0.0])
        # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        # c_max = np.amax(all_ch[all_ch>0.0])
        # chemplot(plt_s, plt_ss, 3, 5, 4, 1, 'ph_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='pH')
        # chemplot(plt_d, plt_dd, 3, 5, 9, 1, 'ph_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 27, 1, 'ph_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 28, 1, 'ph_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        #
        #
        #
        # # # col 5
        #
        # plt_s = dic0
        # plt_a = dic0_a
        # plt_b = dic0_b
        # plt_d = dic0_d
        # plt_ss = dic
        # plt_aa = dic_a
        # plt_bb = dic_b
        # plt_dd = dic_d
        # all_ch = [np.min(plt_s[plt_s>0.0]), np.min(plt_a[plt_a>0.0]), np.min(plt_b[plt_b>0.0]), np.min(plt_d[plt_d>0.0])]
        # c_min = np.amin(all_ch[all_ch>0.0])
        # all_ch = [np.max(plt_s[plt_s>0.0]), np.max(plt_a[plt_a>0.0]), np.max(plt_b[plt_b>0.0]), np.max(plt_d[plt_d>0.0])]
        # c_max = np.amax(all_ch[all_ch>0.0])
        # chemplot(plt_s, plt_ss, 3, 5, 5, 1, 'dic_s', xtix=0, ytix=0, cb=1, cb_min=c_min, cb_max=c_max, cb_title='DIC')
        # chemplot(plt_s, plt_dd, 3, 5, 10, 1, 'dic_d', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_a, plt_aa, 3, 10, 29, 1, 'dic_a', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        # chemplot(plt_b, plt_bb, 3, 10, 30, 1, 'dic_b', xtix=0, ytix=0, cb=0, cb_min=c_min, cb_max=c_max)
        #
        #

        plt.savefig(outpath+'jdfSol0_'+str(i+restart)+'.png')
        
        
        
        
        
        
        
        
        

    
        ######################
        #    CHEM 1 PLOT    ##
        ######################

        fig=plt.figure(figsize=(11.0,4.25))
        
        # if np.max(dsecStep[:,:,16]) != 0.0:
        #     chemcont(secMat[:,:,16], secStep[:,:,16], 3, 3, 1, 1, 'mineral distribution in solo chamber', xtix=0, ytix=1,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        # if np.max(dsecStep[:,:,46]) != 0.0:
        #     chemcont(dsecMat[:,:,46], dsecStep[:,:,46], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        # if np.max(dsecStep[:,:,5]) != 0.0:
        #     chemcont(dsecMat[:,:,5], dsecStep[:,:,5], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        # if np.max(dsecStep[:,:,10]) != 0.0:
        #     chemcont(dsecMat[:,:,10], dsecStep[:,:,10], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        # if np.max(dsecStep[:,:,8]) != 0.0:
        #     chemcont(dsecMat[:,:,8], dsecStep[:,:,8], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='O')
        # if np.max(dsecStep[:,:,6]) != 0.0:
        #     chemcont(dsecMat[:,:,6], dsecStep[:,:,6], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=0)
        
        #

        chemcont(secMat[:,:,16], secStep[:,:,16], 3, 3, 1, 1, 'mineral distribution in solo chamber', xtix=0, ytix=1,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        chemcont(secMat[:,:,46], secStep[:,:,46], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        chemcont(secMat[:,:,5], secStep[:,:,5], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        chemcont(secMat[:,:,10], secStep[:,:,10], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        chemcont(secMat[:,:,8], secStep[:,:,8], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='O')
        chemcont(secMat[:,:,6], secStep[:,:,6], 3, 3, 1, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=0)
        
        
        chemcont(secMat_a[:,:,16]+secMat_b[:,:,16], secStep_a[:,:,16]+secStep_b[:,:,16], 3, 3, 4, 1, 'dual chamber (a+b total)', xtix=1, ytix=1,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        chemcont(secMat_a[:,:,46]+secMat_b[:,:,46], secStep_a[:,:,46]+secStep_b[:,:,46], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        chemcont(secMat_b[:,:,40], secStep_b[:,:,40], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='gold',to_hatch=0)
        chemcont(secMat_a[:,:,5]+secMat_b[:,:,5], secStep_a[:,:,5]+secStep_b[:,:,5], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        chemcont(secMat_a[:,:,10]+secMat_b[:,:,10], secStep_a[:,:,10]+secStep_b[:,:,10], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        chemcont(secMat_a[:,:,8]+secMat_b[:,:,8], secStep_a[:,:,8]+secStep_b[:,:,8], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='O')
        chemcont(secMat_a[:,:,6]+secMat_b[:,:,6], secStep_a[:,:,6]+secStep_b[:,:,6], 3, 3, 4, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=0)
        
        
        chemcont(secMat_a[:,:,16], secStep_a[:,:,16], 3, 6, 13, 1, 'chamber a only', xtix=0, ytix=1,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        chemcont(secMat_a[:,:,46], secStep_a[:,:,46], 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        chemcont(secMat_a[:,:,5], secStep_a[:,:,5], 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        chemcont(secMat_a[:,:,10], secStep_a[:,:,10], 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        chemcont(secMat_a[:,:,8], secStep_a[:,:,8], 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='O')
        chemcont(secMat_a[:,:,6], secStep_a[:,:,6], 3, 6, 13, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=0)
        
        chemcont(secMat_b[:,:,16], secStep_b[:,:,16], 3, 6, 14, 1, 'chamber b only', xtix=0, ytix=0,perm_lines=0, frame_lines=1, min_color='r',to_hatch=0)
        chemcont(secMat_b[:,:,46], secStep_b[:,:,46], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='g',to_hatch=0)
        chemcont(secMat_b[:,:,40], secStep_b[:,:,40], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='gold',to_hatch=0)
        chemcont(secMat_b[:,:,5], secStep_b[:,:,5], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=1,hatching='////')
        chemcont(secMat_b[:,:,10], secStep_b[:,:,10], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='\\')
        chemcont(secMat_b[:,:,8], secStep_b[:,:,8], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='k',to_hatch=1,hatching='O')
        chemcont(secMat_b[:,:,6], secStep_b[:,:,6], 3, 6, 14, 1, '', xtix=0, ytix=0,perm_lines=0, frame_lines=0, min_color='b',to_hatch=0)
        
        

        lg1 = Patch(facecolor='r', label='calcite', alpha=0.5)
        lg2 = Patch(facecolor='g', label='chlor', alpha=0.5)
        lg3 = Patch(facecolor='gold', label='talc', alpha=0.5)
        lg4 = Patch(facecolor='w', label='sap mg', hatch ='////')
        lg5 = Patch(facecolor='w', label='goethite', hatch ='\\')
        lg6 = Patch(facecolor='w', label='pyrite', hatch ='O')
        lg7 = Patch(facecolor='b', label='celadonite', alpha=0.5)
        
        plt.legend([lg1,lg2,lg3,lg4,lg5,lg6,lg7],['calcite','chlor','talc','sap mg', 'goethite', 'pyrite','celadonite'],fontsize=10,ncol=3,bbox_to_anchor=(0.0, -2.0),loc=8)
        
        
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
        
        
        
        
        
        #######################
        ##     FLUID COMP    ##
        #######################
    
        # fig=plt.figure()
        #
        # chemplot(ca0, ca, 3, 2, 1, 1, '[Ca]',1, 0, 0)
        # chemplot(mg0, mg, 3, 2, 2, 1, '[Mg]',1, 0, 0)
        # chemplot(na0, na, 3, 2, 3, 1, '[Na]',1, 0, 0)
        # chemplot(k0, k1, 3, 2, 4, 1, '[K]',1, 0, 0)
        # chemplot(al0, al, 3, 2, 5, 1, '[Al]',1, 0, 0)
        # chemplot(si0, si, 3, 2, 6, 1, '[Si]',1, 0, 0)
        # # chemplot(fe0, fe, 3, 3, 7, 1, '[Fe]',1, 0, 0)
        #
        #
        # fig.set_tight_layout(True)
        # plt.savefig(outpath+'jdfChemFluid_'+str(i+restart)+'.png')
        #
        
        

    
    
    
        #######################
        ##     MISC PLOT     ##
        #######################
        
        
        
    

                
    
            
        # fig=plt.figure()
        #
        # chemplot(alt_vol0, alt_vol, 3, 2, 1, 1, 'alteraction fraction',0)
        # chemplot(water, water, 3, 2, 2, 1, 'water',1)
        # chemplot(inert, inert, 3, 2, 3, 1, 'inert',0)
        #
        # plt.savefig(outpath+'jdfChem_misc_'+str(i+restart)+'.png')
        #
        #
        #
        
        
        #######################
        ##    GROUP PLOT     ##
        #######################
        

        

        #
        #
        # fig=plt.figure()
        #
        # chemplot(glass0_p, glass_p, 3, 2, 1, 1, 'Fractional volume of primary basalt',1)
        #
        # chemplot(alt_vol0, alt_vol, 3, 2, 2, 1, 'Total secondary mineral fractional volume',0)
        #
        # chemplot(smectites0, smectites, 3, 2, 3, 1, 'Fractional volume of smectite minerals',0)
        #
        # #chemplot(zeolites0, zeolites, 3, 2, 4, 1, 'Fractional volume of zeolite minerals',0)
        #
        # #chemplot(chlorites0, chlorites, 3, 2, 5, 1, 'Fractional volume of chlorite minerals',0)
        #
        # chemplot(alk_flux, alk_flux, 3, 2, 6, 1, 'Alkalinity flux to ocean [eq/L/yr]',0)
        #
        #
        # fig.set_tight_layout(True)
        # plt.savefig(outpath+'jdfGroup_'+str(i+restart)+'.png')
        
        
        
        #######################
        ##    GROUP PLOT     ##
        #######################
        

        

        #
        #
        # fig=plt.figure()
        #
        # chemplot(glass0_p, glass_p, 3, 1, 1, 1, 'Fractional volume of primary basalt',1 , 0, 0)
        #
        # chemplot(alt_vol0, alt_vol, 3, 1, 2, 1, 'Total secondary mineral fractional volume',0 , 0, 0)
        #
        # chemplot(alk_flux, alk_flux, 3, 1, 3, 1, 'Alkalinity flux to ocean [eq/L/yr]',0 , 0, 0)
        #
        #
        # fig.set_tight_layout(True)
        # plt.savefig(outpath+'jdfVis_'+str(i+restart)+'.png')
        
        
    
    
        ###########################
        ##    SECONDARY PLOTZ    ##
        ###########################
    
        # print " chem2: " ,
        # fig=plt.figure()
        #
        # sp = 0
        # j_last = 0
        # for j in range(0,minNum):
        #     if np.max(secMat[:,:,j]) > 0.0 and sp < 6:
        #         sp = sp + 1
        #         j_last = j
        #         print j_last ,
        #         chemplot(secMat[:,:,j], secStep[:,:,j], 3, 2, sp, 1, secondary[j] + ' [cm$^3$]',0 , 0, 0)
        #
        #
        # fig.set_tight_layout(True)
        # plt.savefig(outpath+'jdfChem2_'+str(i+restart)+'.png')
    
    
    
    
    
    
    
        # print " "
        # print " chem3: " ,
        # fig=plt.figure()
        #
        # sp = 0
        # for j in range(j_last+1,minNum):
        #     if np.max(secMat[:,:,j]) > 0.0 and sp < 6:
        #         sp = sp + 1
        #         j_last = j
        #         print j_last ,
        #         chemplot(secMat[:,:,j], secStep[:,:,j], 3, 2, sp, 1, secondary[j] + ' [cm$^3$]',0 , 0, 0)
        #
        #
        # fig.set_tight_layout(True)
        # plt.savefig(outpath+'jdfChem3_'+str(i+restart)+'.png')
        #
        # print " "
        # print " chem4: " ,
        # fig=plt.figure()
        #
        # sp = 0
        # for j in range(j_last+1,minNum):
        #     if np.max(secMat[:,:,j]) > 0.0 and sp < 6:
        #         sp = sp + 1
        #         j_last = j
        #         print j_last ,
        #         chemplot(secMat[:,:,j], secStep[:,:,j], 3, 2, sp, 1, secondary[j] + ' [cm$^3$]',0 , 0, 0)
        #
        #
        # fig.set_tight_layout(True)
        # plt.savefig(outpath+'jdfChem4_'+str(i+restart)+'.png')
    
    
    plt.close('all')
    

    
    #
    #
    # #######################
    # ##    CHEM 2 PLOT    ##
    # #######################
    #
    # fig=plt.figure()
    #
    # varMat = temp
    # varStep = temp
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    # ax1=fig.add_subplot(4,2,1, aspect=asp/1.0,frameon=False)
    # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    # p = plt.contour(xg,yg,perm,[-12.0,-13.5],colors='black',linewidths=np.array([1.0]))
    # cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
    # plt.title('TEMPERATURE [$^{o}$C]')
    # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    #
    # chemplot(secMat[:,:,27], secStep[:,:,27], 4, 2, 2, 4, 'HEMATITE [mol]',0)
    # chemplot(secMat[:,:,31], secStep[:,:,31], 4, 2, 3, 4, 'PHILLIPSITE [mol]',0)
    # chemplot(secMat[:,:,37], secStep[:,:,37], 4, 2, 4, 4, 'VERMICULITE MG [mol]',0)
    # chemplot(secMat[:,:,40], secStep[:,:,40], 4, 2, 5, 4, 'TALC [mol]',0)
    # chemplot(secMat[:,:,46], secStep[:,:,46], 4, 2, 6, 4, 'CLINOCHLORE 14a [mol]',0)
    # chemplot(secMat[:,:,57], secStep[:,:,57], 4, 2, 7, 4, 'MONTMORILLONITE MG [mol]',0)
    # #chemplot(secMat[:,:,26], secStep[:,:,26], 4, 2, 8, 4, 'MESOLITE [mol]')
    #
    #
    # fig.set_tight_layout(True)
    # plt.savefig(outpath+'jdfChem2_'+str(i+restart)+'.png')
    #
 
 
