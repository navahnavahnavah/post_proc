# revived_JDF.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import streamplot as sp
import multiplot_data as mpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams['axes.titlesize'] = 12

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

print secondary.shape
print density.shape
print molar.shape

##############
# INITIALIZE #
##############

#steps = 400
steps = 10
minNum = 57
ison=10000
trace = 0
chem = 1
iso = 0

#outpath = "output/revival/eggs_c/o600w1000h200orhs800/"
#outpath = "output/revival/eggs_a/o300w1000h200orhs700/"
outpath = "output/revival/win_geochem/2f/"
#outpath = "output/revival/may16/o250orhs500w1000wrhs1000h100/"

#outpath = "output/revival/may16/id_h100_orhs500/w1000wrhs1000/"
path = outpath
param_w = 300.0
param_w_rhs = 200.0


# load output
x0 = np.loadtxt(path + 'x.txt',delimiter='\n')
y0 = np.loadtxt(path + 'y.txt',delimiter='\n')

fig=plt.figure()
plt.plot(x0)
fig.savefig(outpath+'x.png')

# format plotting geometry
x=x0
y=y0

asp = np.abs(np.max(x)/np.min(y))/4.0
print asp
bitsx = len(x)
bitsy = len(y)
restart = 1


dx = float(np.max(x))/float(bitsx)
dy = np.abs(float(np.max(np.abs(y)))/float(bitsy))

xCell = x0
yCell = y0


xg, yg = np.meshgrid(x[:],y[:])

mask = np.loadtxt(path + 'mask.txt')
maskP = np.loadtxt(path + 'maskP.txt')
psi0 = np.loadtxt(path + 'psiMat.txt')
rho0 = np.loadtxt(path + 'rhoMat.txt')
# perm0 = np.loadtxt(path + 'permMat.txt')
perm = np.loadtxt(path + 'permeability.txt')
# perm_kx = np.loadtxt(path + 'perm_kx.txt')
# perm_ky = np.loadtxt(path + 'perm_ky.txt')

perm = np.log10(perm)

temp0 = np.loadtxt(path + 'hMat.txt')
temp0 = temp0 - 273.0
u0 = np.loadtxt(path + 'uMat.txt')
v0 = np.loadtxt(path + 'vMat.txt')
lambdaMat = np.loadtxt(path + 'lambdaMat.txt')

u_ts = np.zeros([steps])

lam = np.loadtxt(path + 'lambdaMat.txt')

if trace == 1:
    iso_trace0 = np.loadtxt(path + 'isopart_1.txt')
    inert_trace0 = np.loadtxt(path + 'inert_trace.txt')

if iso == 1:
    c140 = np.loadtxt(path + 'iso_c14.txt')
    control0 = np.loadtxt(path + 'iso_control.txt')


def cut(geo0,index):
    #geo_cut = geo0[(index*len(y0)/cell):(index*len(y0)/cell+len(y0)/cell),:]
    geo_cut = geo0[:,(index*len(x0)):(index*len(x0)+len(x0))]
    geo_cut = np.append(geo_cut, geo_cut[-1:,:], axis=0)
    geo_cut = np.append(geo_cut, geo_cut[:,-1:], axis=1)
    return geo_cut
    
def cut_chem(geo0,index):
    geo_cut_chem = geo0[:,(index*len(x0)):(index*len(x0)+len(x0))]
    return geo_cut_chem

def chemplot(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, ditch):
    # print "check"
    # print xCell.shape
    # print yCell.shape
    # print varStep.shape
    if ditch==0:
        contours = np.linspace(np.min(varMat),np.max(varMat),5)
    if ditch==1:
        contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),5)
        # varStep[varStep == 0.0] = None
        #contours = np.linspace(4.8,np.max(varStep),20)
    if ditch==2:
        contours = np.linspace(np.min(varMat),np.max(varMat[varMat<varStep[bitsy-25,bitsx/2]])/5.0,20)
        #contours = np.linspace(contours[0],contours[-1],20)
    ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp,frameon=False)
    #pGlass = plt.contourf(xCell,yCell,varStep,contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0,antialiased=True)
    if ditch==0:
        pGlass = plt.pcolor(xCell,yCell,varStep,cmap=cm.rainbow,vmin=np.min(contours), vmax=np.max(contours))
    if ditch==1:
        pGlass = plt.pcolor(xCell,yCell,varStep,cmap=cm.rainbow,vmin=np.min(contours), vmax=np.max(contours))
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([0.5]))
    plt.xticks([])
    plt.yticks([])
    plt.ylim([np.min(y0),0.])
    #cMask = plt.contourf(xg,yg,maskP,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
    plt.title(cp_title,fontsize=10)
    cbar = plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::contour_interval])
    fig.set_tight_layout(True)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")
    pGlass.set_edgecolor("face")
    return chemplot


secMat = np.zeros([bitsy,bitsx*steps,minNum+1])
#satMat = np.zeros([(bitsCy-1),(bitsCx-1)*steps,minNum+1])

secStep = np.zeros([bitsy,bitsx,minNum+1])
dsecStep = np.zeros([bitsy,bitsx,minNum+1])
#satStep = np.zeros([bitsCy,bitsCx,minNum+1])


if chem == 1:
    # IMPORT MINERALS
    for j in range(1,minNum):
        if j % 5 ==0:
            print 'loading minerals', str(j-5), "-", str(j)
        secMat[:,:,j] = np.loadtxt(path + 'z_sec' + str(j) + '.txt')
        secMat[:,:,j] = secMat[:,:,j]*molar[j]/density[j]
        #satMat[:,:,j] = np.loadtxt(path + 'sat' + str(j) + '.txt')
    inert0 = np.loadtxt(path + 'z_sol_inert.txt')
    dic0 = np.loadtxt(path + 'z_sol_c.txt')
    ca0 = np.loadtxt(path + 'z_sol_ca.txt')
    mg0 = np.loadtxt(path + 'z_sol_mg.txt')
    na0 = np.loadtxt(path + 'z_sol_na.txt')
    k0 = np.loadtxt(path + 'z_sol_k.txt')
    fe0 = np.loadtxt(path + 'z_sol_fe.txt')
    si0 = np.loadtxt(path + 'z_sol_si.txt')
    al0 = np.loadtxt(path + 'z_sol_al.txt')
    ph0 = np.loadtxt(path + 'z_sol_ph.txt')
    alk0 = np.loadtxt(path + 'z_sol_alk.txt')
    glass0 = np.loadtxt(path + 'z_pri_glass.txt')
    glass0 = glass0*110.3839/(2.7)
    glass0_p = glass0/(np.max(glass0))
    water0 = np.loadtxt(path + 'z_med_v_water.txt')
    print np.min(glass0>0.0)
    print np.max(glass0)


smectites0 = np.zeros(secMat[:,:,1].shape)
zeolites0 = np.zeros(secMat[:,:,1].shape)
chlorites0 = np.zeros(secMat[:,:,1].shape)
alt_vol0 = np.zeros(secMat[:,:,1].shape)


smectites0 = secMat[:,:,9] + secMat[:,:,28] + secMat[:,:,56] + secMat[:,:,57] + secMat[:,:,5] + secMat[:,:,13] + secMat[:,:,19] + secMat[:,:,20] + secMat[:,:,21] + secMat[:,:,22] + secMat[:,:,23] + secMat[:,:,24] + secMat[:,:,48] + secMat[:,:,12] + secMat[:,:,41]
zeolites0 = secMat[:,:,6] + secMat[:,:,26] + secMat[:,:,30] + secMat[:,:,34] + secMat[:,:,38] + secMat[:,:,44] + secMat[:,:,39] + secMat[:,:,31]
chlorites0 = secMat[:,:,43] + secMat[:,:,45] + secMat[:,:,46] + secMat[:,:,47] + secMat[:,:,53] + secMat[:,:,54]


for j in range(len(x)*steps):
    for jj in range(len(y)):
        if glass0[jj,j] > 0.0:
            alt_vol0[jj,j] = np.sum(secMat[jj,j,:])
        if glass0[jj,j] + smectites0[jj,j] > 0.0:
            smectites0[jj,j] = smectites0[jj,j] / (glass0[jj,j] + smectites0[jj,j])
        if glass0[jj,j] + zeolites0[jj,j] > 0.0:
            zeolites0[jj,j] = zeolites0[jj,j] / (glass0[jj,j] + zeolites0[jj,j])
        if glass0[jj,j] + chlorites0[jj,j] > 0.0:
            chlorites0[jj,j] = chlorites0[jj,j] / (glass0[jj,j] + chlorites0[jj,j])
        if glass0[jj,j] + alt_vol0[jj,j] > 0.0:
            alt_vol0[jj,j] = alt_vol0[jj,j] / (glass0[jj,j] + alt_vol0[jj,j])


delta = np.zeros(lambdaMat.shape)
    
    
for i in range(0,steps,1): 
    print " "
    print " "
    print "step =", i
    
    # single time slice matrices
    psi = psi0[:,i*len(x):((i)*len(x)+len(x))]
    rho = rho0[:,i*len(x):((i)*len(x)+len(x))]
    # perm = perm0[:,i*len(x):((i)*len(x)+len(x))]
    temp = temp0[:,i*len(x):((i)*len(x)+len(x))]
    u = u0[:,i*len(x):((i)*len(x)+len(x))]
    v = v0[:,i*len(x):((i)*len(x)+len(x))]

    if iso == 1:
        c14 = c140[:,i*len(x):((i)*len(x)+len(x))]
        control = control0[:,i*len(x):((i)*len(x)+len(x))]
        print c14.shape
        print control.shape
    
    if chem == 1:
        for j in range(1,minNum):
            secStep[:,:,j] = cut_chem(secMat[:,:,j],i)
            #satStep[:,:,j] = cut(satMat[:,:,j],i)
            #dsecStep[:,:,j] = cut(secMat[:,:,j],i)# - cut(secMat[:,:,j],i-1)
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
        water = cut_chem(water0,i)
        print np.max(glass)
        print np.min(glass)
        smectites = cut_chem(smectites0,i)
        zeolites = cut_chem(zeolites0,i)
        chlorites = cut_chem(chlorites0,i)
        alt_vol = cut_chem(alt_vol0,i)
        # print np.min(glass[glass>0.0])

    alk_flux = np.zeros(secStep[:,:,1].shape)
    
    for j in range(len(x)):
        for jj in range(len(y)):
            if secStep[jj,j,16] > 0.0 and water[jj,j] > 0.0:
                alk_flux[jj,j] = (secStep[jj,j,16]*density[16]/molar[16] * 2.0) / (water[jj,j] * 10000.0 * (i+1.0) )
            if secStep[jj,j,16] == 0.0 and water[jj,j] > 0.0:
                alk_flux[jj,j] = (alk[jj,j] - alk[1,1]) / (water[jj,j] * 1000.0)
    print "alk_flux sum" , np.sum(alk_flux)

    
    
    
    if iso == 1:
    
        #####################################
        #        FINITE DIFF ISO PLOT       #
        #####################################

        fig=plt.figure()

    
        # c14
        varStep0 = c14
        varStep = np.zeros(c14.shape)
        varStep = c14
        
        year_max = .8

        for jj in range(len(x)-1):
            for j in range(len(y)-1):
                    if varStep0[j,jj] > year_max:
                        varStep[j,jj] = 5730.0*np.log(varStep0[j,jj])/(-.693)
                    if varStep0[j,jj] <= year_max:
                        varStep[j,jj] = 5730.0*np.log(year_max)/(-.693)

    
        iso_col_mean = np.zeros(len(x))
        iso_col_top = np.zeros(len(x))
        varStepMask = np.zeros(varStep.shape)
        for jj in range(len(x)-1):
            for j in range(len(y)-1):
                if perm[j,jj] > -13 and maskP[j,jj] ==1.0:
                    varStepMask[j,jj] = varStep[j,jj]
                # if varStepMask[j,jj] >= 50000.0:
                #     varStepMask[j,jj] = 0.0
        #varStepMask = varStep
                
        for jj in range(len(x)-1):
            iso_col_mean[jj] = np.sum(varStepMask[:,jj])/np.count_nonzero(varStepMask[:,jj])
            for j in range(len(y)-2):
                if perm[j,jj] > -13.0 and perm[j+1,jj] <= -15.0:
                    iso_col_top[jj] = varStepMask[j,jj]
                
        contours = np.linspace(np.min(varStep),np.max(varStep),10)
        
        print "14c mean" , np.mean(varStep)
    
        ax = fig.add_axes([0.1, 0.4, .8, .3])
        pGlass=plt.pcolor(x,y,varStep,cmap=cm.rainbow)
        #pGlass=plt.contourf(x,y,varStepMask,40,cmap=cm.rainbow)
        cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
        p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([0.5]))
        plt.plot([param_w + 3300.0, param_w + 3300.0], [np.min(y), np.max(y)],'k',lw=2)
        plt.plot([param_w + 6700.0, param_w + 6700.0], [np.min(y), np.max(y)],'k',lw=2)
        plt.plot([param_w + 14600.0, param_w + 14600.0], [np.min(y), np.max(y)],'k',lw=2)
        plt.plot([param_w + 21000.0, param_w + 21000.0], [np.min(y), np.max(y)],'k',lw=2)
    
        cbar= plt.colorbar(pGlass, orientation='vertical')
        ax.axes.get_xaxis().set_ticks([])
        plt.ylim([np.min(y),0.0])
        cbar.ax.set_ylabel('carbon 14 age [years]')
    
    
        # control no decay
        varStep0 = control
        varStep = np.zeros(control.shape)
        varStep = control


        for jj in range(len(x)-1):
            for j in range(len(y)-1):
                    if varStep0[j,jj] > year_max:
                        varStep[j,jj] = 5730.0*np.log(varStep0[j,jj])/(-.693)
                    if varStep0[j,jj] <= year_max:
                        varStep[j,jj] = 5730.0*np.log(year_max)/(-.693)
                        
        print "control mean" , np.mean(varStep)

        iso_col_mean2 = np.zeros(len(x))
        iso_col_top2 = np.zeros(len(x))
        #varStepMask = varStep
        varStepMask = np.zeros(varStep.shape)
        for jj in range(len(x)-1):
            for j in range(len(y)-1):
                if perm[j,jj] > -13:
                    varStepMask[j,jj] = varStep[j,jj]
                # if varStepMask[j,jj] >= 50000.0:
                #     varStepMask[j,jj] = 0.0
        #varStepMask = varStep
    
                
        contours = np.linspace(np.min(varStep),np.max(varStep),10)
    
        ax = fig.add_axes([0.1, 0.08, .8, .3])
        pGlass=plt.pcolor(x,y,varStep,cmap=cm.rainbow)
        #pGlass=plt.contourf(x,y,varStepMask,40,cmap=cm.rainbow)
        cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
        p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([0.5]))
        plt.plot([param_w + 3300.0, param_w + 3300.0], [np.min(y), np.max(y)],'k',lw=2)
        plt.plot([param_w + 6700.0, param_w + 6700.0], [np.min(y), np.max(y)],'k',lw=2)
        plt.plot([param_w + 14600.0, param_w + 14600.0], [np.min(y), np.max(y)],'k',lw=2)
        plt.plot([param_w + 21000.0, param_w + 21000.0], [np.min(y), np.max(y)],'k',lw=2)
    
        cbar= plt.colorbar(pGlass, orientation='vertical')
        plt.ylim([np.min(y),0.0])
        cbar.ax.set_ylabel('no decay [years]')

    

        for jj in range(len(x)-1):
            iso_col_mean2[jj] = np.sum(varStepMask[:,jj])/(2.0*np.count_nonzero(varStepMask[:,jj]))
            for j in range(len(y)-2):
                if perm[j,jj] > -12.0 and perm[j+1,jj] <= -15.0:
                    iso_col_top2[jj] = varStepMask[j,jj]
                

    
        # top plot
        ax = fig.add_axes([0.1, 0.72, .64, .15],frameon=True)
        # plt.plot(x,iso_col_mean,c='g',label='aquifer column mean')
        plt.plot(x[10:-12],iso_col_mean[10:-12]-iso_col_mean2[10:-12],c='b',label='mean of aquifer (decay)',lw=1)
        plt.plot(x,iso_col_top,c='c',label='top of aquifer (decay)',lw=1)
        plt.scatter([param_w + 3300.0, param_w + 6700.0, param_w + 14600.0, param_w + 21000.0, param_w + 21000.0], [1000.0, 6210.0, 9930.0, 7720.0, 7810.0],c='gold',s=60,edgecolor='k',label='data',zorder=10)
        ax.axes.get_xaxis().set_visible(False)
        plt.xlim([0.0,np.max(x)])
        plt.ylim([-1000.0,80000.0])
        ax.grid(True)
        ax.axes.get_xaxis().set_ticks([])
        plt.legend(bbox_to_anchor=(0.5, 1.5), loc='upper center',fontsize=9,ncol=3)
        ax.yaxis.set_ticks_position('left')




        plt.savefig(outpath+'jdf_iso'+str(i+restart)+'.png')



        fig=plt.figure()
        ax1=fig.add_subplot(1,1,1)
    
    
        # plt.plot(x,iso_col_mean,c='g',label='aquifer column mean')
        plt.plot(x[10:-12],iso_col_mean[10:-12]-iso_col_mean2[10:-12],c='b',label='mean of aquifer (decay)',lw=2)
        plt.plot(x[10:-12],iso_col_top[10:-12],c='c',label='top of aquifer (decay)',lw=2)
        plt.scatter([param_w + 3300.0, param_w + 6700.0, param_w + 14600.0, param_w + 21000.0, param_w + 21000.0], [1000.0, 6210.0, 9930.0, 7720.0, 7810.0],c='gold',s=60,edgecolor='k',label='data',zorder=10)
        ax.axes.get_xaxis().set_visible(False)
        plt.xlim([0.0,np.max(x)])
        #plt.ylim([-1000.0,20000.0])
        ax.grid(True)
        ax.axes.get_xaxis().set_ticks([])
        plt.legend(bbox_to_anchor=(0.5, 1.4), loc='upper center',fontsize=9,ncol=3)
        ax.yaxis.set_ticks_position('left')
        plt.legend()

        plt.savefig(outpath+'jdf_iso_real'+str(i+restart)+'.png')



    ##########################
    #        U_TS PLOT       #
    ##########################

    cap1 = int((param_w/50.0)) + 4
    cap2 = int((param_w_rhs/50.0)) + 4
    
        
    colMax = np.zeros(len(x))
    for n in range(cap1,len(x)-cap2):
        cmax = np.max(u[:,n])*(3.14e7)
        cmin = np.min(u[:,n])*(3.14e7)
        if np.abs(cmax) > np.abs(cmin):
            colMax[n] = cmax
        if np.abs(cmax) < np.abs(cmin):
            colMax[n] = cmin
        #colMax[n] = cmax
        #print colMax[n]
    colMean = np.sum(colMax)/len(x[cap1:-cap2])
    print u_ts.shape
    u_ts[i] = colMean


    


    
        
    ##########################
    #        HEAT PLOT       #
    ##########################

    fig=plt.figure()
    

    # temp plot
    varStep = temp 
    varMat = varStep
    contours = np.linspace(np.min(varMat),np.max(varMat),30)
              
    ax1=fig.add_subplot(2,1,2, aspect=asp,frameon=False)
    pGlass = plt.contourf(x, y, varStep, contours, cmap=cm.rainbow, alpha=1.0,color='#444444',antialiased=True)
    p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    CS = plt.contour(xg, yg, psi, 8, colors='black',linewidths=np.array([0.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    
    cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::2])
    cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
    


    # v velocity plot
    varMat = v*(3.14e7)#dic0
    varStep = v*(3.14e7)#dic
    contours = np.linspace(np.min(varMat),np.max(varMat),10)
    
    ax1=fig.add_subplot(2,2,2, aspect=asp,frameon=False)
    pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    
    cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    cbar.ax.set_xlabel('v [m/yr]')


    # u velocity plot
    varMat = u*(3.14e7)#ca0
    varStep = u*(3.14e7)#ca
    contours = np.linspace(np.min(varMat),np.max(varMat),10)
    
    ax1=fig.add_subplot(2,2,1, aspect=asp,frameon=False)
    pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    
    
    cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    cbar.ax.set_xlabel('u [m/yr]')
    
    

    plt.savefig(outpath+'jdf_'+str(i+restart)+'.png')
    
    
    
    #############################
    #        AQUIFER PLOT       #
    #############################

    fig=plt.figure()
    

    # aqx = 17
    # aqx2 = (len(x)) - 17
    aqx = int((param_w/50.0)) +2 #+ 20
    aqx2 = len(x) - int((param_w_rhs/50.0)) - 6 #- 40
    aqy = 0
    aqy2 = len(y)

    
    # u velocity in the channel
    varMat = u[aqy:aqy2,aqx:aqx2]*(3.14e7)#ca0
    varStep = u[aqy:aqy2,aqx:aqx2]*(3.14e7)#ca
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    
    ax1=fig.add_subplot(2,1,1, aspect=asp*1.0,frameon=False)
    pGlass=plt.contourf(x[aqx:aqx2],y[aqy:aqy2],varStep,contours,cmap=cm.rainbow,alpha=1.0,antialiased=True)
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    p = plt.contour(xg[aqy:aqy2,aqx:aqx2],yg[aqy:aqy2,aqx:aqx2],perm[aqy:aqy2,aqx:aqx2],[-14.9],colors='black',linewidths=np.array([0.5]))
    
    plt.ylim([y[aqy],y[aqy2-1]])
    cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    cbar.ax.set_xlabel('u [m/yr]')



    varMat = v[aqy:aqy2,aqx:aqx2]*(3.14e7)#c14[aqy:aqy2,aqx:aqx2]#*(3.14e7)#dic0
    varStep = v[aqy:aqy2,aqx:aqx2]*(3.14e7)#c14[aqy:aqy2,aqx:aqx2]#*(3.14e7)#dic
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    
    ax1=fig.add_subplot(2,1,2, aspect=asp*1.0,frameon=False)
    pGlass = plt.contourf(x[aqx:aqx2], y[aqy:aqy2], varStep, contours, cmap=cm.rainbow, alpha=1.0,antialiased=True)
    
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    p = plt.contour(xg[aqy:aqy2,aqx:aqx2],yg[aqy:aqy2,aqx:aqx2],perm[aqy:aqy2,aqx:aqx2],[-14.9],colors='black',linewidths=np.array([0.5]))


    
    plt.ylim([y[aqy],y[aqy2-1]])
    cbar= plt.colorbar(pGlass, orientation='horizontal')
    cbar.ax.set_xlabel('v [m/yr]')


    plt.savefig(outpath+'jdfaq_'+str(i+restart)+'.png')
    
    
    
    xn = len(x)
    lim_a = 0.0
    lim_b = 1000.0
    lim_a0 = int(lim_a/(x[1]-x[0]))
    lim_b0 = int(lim_b/(x[1]-x[0]))
    lim_u = 0
    lim_o = len(y)

    aspSQ = asp/15.0
    aspZ = asp

    if i==0:

        ##########################
        #        ZOOM PLOT       #
        ##########################

        fig=plt.figure()

        varMat = maskP[lim_u:lim_o,lim_a0:lim_b0]
        varStep = maskP[lim_u:lim_o,lim_a0:lim_b0]
        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        
        ax1=fig.add_subplot(2,2,1,aspect=aspSQ,frameon=False)
        pGlass = plt.pcolor(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep)
        #p = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],perm[lim_u:lim_o,lim_a0:lim_b0],
        #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        
        plt.xlim([lim_a,lim_b])
        plt.title('LEFT OUTCROP maskP')

        varMat = maskP[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        varStep = maskP[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        
        ax1=fig.add_subplot(2,2,2,aspect=aspSQ,frameon=False)
        pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y[lim_u:lim_o], varStep)
        #p = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0],
        #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        
        plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
        plt.title('RIGHT OUTCROP maskP')
        
        
        varMat = perm#mask+maskP
        varStep = perm#mask+maskP
        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        
        ax1=fig.add_subplot(2,1,2,aspect=asp,frameon=False)
        pGlass = plt.pcolor(x, y, varStep)
        cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))

        plt.title('LEFT OUTCROP mask')


        plt.savefig(outpath+'jdfZoom_'+str(i)+'.png')
        
        
        ##########################
        #        PERM PLOT       #
        ##########################

        fig=plt.figure()

        varMat = perm[lim_u:lim_o,lim_a0:lim_b0]
        varStep = perm[lim_u:lim_o,lim_a0:lim_b0]
        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        
        ax1=fig.add_subplot(2,2,1,aspect=aspSQ,frameon=False)
        pGlass = plt.pcolor(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep)
        #p = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],perm[lim_u:lim_o,lim_a0:lim_b0],
        #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        
        plt.xlim([lim_a,lim_b])
        plt.title('LEFT OUTCROP kx')

        varMat = perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        varStep = perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        
        ax1=fig.add_subplot(2,2,2,aspect=aspSQ,frameon=False)
        pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y[lim_u:lim_o], varStep)
        #p = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0],
        #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        
        plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
        plt.title('RIGHT OUTCROP kx')
        
        
        
        
        varMat = perm[lim_u:lim_o,lim_a0:lim_b0]
        varStep = perm[lim_u:lim_o,lim_a0:lim_b0]
        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        
        ax1=fig.add_subplot(2,2,3,aspect=aspSQ,frameon=False)
        pGlass = plt.pcolor(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep)
        #p = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],perm[lim_u:lim_o,lim_a0:lim_b0],
        #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        
        plt.xlim([lim_a,lim_b])
        plt.title('LEFT OUTCROP ky')

        varMat = perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        varStep = perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        
        ax1=fig.add_subplot(2,2,4,aspect=aspSQ,frameon=False)
        pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y[lim_u:lim_o], varStep)
        #p = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0],
        #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        
        plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
        plt.title('RIGHT OUTCROP ky')
        
        



        plt.savefig(outpath+'jdfZoomK_'+str(i)+'.png')


    ######################################
    #        ZOOM OUTCROP PSI PLOT       #
    ######################################
    
    fig=plt.figure()
    
    varMat = temp[lim_u:lim_o,lim_a0:lim_b0]
    varStep = temp[lim_u:lim_o,lim_a0:lim_b0]
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    
    ax1=fig.add_subplot(2,2,1, aspect=aspSQ,frameon=False)
    pGlass = plt.contourf(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep, 40, cmap=cm.rainbow,vmin = np.min(temp),vmax=180)
    CS = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0], yg[lim_u:lim_o,lim_a0:lim_b0], psi[lim_u:lim_o,lim_a0:lim_b0], 8, colors='black',linewidths=np.array([0.5]))
    p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    
    plt.xlim([lim_a,lim_b])
    plt.ylim([np.min(y),0.])
    plt.colorbar(pGlass,orientation='horizontal')
    plt.title('LEFT OUTCROP')
    
    
    varMat = temp[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    varStep = temp[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    
    ax1=fig.add_subplot(2,2,2, aspect=aspSQ,frameon=False)
    pGlass = plt.contourf(x[xn-lim_b0:xn-lim_a0], y[lim_u:lim_o], varStep, cmap=cm.rainbow,vmin = np.min(temp),vmax=180)
    CS = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0], yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0], psi[lim_u:lim_o,xn-lim_b0:xn-lim_a0], 8, colors='black',linewidths=np.array([0.5]))
    p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    
    plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
    plt.ylim([np.min(y),0.])
    plt.colorbar(pGlass,orientation='horizontal')
    plt.title('RIGHT OUTCROP')
    

    varStep = psi[lim_u:lim_o,lim_a0:lim_b0]
    varMat = varStep

    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    ax1=fig.add_subplot(2,2,3, aspect=aspSQ,frameon=False)
    pGlass = plt.pcolor(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep, cmap=cm.rainbow)
    #p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    #cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    
    plt.xlim([lim_a,lim_b])
    plt.ylim([np.min(y),0.])
    
    
    varStep = psi[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    varMat = varStep

    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    ax1=fig.add_subplot(2,2,4, aspect=aspSQ,frameon=False)
    pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y[lim_u:lim_o], varStep, cmap=cm.rainbow)
    #p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    #cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))

    plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
    plt.ylim([np.min(y),0.])

    plt.savefig(outpath+'jdfZoomVel_'+str(i+restart)+'.png')
    
    
    ######################################
    #        ZOOM OUTCROP U,V PLOT       #
    ######################################
    
    fig=plt.figure()
    
    varMat = u[:,lim_a0:lim_b0]*(3.14e7)
    varStep = u[:,lim_a0:lim_b0]*(3.14e7)
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    
    ax1=fig.add_subplot(2,2,1, aspect=aspSQ,frameon=False)
    pGlass = plt.pcolor(x[lim_a0:lim_b0], y, varStep, cmap=cm.rainbow)
    p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    
    plt.colorbar(pGlass,orientation='horizontal',label='x velocity [m/yr]')
    
    plt.xlim([lim_a,lim_b])
    plt.ylim([np.min(y),0.])
    plt.title('LEFT OUTCROP')
    
    
    varMat = u[:,xn-lim_b0:xn-lim_a0]*(3.14e7)
    varStep = u[:,xn-lim_b0:xn-lim_a0]*(3.14e7)
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    
    ax1=fig.add_subplot(2,2,2, aspect=aspSQ,frameon=False)
    pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y, varStep,cmap=cm.rainbow)
    p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    
    plt.colorbar(pGlass,orientation='horizontal',label='x velocity [m/yr]')
    
    plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
    plt.ylim([np.min(y),0.])
    plt.title('RIGHT OUTCROP')
    


    varMat = v[:,lim_a0:lim_b0]*(3.14e7)
    varStep = v[:,lim_a0:lim_b0]*(3.14e7)
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    
    ax1=fig.add_subplot(2,2,3, aspect=aspSQ,frameon=False)
    pGlass = plt.pcolor(x[lim_a0:lim_b0], y, varStep, cmap=cm.rainbow)
    p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    
    plt.colorbar(pGlass,orientation='horizontal',label='y velocity [m/yr]')
    
    plt.xlim([lim_a,lim_b])
    plt.ylim([np.min(y),0.])
    
    
    varMat = v[:,xn-lim_b0:xn-lim_a0]*(3.14e7)
    varStep = v[:,xn-lim_b0:xn-lim_a0]*(3.14e7)
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    
    ax1=fig.add_subplot(2,2,4, aspect=aspSQ,frameon=False)
    pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y, varStep,cmap=cm.rainbow)
    p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    
    plt.colorbar(pGlass,orientation='horizontal',label='y velocity [m/yr]')

    plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
    plt.ylim([np.min(y),0.])
    

      
    plt.savefig(outpath+'jdfZoomOC_'+str(i+restart)+'.png')
    
    
 

    

    ################################
    ##       PLOT HEAT FLUX       ##
    ################################

    fig=plt.figure()

    ax1=fig.add_subplot(1,1,1)
    hf_lith = np.zeros(len(x))
    hf_lith2 = np.zeros(len(x))
    hf_top = np.zeros(len(x))
    hf_top2 = np.zeros(len(x))
    hf_lith = 1.8*(temp[0,:] - temp[1,:])/(y[1]-y[0])*1000.0
    hf_lambda = np.zeros(len(x))

    for ii in range(len(x)):

        for j in range(1,len(y)-1):
            if mask[j,ii] == 50.0:
                hf_top[ii] = lambdaMat[j,ii]*(temp[j,ii] - temp[j+1,ii])/(y[1]-y[0])*1000.0
                hf_top2[ii] = 1.2*(temp[j-1,ii] - temp[j,ii])/(y[1]-y[0])*1000.0
                hf_lith2[ii] = lambdaMat[j,ii]*(temp[0,ii] - temp[1,ii])/(y[1]-y[0])*1000.0
                hf_lambda[ii] = lambdaMat[j,ii]



    sed = np.abs(mpd.interp_s)
    sed1 = np.abs(mpd.interp_b)
    sed2 = np.abs(mpd.interp_s - mpd.interp_b)
    
    scale_bit0 = np.abs(mpd.bas_new[6:] - (mpd.sed_new[6:]))
    scale_bit = (mpd.bas_new[6:]-mpd.sed_new[6:])/(mpd.bas_new[6:]+100.0)
    
    scale = scale_bit0/np.mean(scale_bit0)
    
    #if i == 0:
    #    print (mpd.sed_new[:-5]+100.0)/(mpd.bas_new[:-5]-mpd.sed_new[:-5])

    # m1 = hf_top[10:-10]*np.min(sed1[10:-10])/(sed1[10:-10])
    # m2 = hf_top[10:-10]*np.min(sed2[10:-10])/(sed2[10:-10])
    # m3 = (m1 + m2) / 2.0
    pl = plt.plot(x,hf_lith,'r',label='hf_lith')
    #pl = plt.plot(x[10:-10],hf_top[10:-10]*np.min(sed1[10:-10])/(sed1[10:-10]),'c',label='model-predicted heat flow')
    pl = plt.plot(x,hf_top2,'b',label='model output scaled to 1.2')
    # pl = plt.plot(x[:-7],hf_top[:-7]/(scale_bit*scale),'m',label='model output scaled to lambda') #(mpd.sed_new[6:]/(-100.0))*
    # pl = plt.plot(x[6:],hf_top[6:] + hf_top[6:]*(mpd.sed_new[:-5]+np.abs(np.mean(mpd.sed_new)-mpd.sed_new[:-5]-100.0))/(mpd.bas_new[:-5]-mpd.sed_new[:-5]),'k',label='model output scaled diferently') #(mpd.sed_new[6:]/(-100.0))*
    pl = plt.plot(x[6:],hf_top[6:]*(np.mean(mpd.sed_new[:-5]+250.0)/(mpd.sed_new[:-5]+250.0))/((mpd.sed_new[:-5]-mpd.bas_new[:-5])/np.mean(mpd.sed_new[:-5]-mpd.bas_new[:-5])),'m',label='model output scaled diferently') #(mpd.sed_new[6:]/(-100.0))*
    pl = plt.plot(x[7:],mpd.bas_new[6:],'g',label='mpd.bas_new')
    pl = plt.plot(x[7:],mpd.sed_new[6:],'k',label='mpd.sed_new')
    # pl = plt.plot(x[10:-10],hf_top[10:-10]*np.min(sed1[10:-10]+sed[10:-10])/(sed1[10:-10]+sed[10:-10]),'g',label='questionable model output')
    # pl = plt.plot(x[10:-10],hf_top[10:-10]*np.min(sed2[10:-10])/(sed2[10:-10]),'gold',label='model-predicted heat flow')
    # pl = plt.plot(x[10:-10],m3,'r',label='model-predicted heat flow')

    pl = plt.scatter((5.0/5.0)*mpd.ripXdata+300.0,mpd.ripQdata,label='heat flow data')

    plt.ylim([0.0,700.0])
    #plt.xlim(0.0,26000.0)
    plt.legend(fontsize=8,loc='upper right')
    ax1.grid(True)





    plt.savefig(outpath+'jdfzzhf_'+str(i+restart)+'.png')
    
    



    
    
    
    
    
    
    

      
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
    
        #######################
        ##    CHEM 1 PLOT    ##
        #######################
    
        fig=plt.figure()
    
        chemplot(ca0, ca, 3, 2, 1, 1, '[Ca]',1)
        chemplot(glass0_p, glass_p, 3, 2, 2, 1, 'BASLATIC GLASS [fraction remaining]',1)

        chemplot(ph0, ph, 3, 2, 3, 2, 'pH',1)
        chemplot(alk0, alk, 3, 2, 4, 1, 'alkalinity',1)
        chemplot(dic0, dic, 3, 2, 5, 1, 'DIC',1)
        chemplot(secMat[:,:,16], secStep[:,:,16], 3, 2, 6, 1, 'CALCITE [cm$^3$]',0)

        fig.set_tight_layout(True)
        plt.savefig(outpath+'jdfChem1_'+str(i+restart)+'.png')
        
        
        
        
        
        #######################
        ##     FLUID COMP    ##
        #######################
    
        fig=plt.figure()
    
        chemplot(ca0, ca, 3, 2, 1, 1, '[Ca]',1)
        chemplot(mg0, mg, 3, 2, 2, 1, '[Mg]',1)
        chemplot(na0, na, 3, 2, 3, 1, '[Na]',1)
        chemplot(k0, k1, 3, 2, 4, 1, '[K]',1)
        chemplot(al0, al, 3, 2, 5, 1, '[Al]',1)
        chemplot(si0, si, 3, 2, 6, 1, '[Si]',1)
        # chemplot(fe0, fe, 3, 3, 7, 1, '[Fe]',1)


        fig.set_tight_layout(True)
        plt.savefig(outpath+'jdfChemFluid_'+str(i+restart)+'.png')
        
        
        

    
    
    
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
        

        


    
        fig=plt.figure()
        
        chemplot(glass0_p, glass_p, 3, 2, 1, 1, 'Fractional volume of primary basalt',1)
        
        chemplot(alt_vol0, alt_vol, 3, 2, 2, 1, 'Total secondary mineral fractional volume',0)
    
        chemplot(smectites0, smectites, 3, 2, 3, 1, 'Fractional volume of smectite minerals',0)
        
        chemplot(zeolites0, zeolites, 3, 2, 4, 1, 'Fractional volume of zeolite minerals',0)
        
        chemplot(chlorites0, chlorites, 3, 2, 5, 1, 'Fractional volume of chlorite minerals',0)
        
        chemplot(alk_flux, alk_flux, 3, 2, 6, 1, 'Alkalinity flux to ocean [eq/L/yr]',0)


        fig.set_tight_layout(True)
        plt.savefig(outpath+'jdfGroup_'+str(i+restart)+'.png')
        
        
        
        #######################
        ##    GROUP PLOT     ##
        #######################
        

        


    
        fig=plt.figure()
        
        chemplot(glass0_p, glass_p, 3, 1, 1, 1, 'Fractional volume of primary basalt',1)
        
        chemplot(alt_vol0, alt_vol, 3, 1, 2, 1, 'Total secondary mineral fractional volume',0)
        
        chemplot(alk_flux, alk_flux, 3, 1, 3, 1, 'Alkalinity flux to ocean [eq/L/yr]',0)


        fig.set_tight_layout(True)
        plt.savefig(outpath+'jdfVis_'+str(i+restart)+'.eps')
        
        
    
    
        ###########################
        ##    SECONDARY PLOTZ    ##
        ###########################
    
    
        fig=plt.figure()

        # chemplot(secMat[:,:,6], secStep[:,:,5], 4, 2, 1, 4, 'CELADONITE [mol]',0)
        # chemplot(secMat[:,:,6], secStep[:,:,6], 4, 2, 2, 4, 'CELADONITE [mol]',0)
        # chemplot(secMat[:,:,8], secStep[:,:,8], 4, 2, 3, 4, 'PYRITE [mol]',0)
        # chemplot(secMat[:,:,19], secStep[:,:,19], 4, 2, 4, 4, 'SAPONITE NA [mol]',0)
        # chemplot(secMat[:,:,20], secStep[:,:,20], 4, 2, 5, 4, 'NONTRONITE NA [mol]',0)
        #
        # chemplot(secMat[:,:,14], secStep[:,:,14], 4, 2, 6, 4, 'NONTRONITE MG [mol]',0)
        # chemplot(secMat[:,:,17], secStep[:,:,17], 4, 2, 7, 4, 'MUSCOVITE [mol]',0)
        #
        # chemplot(secMat[:,:,26], secStep[:,:,26], 4, 2, 8, 4, 'MESOLITE [mol]',0)
    
        sp = 0
        j_last = 0
        for j in range(0,minNum):
            if np.max(secMat[:,:,j]) > 0.0 and sp < 6:
                sp = sp + 1
                j_last = j
                print j_last
                chemplot(secMat[:,:,j], secStep[:,:,j], 3, 2, sp, 1, secondary[j] + ' [cm$^3$]',0)

    
        fig.set_tight_layout(True)
        plt.savefig(outpath+'jdfChem2_'+str(i+restart)+'.png')
    
        print " "
        fig=plt.figure()

        sp = 0
        for j in range(j_last+1,minNum):
            if np.max(secMat[:,:,j]) > 0.0 and sp < 6:
                sp = sp + 1
                j_last = j
                print j_last
                chemplot(secMat[:,:,j], secStep[:,:,j], 3, 2, sp, 1, secondary[j] + ' [cm$^3$]',0)

    
        fig.set_tight_layout(True)
        plt.savefig(outpath+'jdfChem3_'+str(i+restart)+'.png')
    
    
        print " "
        fig=plt.figure()

        sp = 0
        for j in range(j_last+1,minNum):
            if np.max(secMat[:,:,j]) > 0.0 and sp < 6:
                sp = sp + 1
                j_last = j
                print j_last
                chemplot(secMat[:,:,j], secStep[:,:,j], 3, 2, sp, 1, secondary[j] + ' [cm$^3$]',0)

    
        fig.set_tight_layout(True)
        plt.savefig(outpath+'jdfChem4_'+str(i+restart)+'.png')
    
    
    
    

    
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
 
 
    
##########################
##    PLOT U MEAN TS    ##
##########################
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)

plt.plot(u_ts)

#plt.ylim([-0.2,0.2])

plt.savefig(outpath+'jdf_u_ts.eps')