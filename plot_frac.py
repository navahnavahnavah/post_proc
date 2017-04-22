# revived_JDF.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import streamplot as sp
import multiplot_data as mpd
import heapq
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
chem = 0
iso = 0

outpath = "output/revival/benchmarks17/fracture/mask/"

path = outpath
param_w = 300.0
param_w_rhs = 200.0


# load output
x0 = np.loadtxt(path + 'x.txt',delimiter='\n')
y0 = np.loadtxt(path + 'y.txt',delimiter='\n')

temp6 = np.loadtxt(path + 'temp6.txt')
finite_diff = np.loadtxt(path + 'fdiff2.txt')

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
print "bitsy" , bitsy


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



delta = np.zeros(lambdaMat.shape)
    
conv_mean_qu = 0.0
conv_max_qu = 0.0
conv_mean_psi = 0.0
conv_max_psi = 0.0
conv_tot_hf = 0.0
conv_count = 0

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



    alk_flux = np.zeros(secStep[:,:,1].shape)

    

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
    #       TEMP6 PLOT       #
    ##########################
    
    fig=plt.figure()
    
    plt.plot(np.arange(30.0),temp6[0,32:]-273.0)
    plt.plot([29.0, 30.0],[temp6[0,-1]-273.0, 0.0])
    # plt.plot((31.0/20.0)*np.arange(20.0),finite_diff[:,1])
    plt.plot((31.0/20.5)*np.arange(21.0),finite_diff[:,1])
    
    plt.savefig(outpath+'jdf_temp6_eps'+str(i+restart)+'.eps')


    
        
    ##########################
    #        HEAT PLOT       #
    ##########################

    fig=plt.figure()
    

    # temp plot
    varStep = temp 
    varMat = varStep
    contours = np.linspace(np.min(varMat),np.max(varMat),30)
              
    ax1=fig.add_subplot(2,1,2, aspect=1.5,frameon=True)
    # pGlass = plt.contourf(x, y, varStep, contours, cmap=cm.rainbow, alpha=1.0,color='#444444',antialiased=True)
    pGlass = plt.contour(x, y, varStep, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], cmap=cm.rainbow, alpha=1.0,color='#444444',antialiased=True)
    p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    CS = plt.contour(xg, yg, psi, 8, colors='black',linewidths=np.array([0.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    
    # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::2])
    # cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
    


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
    
    

    plt.savefig(outpath+'jdf_eps_'+str(i+restart)+'.eps')
    
    
    
    #############################
    #        AQUIFER PLOT       #
    #############################
    
    #-aquifer

    fig=plt.figure()
    

    # aqx = 17
    # aqx2 = (len(x)) - 17
    aqx = 0 #+ 20
    aqx2 = -1 #- 40
    aqy = 0
    aqy2 = len(y)

    
    # u velocity in the channel
    varMat = u[aqy:aqy2,aqx:aqx2]*(3.14e7)#ca0
    varStep = u[aqy:aqy2,aqx:aqx2]*(3.14e7)#ca
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    scanned = varStep[:,bitsx/2]
    #print scanned
    print "mean scanned qu" , np.mean(scanned[np.abs(scanned)>0.001])
    # print "sum scanned qu" , np.sum(scanned[scanned>0.001])/(200.0/(y[1]-y[0]))
    print "sum scanned qu" , np.sum(scanned[scanned>0.001])/((200.0+1.5*(y[1]-y[0]))/(y[1]-y[0]))
    print "mmaaxx scanned qu" , np.max(scanned)
    # print "max qu" , np.max(scanned)
    # print "mean psi" , np.mean(psi)
    print "max psi" , np.max(psi)
    # print "max psi, aquifer" , np.max(psi[aqy:aqy2,aqx:aqx2])
    # print "mean psi, aquifer" , np.mean(psi[aqy:aqy2,aqx:aqx2])
    
    print " "
    print "new thing"
    #print np.sum(heapq.nlargest(4,scanned))/4.0
    
    print " "
    
    if i >= 4:
        conv_count = conv_count + 1
        # conv_mean_qu = conv_mean_qu + np.sum(scanned[scanned>0.001])/(200.0/(y[1]-y[0]))
        conv_mean_qu = conv_mean_qu + np.mean(scanned[np.abs(scanned)>0.1])#np.sum(scanned)/(200.0/(y[1]-y[0]))
        conv_max_qu = conv_max_qu + np.max(varStep)
        conv_mean_psi = conv_mean_psi + np.mean(psi)
        conv_max_psi = conv_max_psi + np.max(psi)
    if i == 9:
    #if i >= 4:
        print "means over time:"
        print "conv_mean_qu" , conv_mean_qu/float(conv_count)
        print "conv_max_qu" , conv_max_qu/float(conv_count)
        print "conv_mean_psi" , conv_mean_psi/float(conv_count)
        print "conv_max_psi" , conv_max_psi/float(conv_count)
    
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
    lim_b = 20.0
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
        
        ax1=fig.add_subplot(2,2,1,frameon=False)
        pGlass = plt.pcolor(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep)
        #p = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],perm[lim_u:lim_o,lim_a0:lim_b0],
        #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        
        plt.xlim([lim_a,lim_b])
        plt.title('LEFT OUTCROP maskP')

        varMat = maskP[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        varStep = maskP[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        
        ax1=fig.add_subplot(2,2,2,frameon=False)
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



    
 



 
    
##########################
##    PLOT U MEAN TS    ##
##########################
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)

plt.plot(u_ts)

#plt.ylim([-0.2,0.2])

plt.savefig(outpath+'jdf_u_ts.eps')