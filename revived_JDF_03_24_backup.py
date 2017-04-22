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

cell = 1
#steps = 400
steps = 5
minNum = 58
ison=10000
inertn = 100000

#outpath = "output/revival/eggs_c/o600w1000h200orhs800/"
#outpath = "output/revival/eggs_a/o300w1000h200orhs700/"
outpath = "output/revival/jan16/o600w1000h200orhs600/"
path = outpath
param_w = 1200.0


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
restart = 0

xCell = x0
yCell = y0
xCell = xCell[::cell]
yCell= yCell[::cell]
xCell = np.append(xCell, np.max(xCell)+xCell[1])
yCell = np.append(yCell, np.max(yCell)-yCell[-1])
bitsCx = len(xCell)
bitsCy = len(yCell)
# print bitsCx
# print bitsCy
#
# print xCell
# print yCell

xg, yg = np.meshgrid(x[:],y[:])

mask = np.loadtxt(path + 'mask.txt')
maskP = np.loadtxt(path + 'maskP.txt')
psi0 = np.loadtxt(path + 'psiMat.txt')
perm0 = np.loadtxt(path + 'permMat.txt')
perm0 = np.log10(perm0)

temp0 = np.loadtxt(path + 'hMat.txt')
temp0 = temp0 - 273.0
u0 = np.loadtxt(path + 'uMat.txt')
v0 = np.loadtxt(path + 'vMat.txt')
lambdaMat = np.loadtxt(path + 'lambdaMat.txt')

c140 = np.loadtxt(path + 'iso_c14.txt')
control0 = np.loadtxt(path + 'iso_control.txt')

# dic0 = np.loadtxt(path + 'sol_c.txt')
# ca0 = np.loadtxt(path + 'sol_ca.txt')
# glass0 = np.loadtxt(path + 'pri_glass.txt')

#temp0[temp0<273.0] = 273.0
#temp0 = temp0 - 273.0

#mesh = np.loadtxt(path + 'mesh.txt')
lam = np.loadtxt(path + 'lambdaMat.txt')

isopart_10 = np.loadtxt(path + 'isopart_1.txt')
inert_trace0 = np.loadtxt(path + 'inert_trace.txt')

# IMPORT MINERALS

# uCoarseMat = np.loadtxt(path + 'uCoarseMat.txt')
# vCoarseMat = np.loadtxt(path + 'vCoarseMat.txt')
# psiCoarseMat = np.loadtxt(path + 'psiCoarseMat.txt')

def cut(geo0,index):
    #geo_cut = geo0[(index*len(y0)/cell):(index*len(y0)/cell+len(y0)/cell),:]
    geo_cut = geo0[:,(index*len(x0)/cell):(index*len(x0)/cell+len(x0)/cell)]
    geo_cut = np.append(geo_cut, geo_cut[-1:,:], axis=0)
    geo_cut = np.append(geo_cut, geo_cut[:,-1:], axis=1)
    return geo_cut

def chemplot(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, ditch):
    if ditch==0:
        contours = np.linspace(np.min(varMat),np.max(varMat),20)
    if ditch==1:
        contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),20)
        #contours = np.linspace(contours[0],contours[-1],20)
    if ditch==2:
        contours = np.linspace(np.min(varMat),np.max(varMat[varMat<varStep[bitsy-25,bitsx/2]])/5.0,20)
        #contours = np.linspace(contours[0],contours[-1],20)
    ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp/1.0,frameon=False)
    pGlass = plt.contourf(xCell,yCell,varStep,contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0,antialiased=True)
    p = plt.contour(xg,yg,perm,[-12.0,-13.5],colors='black',linewidths=np.array([1.0]))
    plt.xticks([])
    plt.yticks([])
    cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
    plt.title(cp_title)
    cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::contour_interval])
    fig.set_tight_layout(True)
    return chemplot


secMat = np.zeros([(bitsCy-1),(bitsCx-1)*steps,minNum+1])
#satMat = np.zeros([(bitsCy-1),(bitsCx-1)*steps,minNum+1])

secStep = np.zeros([bitsCy,bitsCx,minNum+1])
dsecStep = np.zeros([bitsCy,bitsCx,minNum+1])
#satStep = np.zeros([bitsCy,bitsCx,minNum+1])

# uCoarseStep = np.zeros([bitsCy,bitsCx])
# vCoarseStep = np.zeros([bitsCy,bitsCx])
# psiCoarseStep = np.zeros([bitsCy,bitsCx])

# for j in range(1,minNum):
#     secMat[:,:,j] = np.loadtxt(path + 'sec' + str(j) + '.txt')
#     #satMat[:,:,j] = np.loadtxt(path + 'sat' + str(j) + '.txt')



delta = np.zeros(lambdaMat.shape)
    
    
for i in range(0,steps,1): 
    print " "
    print " "
    print " "
    print "step =", i
    
    if i == 0:
        dupe = np.zeros(lambdaMat.shape)
        dupe2 = np.zeros(lambdaMat.shape)
  
    if i > 0:
        dupe = psi
        dupe2 = temp


    
    
    # #rho = rho0[i*len(y):((i)*len(y)+len(y)),:]
    # psi = psi0[i*len(y):((i)*len(y)+len(y)),:]
    # perm = perm0[i*len(y):((i)*len(y)+len(y)),:]
    # temp = temp0[i*len(y):((i)*len(y)+len(y)),:]
    # u = u0[i*len(y):((i)*len(y)+len(y)),:]
    # v = v0[i*len(y):((i)*len(y)+len(y)),:]
#    rho = rho0[:,i*len(x):((i)*len(x)+len(x))]
    psi = psi0[:,i*len(x):((i)*len(x)+len(x))]
    perm = perm0[:,i*len(x):((i)*len(x)+len(x))]
    temp = temp0[:,i*len(x):((i)*len(x)+len(x))]
    u = u0[:,i*len(x):((i)*len(x)+len(x))]
    v = v0[:,i*len(x):((i)*len(x)+len(x))]

    
    c14 = c140[:,i*len(x):((i)*len(x)+len(x))]
    control = control0[:,i*len(x):((i)*len(x)+len(x))]
    #
    # if i > 0:
    #     isopart_1_last = isopart_1
    isopart_1 = isopart_10[i*ison:i*ison+ison,:]
    inert_trace = inert_trace0[i*inertn:i*inertn+inertn,:]


    # if i == 0:
    #     figp=plt.figure()
    #     axp=figp.add_subplot(1,1,1,frameon=False,aspect=4.0*asp)
    #     cMask = axp.contour(xg,yg,maskP,[0.0,0.5],colors='k',linewidths=np.array([0.5]))
    #     p = axp.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    #     plt.xlim([0-3000.0,np.max(x)+3000.0])
    #     plt.ylim([np.min(y)/2,0])
    #     #CS = axp.contour(xg, yg, psi, 8, colors='black',linewidths=np.array([0.5]))
    # #plt.grid()
    # if i > 0:
    #     axp.scatter(isopart_1[:,0],isopart_1[:,1],s=8,c=isopart_1[:,2],cmap=cm.rainbow,edgecolor='none')
    #     # for n in range(0,ison,10):
    #     #     if (np.abs(isopart_1_last[n,0] - isopart_1[n,0]) < 2000.0) & (np.abs(isopart_1_last[n,1] - isopart_1[n,1]) < 400.0):
    #     #         axp.plot([isopart_1_last[n,0],isopart_1[n,0]],[isopart_1_last[n,1],isopart_1[n,1]],'b')
    #
    #
    #
    # if i == 4:
    #     figp.savefig(outpath+'jdf_particletest_0' + str('one') + '.png')
    


    
    isopart_1_01 = isopart_1[isopart_1[:,2] == 1.0,:]
    inert_trace_01 = inert_trace[inert_trace[:,2] == 10.0,:]

    
    # make the x frequency plot
    iso_1_01_x = np.zeros(len(x))
    inert_trace_01_x = np.zeros(len(x))
    for j in range(len(x)-1):
        iso_1_01_x[j] = np.count_nonzero(isopart_1_01[isopart_1_01[:,0] >= x[j],:])
        iso_1_01_x[j] = iso_1_01_x[j] - np.count_nonzero(isopart_1_01[isopart_1_01[:,0] > x[j+1],:])
        
        inert_trace_01_x[j] = np.count_nonzero(inert_trace_01[inert_trace_01[:,0] >= x[j],:])
        inert_trace_01_x[j] = inert_trace_01_x[j] - np.count_nonzero(inert_trace_01[inert_trace_01[:,0] > x[j+1],:])
        
    # make the y frequency plot
    iso_1_01_y = np.zeros(len(y))
    inert_trace_01_y = np.zeros(len(y))
    for j in range(len(y)-1):
        iso_1_01_y[j] = np.count_nonzero(isopart_1_01[isopart_1_01[:,1] >= y[j],:])
        iso_1_01_y[j] = iso_1_01_y[j] - np.count_nonzero(isopart_1_01[isopart_1_01[:,1] >= y[j+1],:])
        
        inert_trace_01_y[j] = np.count_nonzero(inert_trace_01[inert_trace_01[:,1] >= y[j],:])
        inert_trace_01_y[j] = inert_trace_01_y[j] - np.count_nonzero(inert_trace_01[inert_trace_01[:,1] >= y[j+1],:])



    fig=plt.figure()
    
    
    # ax = fig.add_axes([left, bottom, width, height])
    

    
    ax = fig.add_axes([0.15, 0.2, .6, .4])
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='k',linewidths=np.array([1.5]))
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([0.5]))
    # plt.plot([np.min(x),np.min(x)],[np.min(y),np.max(y)],'k')
    # plt.plot([np.max(x),np.max(x)],[np.min(y),np.max(y)],'k')


    
    plt.scatter(isopart_1_01[:,0],isopart_1_01[:,1],s=1,c='b',edgecolor='none')
    plt.scatter(inert_trace_01[:,0],inert_trace_01[:,1],s=1,c='g',edgecolor='none')

    
    plt.xlim([0,np.max(x)])
    plt.ylim([np.min(y)/2,0])
    
    # right plot
    ax = fig.add_axes([0.77, 0.2, .15, .4],frameon=True)
    plt.plot(inert_trace_01_y,y,'g')
    plt.plot(iso_1_01_y,y,'b')
    plt.ylim([np.min(y)/2,0])
    plt.xticks(np.linspace(0,np.max([iso_1_01_y,inert_trace_01_y]),5))
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.xaxis.set_ticks_position('bottom')
    
    # top plot
    ax = fig.add_axes([0.15, 0.63, .6, .2],frameon=True)
    plt.plot(inert_trace_01_x,'g')
    plt.plot(x,iso_1_01_x,'b')
    plt.xlim([0-600.0,np.max(x)+600.0])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_xaxis().set_ticks([])
    ax.yaxis.set_ticks_position('left')
    

    
    fig.savefig(outpath+'jdf_particletest_0' + str(i) + '.png')
    
    
    
    
    
    
    
    
    fig=plt.figure()
    
    
    # ax = fig.add_axes([left, bottom, width, height])
    

    
    ax = fig.add_axes([0.15, 0.2, .6, .4])
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='k',linewidths=np.array([1.5]))
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([0.5]))

    plt.scatter(isopart_1_01[:,0],isopart_1_01[:,1],s=1,c='b',edgecolor='none')
    plt.scatter(inert_trace_01[:,0],inert_trace_01[:,1],s=1,c='g',edgecolor='none')

    
    plt.xlim([0,np.max(x)])
    plt.ylim([np.min(y)/2,0])
    
    # right plot
    ax = fig.add_axes([0.77, 0.2, .15, .4],frameon=True)
    plt.plot(inert_trace_01_y,y,'g')
    plt.plot(iso_1_01_y,y,'b')
    plt.ylim([np.min(y)/2,0])
    plt.xticks(np.linspace(0,np.max([iso_1_01_y,inert_trace_01_y]),5))
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.xaxis.set_ticks_position('bottom')
    
    # top plot
    ax = fig.add_axes([0.15, 0.63, .6, .2],frameon=True)
    plt.plot(inert_trace_01_x,'g')
    plt.plot(x,iso_1_01_x,'b')
    plt.xlim([0-600.0,np.max(x)+600.0])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_xaxis().set_ticks([])
    ax.yaxis.set_ticks_position('left')
    

    
    fig.savefig(outpath+'jdf_particle_conc_0' + str(i) + '.png')

            
    
    # print 10.0**perm[41,:]
    # print 10.0**perm[40,:]
    # print 10.0**perm[39,:]
                


    # for j in range(1,minNum):
    #     secStep[:,:,j] = cut(secMat[:,:,j],i)
    #     #satStep[:,:,j] = cut(satMat[:,:,j],i)
    #     dsecStep[:,:,j] = cut(secMat[:,:,j],i)# - cut(secMat[:,:,j],i-1)
    # dic = cut(dic0,i)
    # ca = cut(ca0,i)
    # glass = cut(glass0,i)

    
    
        
    #############
    # HEAT PLOT #
    #############

    fig=plt.figure()
    

    varStep = temp 
    varMat = varStep
              
    contours = np.linspace(np.min(varStep),np.max(varStep),10)
    ax1=fig.add_subplot(2,1,2, aspect=asp,frameon=False)
    contours = np.linspace(np.min(varMat),np.max(varMat),10)
    pGlass = plt.contourf(x, y, varStep, cmap=cm.rainbow, alpha=1.0,color='#444444',antialiased=True)
    cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')

    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    if np.max(np.abs(psi)) > 0.0:
        CS = plt.contour(xg, yg, psi, 8, colors='black',linewidths=np.array([0.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    cMask = plt.contour(xg,yg,mask,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    

    #[-11.15,-14.50]
    varMat = v*(3.14e7)#dic0
    varStep = v*(3.14e7)#dic

    contours = np.linspace(np.min(varMat),np.max(varMat),10)
    ax1=fig.add_subplot(2,2,2, aspect=asp,frameon=False)
    pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    cbar.ax.set_xlabel('v [m/yr]')
    #p = plt.contour(xg,yg,perm,[-12.15,-17.25],colors='black',linewidths=np.array([1.0]))
    # cMask = plt.contourf(xg,yg,maskP,[0.0,0.5],colors='white',alpha=0.4,zorder=10)
    # cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='black',alpha=0.4,zorder=10)
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    cMask = plt.contour(xg,yg,mask,[0.0,0.5],colors='w',linewidths=np.array([0.5]))


    varMat = u*(3.14e7)#ca0
    varStep = u*(3.14e7)#ca

    contours = np.linspace(np.min(varMat),np.max(varMat),10)
    ax1=fig.add_subplot(2,2,1, aspect=asp,frameon=False)
    pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    cbar.ax.set_xlabel('u [m/yr]')
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    cMask = plt.contour(xg,yg,mask,[0.0,0.5],colors='w',linewidths=np.array([0.5]))



    
    
    plt.savefig(outpath+'jdf_'+str(i+restart)+'.png')
    
    
    
    ################
    # AQUIFER PLOT #
    ################

    fig=plt.figure()
    

    aqx = 26 + len(x)*1/26
    aqx2 = (len(x)*25/26) - 4
    aqy = len(y)/2
    aqy2 = len(y)

    
    
    varMat = u[aqy:aqy2,aqx:aqx2]*(3.14e7)#ca0
    varStep = u[aqy:aqy2,aqx:aqx2]*(3.14e7)#ca
    colMax = np.zeros(len(x[aqy:aqy2]))
    contours = np.linspace(np.min(varMat),np.max(varMat),10)
    ax1=fig.add_subplot(2,1,1, aspect=asp*2.0,frameon=False)
    pGlass=plt.contourf(x[aqx:aqx2],y[aqy:aqy2],varStep,contours,cmap=cm.rainbow,alpha=1.0,antialiased=True)
    cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    cbar.ax.set_xlabel('u [m/yr]')
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    p = plt.contour(xg[aqy:aqy2,aqx:aqx2],yg[aqy:aqy2,aqx:aqx2],perm[aqy:aqy2,aqx:aqx2],[-15.9],colors='black',linewidths=np.array([0.5]))
    #CS = plt.contour(xg[aqy:aqy2,aqx:aqx2], yg[aqy:aqy2,aqx:aqx2], psi[aqy:aqy2,aqx:aqx2], 8, colors='black',linewidths=np.array([0.5]))
    plt.ylim([y[aqy],y[aqy2-1]])
    
    #c14[perm <= -13] = 0.0
    
    varMat = v[aqy:aqy2,aqx:aqx2]*(3.14e7)#c14[aqy:aqy2,aqx:aqx2]#*(3.14e7)#dic0
    varStep = v[aqy:aqy2,aqx:aqx2]*(3.14e7)#c14[aqy:aqy2,aqx:aqx2]#*(3.14e7)#dic

    ax1=fig.add_subplot(2,1,2, aspect=asp*2.0,frameon=False)
    pGlass = plt.contourf(x[aqx:aqx2], y[aqy:aqy2], varStep, cmap=cm.rainbow, alpha=1.0,antialiased=True)
    cbar= plt.colorbar(pGlass, orientation='horizontal')
    # cbar.ax.set_xlabel('log$_{10}$($^{14}$C)')
    cbar.ax.set_xlabel('v [m/yr]')
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    p = plt.contour(xg[aqy:aqy2,aqx:aqx2],yg[aqy:aqy2,aqx:aqx2],perm[aqy:aqy2,aqx:aqx2],[-15.9],colors='black',linewidths=np.array([0.5]))
    plt.plot([param_w + 3300.0, param_w + 3300.0], [np.min(y), np.max(y)],'k')
    plt.plot([param_w + 6700.0, param_w + 6700.0], [np.min(y), np.max(y)],'k')
    plt.plot([param_w + 14600.0, param_w + 14600.0], [np.min(y), np.max(y)],'k')
    plt.plot([param_w + 21000.0, param_w + 21000.0], [np.min(y), np.max(y)],'k')
    #CS = plt.contour(xg[aqy:aqy2,aqx:aqx2], yg[aqy:aqy2,aqx:aqx2], psi[aqy:aqy2,aqx:aqx2], 8, colors='black',linewidths=np.array([0.5]))
    plt.ylim([y[aqy],y[aqy2-1]])



    
    
    plt.savefig(outpath+'jdfaq_'+str(i+restart)+'.png')
    
    
    
    xn = len(x)
    lim_a = 0.0
    lim_b = 3000.0
    lim_a0 = int(lim_a/(x[1]-x[0]))
    lim_b0 = int(lim_b/(x[1]-x[0]))
    lim_u = len(y)/3
    lim_o = len(y)

    aspSQ = asp/3.0
    aspZ = asp

    if i==0:

        #####################
        ##    ZOOM PLOT    ##
        #####################

        fig=plt.figure()

        varMat = maskP[lim_u:lim_o,lim_a0:lim_b0]
        varStep = maskP[lim_u:lim_o,lim_a0:lim_b0]

        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        ax1=fig.add_subplot(2,2,1,aspect=2.0*aspSQ,frameon=False)
        pGlass = plt.pcolor(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep, vmin=0.0, vmax=100.0)
        p = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],perm[lim_u:lim_o,lim_a0:lim_b0],
        [-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        # plt.xlim([0,4000])
        plt.xlim([lim_a,lim_b])
        plt.title('LEFT OUTCROP maskP')

        varMat = maskP[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        varStep = maskP[lim_u:lim_o,xn-lim_b0:xn-lim_a0]

        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        ax1=fig.add_subplot(2,2,2,aspect=aspSQ*2.0,frameon=False)
        pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y[lim_u:lim_o], varStep, vmin=0.0, vmax=100.0)
        p = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0],
        [-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        # plt.xlim([np.max(x)-4000.0,np.max(x)])
        plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
        plt.title('RIGHT OUTCROP maskP')
        
        
        varMat = perm#mask+maskP
        varStep = perm#mask+maskP

        contours = np.linspace(np.min(varMat),np.max(varMat),10)
        ax1=fig.add_subplot(2,1,2,aspect=asp,frameon=False)
        pGlass = plt.pcolor(x, y, varStep)
        cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
        #p = plt.contour(xg,yg,perm,[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))

        # plt.xlim([lim_a,lim_b])
        plt.title('LEFT OUTCROP mask')


        plt.savefig(outpath+'jdfZoom_'+str(i)+'.png')


    ###########################
    ##    ZOOM U VEL PLOT    ##
    ###########################
    
    fig=plt.figure()
    
    varMat = temp[lim_u:lim_o,lim_a0:lim_b0]
    varStep = temp[lim_u:lim_o,lim_a0:lim_b0]
    
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    ax1=fig.add_subplot(2,2,1, aspect=aspSQ*2.0,frameon=False)
    pGlass = plt.contourf(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep, cmap=cm.rainbow,vmin = np.min(temp),vmax=180)
    plt.colorbar(orientation='horizontal')
    CS = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0], yg[lim_u:lim_o,lim_a0:lim_b0], psi[lim_u:lim_o,lim_a0:lim_b0], 8, colors='black',linewidths=np.array([0.5]))
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    plt.xlim([lim_a,lim_b])
    plt.ylim([np.min(y)/2.0,0.])
    plt.title('LEFT OUTCROP')
    
    varMat = temp[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    varStep = temp[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    ax1=fig.add_subplot(2,2,2, aspect=aspSQ*2.0,frameon=False)
    pGlass = plt.contourf(x[xn-lim_b0:xn-lim_a0], y[lim_u:lim_o], varStep, cmap=cm.rainbow,vmin = np.min(temp),vmax=180)
    plt.colorbar(orientation='horizontal')
    CS = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0], yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0], psi[lim_u:lim_o,xn-lim_b0:xn-lim_a0], 8, colors='black',linewidths=np.array([0.5]))
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
    plt.ylim([np.min(y)/2.0,0.])
    plt.title('RIGHT OUTCROP')
    
    
    
    # print temp[np.where(meep == np.max(meep))]


    varStep = psi[lim_u:lim_o,lim_a0:lim_b0]# - dupe[lim_u:lim_o,lim_a0:lim_b0]
    varMat = varStep

    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    ax1=fig.add_subplot(2,2,3, aspect=aspSQ*2.0,frameon=False)
    pGlass = plt.pcolor(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep, cmap=cm.rainbow)
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    plt.xlim([lim_a,lim_b])
    plt.ylim([np.min(y)/2.0,0.])
    
    varStep = psi[lim_u:lim_o,xn-lim_b0:xn-lim_a0]# - dupe[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    varMat = varStep

    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    ax1=fig.add_subplot(2,2,4, aspect=aspSQ*2.0,frameon=False)
    pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y[lim_u:lim_o], varStep, cmap=cm.rainbow)
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))

    plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
    plt.ylim([np.min(y)/2.0,0.])
    #plt.ylim([-1800.0,0.0])
    
    # ax1=fig.add_subplot(1,1,1, aspect=1.0,frameon=False)
    # plt.xticks([])
    # plt.yticks([])
    # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    # cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
      
    plt.savefig(outpath+'jdfZoomVel_'+str(i+restart)+'.png')
    
    
    #############################
    ##    ZOOM OUTCROP PLOT    ##
    #############################
    
    fig=plt.figure()
    
    varMat = u[:,lim_a0:lim_b0]
    varStep = u[:,lim_a0:lim_b0]
    
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    ax1=fig.add_subplot(2,2,1, aspect=aspSQ*2.0,frameon=False)
    pGlass = plt.pcolor(x[lim_a0:lim_b0], y, varStep, cmap=cm.rainbow)
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    #CS = plt.contour(xg[:,lim_a0:lim_b0], yg[:,lim_a0:lim_b0], psi[:,lim_a0:lim_b0], 8, colors='black',linewidths=np.array([0.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    plt.xlim([lim_a,lim_b])
    plt.ylim([np.min(y)/2.0,0.])
    plt.title('LEFT OUTCROP')
    
    
    varMat = u[:,xn-lim_b0:xn-lim_a0]
    varStep = u[:,xn-lim_b0:xn-lim_a0]

    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    ax1=fig.add_subplot(2,2,2, aspect=aspSQ*2.0,frameon=False)
    pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y, varStep,cmap=cm.rainbow)
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    #CS = plt.contour(xg[:,xn-lim_b0:xn-lim_a0], yg[:,xn-lim_b0:xn-lim_a0], psi[:,xn-lim_b0:xn-lim_a0], 8, colors='black',linewidths=np.array([0.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
    plt.ylim([np.min(y)/2.0,0.])
    plt.title('RIGHT OUTCROP')
    


    varMat = v[:,lim_a0:lim_b0]
    varStep = v[:,lim_a0:lim_b0]
    
    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    #contours = np.linspace(0.0,10.0,20)
    ax1=fig.add_subplot(2,2,3, aspect=aspSQ*2.0,frameon=False)
    pGlass = plt.pcolor(x[lim_a0:lim_b0], y, varStep, cmap=cm.rainbow)
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    #CS = plt.contour(xg[:,lim_a0:lim_b0], yg[:,lim_a0:lim_b0], psi[:,lim_a0:lim_b0], 8, colors='black',linewidths=np.array([0.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    plt.xlim([lim_a,lim_b])
    plt.ylim([np.min(y)/2.0,0.])
    plt.title('LEFT OUTCROP')
    
    
    varMat = v[:,xn-lim_b0:xn-lim_a0]
    varStep = v[:,xn-lim_b0:xn-lim_a0]

    contours = np.linspace(np.min(varMat),np.max(varMat),20)
    ax1=fig.add_subplot(2,2,4, aspect=aspSQ*2.0,frameon=False)
    pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y, varStep,cmap=cm.rainbow)
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    #CS = plt.contour(xg[:,xn-lim_b0:xn-lim_a0], yg[:,xn-lim_b0:xn-lim_a0], psi[:,xn-lim_b0:xn-lim_a0], 8, colors='black',linewidths=np.array([0.5]))
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    # plt.ylim([-2000.0,0.0])
    plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
    plt.ylim([np.min(y)/2.0,0.])
    plt.title('RIGHT OUTCROP')
    

    
    # ax1=fig.add_subplot(1,1,1, aspect=1.0,frameon=False)
    # plt.xticks([])
    # plt.yticks([])
    # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    # cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
      
    plt.savefig(outpath+'jdfZoomOC_'+str(i+restart)+'.png')
    
    
    
    #


    #####################
    ##    14C PLOT     ##
    #####################

    fig=plt.figure()

    varMat = c14#mask+maskP
    varStep = c14#mask+maskP

    ax1=fig.add_subplot(2,1,1,aspect=asp*2.0,frameon=False)
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    pGlass = plt.pcolor(x, y, varStep,cmap=cm.rainbow)
    plt.colorbar(orientation='horizontal')
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([0.5]))
    plt.ylim([np.min(y)/2, 0.0])
    plt.title('with decay')
    #

    # ax1=fig.add_subplot(2,1,1,aspect=asp,frameon=False)
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    # pGlass = plt.contourf(x, y, varStep,4)
    # plt.colorbar(orientation='horizontal')


    varMat = control#mask+maskP
    varStep = control#mask+maskP

    ax1=fig.add_subplot(2,1,2,aspect=asp*2.0,frameon=False)
    cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    pGlass = plt.pcolor(x, y, varStep,cmap=cm.rainbow)
    plt.colorbar(orientation='horizontal')
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([0.5]))
    plt.ylim([np.min(y)/2, 0.0])
    plt.title('without decay')
    #
    #
    # ax1=fig.add_subplot(2,1,2,aspect=asp,frameon=False)
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    # pGlass = plt.contourf(x, y, varStep, varStep,levels=[0.0,0.6,1.1])
    # plt.colorbar(orientation='horizontal')
    

    plt.xlim([np.min(x),np.max(x)])

    plt.savefig(outpath+'jdfx14c_'+str(i)+'.png')



    

    ##########################
    ##    PLOT HEAT FLUX    ##
    ##########################

    fig=plt.figure()

    hf_lith = np.zeros(len(x))
    hf_lith2 = np.zeros(len(x))
    hf_top = np.zeros(len(x))
    hf_lith = 1.8*(temp[0,:] - temp[1,:])/(y[1]-y[0])*1000.0
    
    for ii in range(len(x)):
       
        for j in range(len(y)):
            if mask[j,ii] == 25.0:
                hf_top[ii] = lambdaMat[j,ii]*(temp[j-1,ii] - temp[j,ii])/(y[1]-y[0])*1000.0
                hf_lith2[ii] = lambdaMat[j,ii]*(temp[0,ii] - temp[1,ii])/(y[1]-y[0])*1000.0
            if mask[j,ii] == 50.0:
                hf_top[ii] = lambdaMat[j,ii]*(temp[j-1,ii] - temp[j,ii])/(y[1]-y[0])*1000.0
                hf_lith2[ii] = lambdaMat[j,ii]*(temp[0,ii] - temp[1,ii])/(y[1]-y[0])*1000.0
            if mask[j,ii] == 12.5:
                hf_top[ii] = lambdaMat[j,ii]*(temp[j-1,ii] - temp[j,ii])/(y[1]-y[0])*1000.0
                hf_lith2[ii] = lambdaMat[j,ii]*(temp[0,ii] - temp[1,ii])/(y[1]-y[0])*1000.0
            if mask[j,ii] == 17.5:
                hf_top[ii] = lambdaMat[j,ii]*(temp[j-1,ii] - temp[j,ii])/(y[1]-y[0])*1000.0
                hf_lith2[ii] = lambdaMat[j,ii]*(temp[0,ii] - temp[1,ii])/(y[1]-y[0])*1000.0
                
                
                #*(1.0+np.abs(lambdaMat[j,ii+1]-lambdaMat[j,ii-1]))
                # if mask[j+1,ii] != 50.0:
                #     hf_top[ii] = hf_top[ii] + (lambdaMat[j,ii]*(temp[j+1,ii+1] - temp[j+1,ii])/(x[1]-x[0])*2000.0)
                # if mask[j-1,ii] != 50.0:
                #     hf_top[ii] = hf_top[ii] + (lambdaMat[j,ii]*(temp[j+1,ii] - temp[j+1,ii-1])/(x[1]-x[0])*2000.0)

    pl = plt.plot(x,hf_lith,'r')
    pl = plt.plot(x,hf_top,'b')
    pl = plt.plot(x,hf_lith2,'m',linewidth=0.5)
    #mpd.ripXdata=mpd.ripXdata*(5.0/26.0)
    pl = plt.scatter((5.0/5.0)*mpd.ripXdata,mpd.ripQdata)
    plt.ylim([0.0,800.0])
    
    #print hf_top


    plt.savefig(outpath+'jdfzzhf_'+str(i+restart)+'.png')

      
    # #######################
    # ##    CHEM 0 PLOT    ##
    # #######################
    #
    # fig=plt.figure()
    #
    # varMat = temp
    # varStep = temp
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    # ax1=fig.add_subplot(3,1,1, aspect=asp/1.0,frameon=False)
    # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    # p = plt.contour(xg,yg,perm,[-12.0,-13.5],colors='black',linewidths=np.array([1.0]))
    # CS = plt.contour(xg, yg, psi, 4, colors='black',linewidths=np.array([0.5]))
    # cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
    # plt.title('TEMPERATURE [$^{o}$C]')
    # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
    #
    # chemplot(ca0, ca, 3, 2, 3, 4, '[Ca]',0)
    # chemplot(dic0, dic, 3, 2, 4, 4, '[DIC]',0)
    # chemplot(glass0, glass, 3, 2, 5, 4, 'BASLATIC GLASS [mol]',1)
    # # chemplot(secMat[:,:,14], secStep[:,:,14], 4, 2, 5, 4, 'ANHYDRITE [mol]')
    # chemplot(secMat[:,:,16], secStep[:,:,16], 3, 2, 6, 4, 'CALCITE [mol]',0)
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
    # #######################
    # ##    CHEM 1 PLOT    ##
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
    # chemplot(secMat[:,:,6], secStep[:,:,6], 4, 2, 2, 4, 'CELADONITE [mol]',0)
    # chemplot(secMat[:,:,8], secStep[:,:,8], 4, 2, 3, 4, 'PYRITE [mol]',0)
    # chemplot(secMat[:,:,19], secStep[:,:,19], 4, 2, 4, 4, 'SAPONITE NA [mol]',0)
    # chemplot(secMat[:,:,20], secStep[:,:,20], 4, 2, 5, 4, 'NONTRONITE NA [mol]',0)
    # chemplot(secMat[:,:,21], secStep[:,:,21], 4, 2, 6, 4, 'NONTRONITE MG [mol]',0)
    # chemplot(secMat[:,:,25], secStep[:,:,25], 4, 2, 7, 4, 'MUSCOVITE [mol]',0)
    # chemplot(secMat[:,:,26], secStep[:,:,26], 4, 2, 8, 4, 'MESOLITE [mol]',0)
    #
    #
    # fig.set_tight_layout(True)
    # plt.savefig(outpath+'jdfChem1_'+str(i+restart)+'.png')
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
 