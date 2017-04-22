# revived_JDF.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import math
import multiplot_data as mpd
from numpy.ma import masked_array
from mpl_toolkits.axes_grid1 import AxesGrid
import scipy.optimize
from scipy import interpolate
from numpy import array
from scipy.optimize import leastsq
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
# plt.rc('title', fontsize=8)
# plt.rc('ylabel', labelsize=8)
# plt.rc('xlabel', labelsize=8)


plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

    
    
def chemplot(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, ditch):
    if ditch==0:
        contours = np.linspace(np.min(varMat),np.max(varMat),20)
    if ditch==1:
        #contours = np.linspace(np.min(varMat[varMat>varStep[bitsy-25,bitsx/2]]),np.max(varMat),20)
        contours = np.linspace(np.min(varMat[varMat>0]),np.max(varMat),20)
        #contours = np.linspace(contours[0],contours[-1],20)
    if ditch==2:
        contours = np.linspace(np.min(varMat),np.max(varMat[varMat<varStep[bitsy-25,bitsx/2]])/5.0,20)
        #contours = np.linspace(contours[0],contours[-1],20)
    ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp/1.0,frameon=False)
    pGlass = plt.contourf(x,y,varStep[:-1,:],contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0,antialiased=True)
    p = plt.contour(xg,yg,perm_last,[-12.0,-13.5],colors='black',linewidths=np.array([1.0]))
    plt.xticks([])
    plt.yticks([])
    cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
    plt.title(cp_title)
    cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::contour_interval])
    fig.set_tight_layout(True)
    return chemplot
    
def cut(geo0,index):
    # geo_cut = geo0[(index*len(y0)/cell):(index*len(y0)/cell+len(y0)/cell),:]
    geo_cut = geo0[:,(index*len(x)):(index*len(x)+len(x))]
    geo_cut = np.append(geo_cut, geo_cut[-1:,:], axis=0)
    #geo_cut = np.append(geo_cut, geo_cut[:,-1:], axis=1)
    #print geo_cut.shape
    return geo_cut
    

##############
# INITIALIZE #
##############

cell = 1
#steps = 400
steps = 5
minNum = 58

tsw = 3040
path = "output/revival/clim_local/"
#path_ex = "output/revival/clim_local/dic" + str(dic) + "_tsw" + str(tsw) + "/"
path_ex = "output/revival/clim_local/dic300_tsw2750/"

# load output
x = np.loadtxt(path_ex + 'x.txt',delimiter='\n')
y = np.loadtxt(path_ex + 'y.txt',delimiter='\n')
asp = np.abs(np.max(x)/np.min(y))/4.0
xg, yg = np.meshgrid(x[:],y[:])

bitsx = len(x)
bitsy = len(y)


ripTrans = np.transpose(mpd.ripSort)
hf_interp = np.zeros(160)
hf_interp = np.interp(x,ripTrans[0,:],ripTrans[1,:])


xg, yg = np.meshgrid(x[:],y[:])

#param_h = np.array([150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
param_dic = np.array([200, 250, 300, 350, 400, 450, 500, 550, 600, 650])
param_tsw = np.array([2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100 ])

#param_dic = np.array([200, 300, 400, 500, 600])
#param_tsw = np.array([2750, 2800, 2850, 2900, 2950])

param_other = np.array([1.0])

calcite_total = np.zeros((len(param_dic)+1,len(param_tsw)+1))
glass_total = np.zeros((len(param_dic)+1,len(param_tsw)+1))
alk_total = np.zeros((len(param_dic)+1,len(param_tsw)+1))
ca_total = np.zeros((len(param_dic)+1,len(param_tsw)+1))
mp_total = np.zeros((len(param_dic)+1,len(param_tsw)+1))

secMat = np.zeros([(bitsy-1),(bitsx-1)*steps,minNum+1])
secStep = np.zeros([bitsy,bitsx,minNum+1])

for i in range(len(param_dic)):
    for k in range(len(param_tsw)):
    
        sim_name = "dic" + str(param_dic[i]) + "_tsw" + str(param_tsw[k])
        path_sim = path + sim_name + "/"
        
        print ""
        print sim_name
    
        # load stuff
        uMat0 = np.loadtxt(path_sim + 'uMat.txt')*(3.14e7)#*10.0
        vMat0 = np.loadtxt(path_sim + 'vMat.txt')*(3.14e7)#*10.0
        psiMat0 = np.loadtxt(path_sim + 'psiMat.txt')
        hMat0 = np.loadtxt(path_sim + 'hMat.txt')
        permMat0 = np.log10(np.loadtxt(path_sim + 'permMat.txt'))
        mask = np.loadtxt(path_sim + 'mask.txt')
        lambdaMat = np.loadtxt(path_sim + 'lambdaMat.txt')
        calcite0 = lambdaMat = np.loadtxt(path_sim + 'sec16.txt')
        
        dic0 = np.loadtxt(path_sim + 'sol_c.txt')
        ca0 = np.loadtxt(path_sim + 'sol_ca.txt')
        alk0 = np.loadtxt(path_sim + 'sol_alk.txt')
        glass0 = np.loadtxt(path_sim + 'pri_glass.txt')
        ph0 = np.loadtxt(path_sim + 'sol_ph.txt')
        
        step = 4
        dic = cut(dic0,step)
        
        glass = cut(glass0,step)
        glass_fit = glass[:-1,:]
        alk = cut(alk0,step)
        ca = cut(ca0,step)
        ca[:25,:] = 0.0
        ca[glass==0] = 0.0
        calcite_last = cut(calcite0,step)
        calcite_last[:25,:] = 0.0
        calcite_last[glass==0] = 0.0
        ph = cut(ph0,step)
        h_last = cut(hMat0,step)
        
        ca1 = cut(ca0,step-1)
        
        
        u_last = uMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
        v_last = vMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
        psi_last = psiMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
        #h_last = hMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
        perm_last = permMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
        

        calcite_total[i,k] = np.sum(calcite_last)*(5.1e14)*(.7)*(6.55e-8)*26000.0 - .0026*160.0*80*.0045
        calcite_total[i,k] = 2.0*calcite_total[i,k]/(5.0e6) 
        glass_total[i,k] = np.sum(glass[mask>0.0])
        alk_total[i,k] = np.sum(alk[mask>0.0])*5.1e14*(2.0/3.0)*(6.55e-8)*26000.0*(0.0000266*26000.0*2000.0)
        #alk_total[i,k] = np.sum(alk[mask>0.0])*5.1e14*(2.0/3.0)*(6.55e-8)*26000.0*(0.0000266*26000.0*2000.0)
        # - np.count_nonzero(mask)*.00245*5.1e14*(2.0/3.0)*(6.55e-8)*26000.0*(0.0000266*26000.0*2000.0)
        # - np.count_nonzero(mask)*.00245*.0266*5.1e14*(2.0/3.0)*(.00006)*162.5
        
        print mask.shape
        print glass.shape
        mask[:25,:] = 0.0
        mask[glass_fit==0] = 0.0
        
        ca_total[i,k] = 2308.0*np.sum(ca[mask>0.0])*5.1e14*(.7)*(6.55e-8)*26000.0#*(2.0*0.0000266*160.0*40.0) 
        print "a" , ca_total[i,k]
        ca_b =  2308.0*np.count_nonzero(mask)*.01028*5.1e14*(.7)*(6.55e-8)*26000.0#*(2.0*0.0000266*160.0*40.0)
        print "b" , ca_b
        ca_total[i,k] = ca_total[i,k] - ca_b
        print "c" , ca_total[i,k]
        mp_total[i,k] = alk_total[i,k] + 2.0*calcite_total[i,k]
        mp_total[i,k] = mp_total[i,k]/(5.0e6)
        ca_total[i,k] = 2.0*ca_total[i,k]/(5.0e6)
        print "calcite" , calcite_total[i,k]
        #print "glass" , glass_total[i,k]
        #print "alk" , alk_total[i,k]
        print "ca" , ca_total[i,k]
        #print "mp" , mp_total[i,k]
        
        
    
        ##############################
        ##    PLOT LAST TIMESTEP    ##
        ##############################
        fig=plt.figure()
    
        # varMat = hMat0
        # varStep = h_last
        # contours = np.linspace(np.min(varStep),np.max(varMat),12)
        # ax1=fig.add_subplot(3,1,1, aspect=asp,frameon=False)
        # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
        # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
        # cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
        # p = plt.contour(xg,yg,perm_last,[-13.15,-17.25],colors='black',linewidths=np.array([2.0]))
        # CS = plt.contour(xg, yg, psi_last, 5, colors='black',linewidths=np.array([0.5]))
        # cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)

        chemplot(hMat0, h_last, 3, 2, 1, 4, 'TEMP',0)
        chemplot(ph0, ph, 3, 2, 2, 4, 'pH',0)
        chemplot(ca0, ca, 3, 2, 3, 4, '[Ca]',0)
        chemplot(dic0, dic, 3, 2, 4, 4, '[DIC]',0)
        chemplot(glass0, glass, 3, 2, 5, 4, 'BASLATIC GLASS [mol]',1)
        chemplot(calcite0, calcite_last, 3, 2, 6, 4, 'CALCITE [mol]',0)

        plt.savefig(path+sim_name+'.png')
        #plt.savefig(path+'eps/'+sim_name+'_'+str(step)+'.eps')


######################
##    FIRST PLOT    ##
######################

fig=plt.figure()

x_param = param_dic/100000.0
y_param = param_tsw/10.0

print "calcite"
print calcite_total
print "glass"
print glass_total

x_shift = x_param[1]-x_param[0]
y_shift = y_param[1]-y_param[0]

x_param = np.append(x_param, x_param[-1]+x_shift)
y_param = np.append(y_param, y_param[-1]+y_shift)
print x_param.shape
print y_param.shape

asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))
print asp_multi

ax1=fig.add_subplot(2,2,1, aspect=asp_multi)

pCol = plt.pcolor(x_param, y_param, np.transpose(calcite_total),cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('DIC [mol/kgw]', fontsize=8)
plt.ylabel('T seawater [C]', fontsize=8)
plt.title('TOTAL CALCITE IN SYSTEM', fontsize=10)
plt.colorbar(pCol, orientation='vertical')


ax1=fig.add_subplot(2,2,2, aspect=asp_multi)

pCol = plt.pcolor(x_param, y_param, np.transpose(alk_total),cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('DIC [mol/kgw]', fontsize=8)
plt.ylabel('T seawater [C]', fontsize=8)
plt.title('TOTAL ALKALINITY IN SYSTEM', fontsize=10)
plt.colorbar(pCol, orientation='vertical')


ax1=fig.add_subplot(2,2,3, aspect=asp_multi)

pCol = plt.pcolor(x_param, y_param, np.transpose(glass_total),cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('DIC [mol/kgw]', fontsize=8)
plt.ylabel('T seawater [C]', fontsize=8)
plt.title('TOTAL REMAINING BASALT', fontsize=10)
plt.colorbar(pCol, orientation='vertical')


ax1=fig.add_subplot(2,2,4, aspect=asp_multi)

pCol = plt.pcolor(x_param, y_param, np.transpose(mp_total),cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('DIC [mol/kgw]', fontsize=8)
plt.ylabel('T seawater [C]', fontsize=8)
plt.title('TOTAL ALK FLUX OF PLANET', fontsize=10)
plt.colorbar(pCol, orientation='vertical')

fig.set_tight_layout(True)
plt.savefig(path+'pCol.png')


fig=plt.figure()

ax1=fig.add_subplot(1,1,1, aspect=asp_multi)

pCol = plt.pcolor(x_param, y_param, np.transpose(ca_total),cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('DIC [mol/kgw]', fontsize=12)
plt.ylabel('T seawater [C]', fontsize=12)
plt.title('GLOBAL ALKALINITY FLUX OF SEAFLOOR WEATHERING', fontsize=14)
plt.colorbar(pCol, orientation='vertical')

fig.set_tight_layout(True)
plt.savefig(path+'p_total_alk.png')







#######################
##    SECOND PLOT    ##
#######################

fig=plt.figure()

x_param = param_dic/100000.0
y_param = param_tsw/10.0

x_param = x_param[:-1]
y_param = y_param[:-1]

print "calcite"
print calcite_total
print "glass"
print glass_total

x_shift = x_param[1]-x_param[0]
y_shift = y_param[1]-y_param[0]



x_param = np.append(x_param, x_param[-1]+x_shift)
y_param = np.append(y_param, y_param[-1]+y_shift)
print x_param.shape
print y_param.shape
print np.transpose(calcite_total[:-1,:-1]).shape

asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))


ax1=fig.add_subplot(2,2,1, aspect=asp_multi)
x_shift = 0.0
y_shift = 0.0
print x_param
print y_param

pCol = plt.contourf(x_param, y_param, np.transpose(calcite_total[:-1,:-1]),20, cmap=cm.jet)
plt.xticks(x_param[::2])
plt.yticks(y_param[::2])
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('DIC [mol/kgw]', fontsize=8)
plt.ylabel('T seawater [C]', fontsize=8)
plt.title('TOTAL CALCITE IN SYSTEM', fontsize=10)
plt.colorbar(pCol, orientation='vertical')


ax1=fig.add_subplot(2,2,2, aspect=asp_multi)

pCol = plt.contourf(x_param, y_param, np.transpose(alk_total[:-1,:-1]),20,cmap=cm.jet)
plt.xticks(x_param[::2])
plt.yticks(y_param[::2])
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('DIC [mol/kgw]', fontsize=8)
plt.ylabel('T seawater [C]', fontsize=8)
plt.title('TOTAL ALKALINITY IN SYSTEM', fontsize=10)
plt.colorbar(pCol, orientation='vertical')


# ax1=fig.add_subplot(2,2,3, aspect=asp_multi)
#
# pCol = plt.contourf(x_param, y_param, np.transpose(glass_total[:-1,:-1]),20,cmap=cm.jet)
# plt.xticks(x_param[::2])
# plt.yticks(y_param[::2])
# plt.xlim([np.min(x_param), np.max(x_param)])
# plt.ylim([np.min(y_param), np.max(y_param)])
# plt.xlabel('DIC [mol/kgw]', fontsize=8)
# plt.ylabel('T seawater [C]', fontsize=8)
# plt.title('TOTAL REMAINING BASALT', fontsize=10)
# plt.colorbar(pCol, orientation='vertical')
#

ax1=fig.add_subplot(2,2,4, aspect=asp_multi)

pCol = plt.contourf(x_param, y_param, np.transpose(ca_total[:-1,:-1]),20,cmap=cm.jet)
plt.xticks(x_param[::2])
plt.yticks(y_param[::2])
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('DIC [mol/kgw]', fontsize=8)
plt.ylabel('T seawater [C]', fontsize=8)
plt.title('GLOBAL ALKALINITY FLUX OF SEAFLOOR WEATHERING', fontsize=10)
plt.colorbar(pCol, orientation='vertical')




fig.set_tight_layout(True)
plt.savefig(path+'pContour.png')







fig=plt.figure()

x_param = x_param*10000.0
y_param = y_param/10.0
x_p = x_param#!np.linspace(np.min(x_param),np.max(x_param),101)
y_p = y_param#np.linspace(np.min(y_param),np.max(y_param),101)
print x_p
print y_p
xx, yy = np.meshgrid(x_p, y_p)
ca_total = ca_total/(1.0e10)
z = np.transpose(ca_total[:-1,:-1])
f = interpolate.interp2d(x_p, y_p, z)

asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))

def func(X, a, b, c, d):
    x_p,y_p = X
    g = a*np.exp(b*x_p) + c*np.exp(d*y_p)
    return g.ravel()
x_p0 = np.linspace(np.min(x_param),np.max(x_param),101)
y_p0 = np.linspace(np.min(y_param),np.max(y_param),101)
x_p, y_p = np.meshgrid(x_p0, y_p0)
a, b, c, d = 10., 4., 6., 2.
znew = f(x_p0,y_p0) #np.transpose(calcite_total[:-1,:-1])
znew = znew.ravel()
print znew.shape
print x_p.shape
print y_p.shape
#print z.shape

popt, pcov= scipy.optimize.curve_fit(func, (x_p,y_p), znew)
print scipy.optimize.curve_fit(func, (x_p,y_p), znew)

data_fitted = func((x_p,y_p), *popt)
ax1=fig.add_subplot(2,2,3, aspect=asp_multi)
pCol = plt.contourf(x_p, y_p, data_fitted.reshape(len(x_p0), len(y_p0)), 20)


plt.xticks(x_param[::2])
plt.yticks(y_param[::2])
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('DIC [mol/kgw]', fontsize=8)
plt.ylabel('T seawater [C]', fontsize=8)
plt.title('CALCITE 2-VARIABLE CURVE FIT', fontsize=10)
plt.colorbar(pCol, orientation='vertical')



fig.set_tight_layout(True)
plt.savefig(path+'pFit.png')





fig=plt.figure()


ax1=fig.add_subplot(1,1,1, aspect=asp_multi)

pCol = plt.contourf(x_param, y_param, np.transpose(ca_total[:-1,:-1]),20,cmap=cm.jet)
plt.xticks(x_param[::2])
plt.yticks(y_param[::2])
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('DIC [mol/kgw]', fontsize=12)
plt.ylabel('T seawater [C]', fontsize=12)
plt.title('GLOBAL ALKALINITY FLUX FROM SEAFLOOR WEATHERING [eq/yr]', fontsize=12)
plt.colorbar(pCol, orientation='vertical')


fig.set_tight_layout(True)
plt.savefig(path+'p_global_contours.png')

#plt.savefig(path+'pFit.png')


#
# ######################
# ##    SECOND PLOT    ##
# ######################
#
# fig=plt.figure()
#
# x_param = param_orhs
# y_param = param_h
#
# u_mean2d = u_mean[:-1,0,0,:-1]
# # u_mean2d = np.insert(u_mean2d, 0, u_mean2d[:,0], axis=1)
# # u_mean2d = np.insert(u_mean2d, 0, u_mean2d[0,:], axis=0)
#
#
#
# x_shift = x_param[1]-x_param[0]
# y_shift = y_param[1]-y_param[0]
# print 'umean'
# print u_mean2d
# print 'umax'
# print u_max[:,0,0,:]
#
#
# # x_param = np.insert(x_param, 0, x_param[0]-x_shift)
# # y_param = np.insert(y_param, 0, y_param[0]-y_shift)
# print x_param
# print y_param
# print u_mean[:,0,0,:].shape
# X, Y = np.meshgrid(x_param,y_param)
#
# print x_param.shape
# print y_param.shape
# print u_mean2d.shape
# print X.shape
# print Y.shape
#
# asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))
#
# ax1=fig.add_subplot(2,2,1, aspect=asp_multi)
# pCol = plt.contourf(X, Y, u_mean2d, 15, cmap=shifted_cmap)
# plt.xticks(x_param)
# plt.yticks(y_param)
# plt.xlim([np.min(x_param), np.max(x_param)])
# plt.ylim([np.min(y_param), np.max(y_param)])
# plt.xlabel('outflow outcrop height [m]', fontsize=10)
# plt.ylabel('flow layer thickness [m]', fontsize=10)
# plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
# cax = fig.add_axes([0.44, 0.53, 0.02, 0.41])
# fig.colorbar(pCol, cax=cax, orientation='vertical')
#
#
#
#
# ax1=fig.add_subplot(2,2,2, aspect=asp_multi)
#
#
# hf_2d = hf_range[:-1,0,0,:-1]
#
# pCol = plt.contourf(x_param, y_param, hf_2d*100.0, 15)
# plt.xticks(x_param)
# plt.yticks(y_param)
# plt.xlim([np.min(x_param), np.max(x_param)])
# plt.ylim([np.min(y_param), np.max(y_param)])
# plt.xlabel('outflow outcrop height [m]', fontsize=10)
# plt.ylabel('flow layer thickness [m]', fontsize=10)
# plt.title('total heat %% dissipated by fluid flow', fontsize=10)
# cax = fig.add_axes([0.915, 0.53, 0.02, 0.41])
# fig.colorbar(pCol,cax=cax, ticks=[30.0, 40.0, 50.0, 60.0, 70.0])
#
#
#
#
#
# fig.set_tight_layout(True)
# plt.savefig(path+'pCont.eps')




# #######################
# ##    SECOND PLOT    ##
# #######################
#
#
# fig=plt.figure()
#
# x_param = param_orhs
# y_param = param_h
#
# x_shift = x_param[1]-x_param[0]
# y_shift = y_param[1]-y_param[0]
# print 'umean'
# print u_mean[:,0,0,:]
# print 'umax'
# print u_max[:,0,0,:]
#
# x_param = np.append(x_param, x_param[-1]+x_shift)
# y_param = np.append(y_param, y_param[-1]+y_shift)
# print x_param.shape
# print y_param.shape
# print u_mean[:,0,0,:].shape
#
# asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))
# col_min = np.min(u_mean)
# col_max = 3.0#np.max(u_max)
#
# ax1=fig.add_subplot(2,2,1, aspect=asp_multi)
#
#
# pCol = plt.contour(x_param, y_param, u_mean[:,0,0,:],5)
# plt.clabel(pCol, fontsize=9, inline=1)
# plt.xticks(x_param+x_shift/2.0, x_param)
# plt.yticks(y_param+y_shift/2.0, y_param)
# plt.xlim([np.min(x_param), np.max(x_param)])
# plt.ylim([np.min(y_param), np.max(y_param)])
# plt.xlabel('outflow outcrop height [m]', fontsize=10)
# plt.ylabel('flow layer thickness [m]', fontsize=10)
# plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
#
#
# ax1=fig.add_subplot(2,2,2, aspect=asp_multi)
#
# pCol = plt.contour(x_param, y_param, u_max[:,0,0,:],5)
# plt.clabel(pCol, fontsize=9, inline=1)
# plt.xticks(x_param+x_shift/2.0, x_param)
# plt.yticks(y_param+y_shift/2.0, y_param)
# plt.xlim([np.min(x_param), np.max(x_param)])
# plt.ylim([np.min(y_param), np.max(y_param)])
# plt.xlabel('outflow outcrop height [m]', fontsize=10)
# plt.ylabel('flow layer thickness [m]', fontsize=10)
# plt.title('max lateral flow velocity [m/yr]', fontsize=10)
#
#
#
#
#
# fig.set_tight_layout(True)
# plt.savefig(path+'pCont.png')
#
#
#
