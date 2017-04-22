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
        contours = np.linspace(np.min(varMat[varMat>0]),np.max(varMat),20)
    if ditch==2:
        contours = np.linspace(np.min(varMat),np.max(varMat[varMat<varStep[bitsy-25,bitsx/2]])/5.0,20)
    ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp/1.0,frameon=False)
    print x.shape
    print y.shape
    print varStep.shape
    pGlass = plt.contourf(x,y,varStep,contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0,antialiased=True)
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
    # geo_cut = np.append(geo_cut, geo_cut[-1:,:], axis=0)
    # geo_cut = np.append(geo_cut, geo_cut[:,-1:], axis=1)
    #print geo_cut.shape
    return geo_cut
    

##############
# INITIALIZE #
##############

cell = 1
#steps = 400
steps = 20
minNum = 58

tsw = 3040
path = "output/revival/scope_law/"
#path_ex = "output/revival/clim_local/dic" + str(dic) + "_tsw" + str(tsw) + "/"
path_ex = "output/revival/scope_law/o600s02h200orhs200/"

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

param_l = np.array([100, 200, 300, 400, 500, 600, 700])
param_r = np.array([100, 200, 300, 400, 500, 600, 700])

param_h = np.array([200, 300, 400, 500])
param_s = ['02', '04', '06', '08', '10', '12', '14', '16', '18']

u_mean = np.zeros((len(param_h)+1,len(param_s)+1))
calcite_total = np.zeros((len(param_h)+1,len(param_s)+1))
glass_total = np.zeros((len(param_h)+1,len(param_s)+1))
pyrite_total = np.zeros((len(param_h)+1,len(param_s)+1))
saponite_total = np.zeros((len(param_h)+1,len(param_s)+1))

secMat = np.zeros([(bitsy-1),(bitsx-1)*steps,minNum+1])
secStep = np.zeros([bitsy,bitsx,minNum+1])

for i in range(len(param_h)):
    for k in range(len(param_s)):
        #if (param_l[i] >= param_r[k]):

        #sim_name =  "o" + str(param_l[i]) + "w1000h200orhs" + str(param_r[k])
        sim_name =  "o600s" + param_s[k] + "h" + str(param_h[i]) + "orhs200"
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
        glass0 = lambdaMat = np.loadtxt(path_sim + 'pri_glass.txt')
        kao0 = lambdaMat = np.loadtxt(path_sim + 'sec3.txt')
        sap0 = lambdaMat = np.loadtxt(path_sim + 'sec5.txt')
        pyrite0 = lambdaMat = np.loadtxt(path_sim + 'sec8.txt')
    
        step = 4
    
        h_last = cut(hMat0,step)
        u_last = cut(uMat0,step)
        perm_last = cut(permMat0,step)
        v_last = cut(vMat0,step)
        psi_last = cut(psiMat0,step)
        calcite_last = cut(calcite0,step)
        glass_last = cut(glass0,step)
        kao_last = cut(kao0,step)
        sap_last = cut(sap0,step)
        pyrite_last = cut(pyrite0,step)

    
        # u_mean[i,k] = np.sum(u_last)#np.sum(u_last[perm_last>-13])/np.count_nonzero(u_last[perm_last>-13])
        #
        # print "u_mean" , u_mean[i,k]
        # if math.isnan(u_mean[i,k]):
        #     u_mean[i,k] = u_mean[i-1,k]
    
        calcite_total[i,k] = np.sum(calcite_last)
        glass_total[i,k] = np.sum(glass_last)/(4.96*160.0*80.0)
        pyrite_total[i,k] = np.sum(pyrite_last)
        saponite_total[i,k] = np.sum(sap_last)

        ##############################
        ##    PLOT LAST TIMESTEP    ##
        ##############################
        fig=plt.figure()

        varMat = hMat0
        varStep = h_last
        contours = np.linspace(np.min(varStep),np.max(varMat),12)
        ax1=fig.add_subplot(3,2,1, aspect=asp,frameon=False)
        pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
        cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
        cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
        p = plt.contour(xg,yg,perm_last,[-13.15,-17.25],colors='black',linewidths=np.array([2.0]))
        CS = plt.contour(xg, yg, psi_last, 5, colors='black',linewidths=np.array([0.5]))
        cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)

        chemplot(calcite0, calcite_last, 3, 2, 2, 4, 'CALCITE [mol]',0)
        
        chemplot(glass0, glass_last, 3, 2, 3, 4, 'REMAINING BASALT [mol]',1)
        
        chemplot(kao0, kao_last, 3, 2, 4, 4, 'KAOLINITE [mol]',0)
        
        chemplot(sap0, sap_last, 3, 2, 5, 4, 'SAPONITE [mol]',0)
        
        chemplot(pyrite0, pyrite_last, 3, 2, 6, 4, 'PYRITE [mol]',0)

        plt.savefig(path+sim_name+'.png')
        
        
        # ##############################
        # ##    PLOT LAST TIMESTEP    ##
        # ##############################
        # fig=plt.figure()
        #
        # chemplot(glass0, glass_last, 1, 1, 1, 4, 'REMAINING BASALT [mol]',1)
        #
        #
        # plt.savefig(path+sim_name+'field.eps')
        #
        
        
        ##########################
        ##    PLOT HEAT FLUX    ##
        ##########################
        #
        # fig=plt.figure()
        #
        # hf_lith = np.zeros(len(x))
        # hf_top = np.zeros(len(x))
        # hf_top = -1.8*(h_last[-1,:] - h_last[-2,:])/(y[1]-y[0])*1000.0
        # #print hf_top
        # hf_lith = -1.8*(h_last[0,:] - h_last[1,:])/(y[1]-y[0])*1000.0
        # #print hf_lith
        #
        # pl = plt.plot(x,hf_lith)
        # pl = plt.plot(x,hf_top)
        # plt.ylim([-500.0,500.0])
        #
        #
        # plt.savefig(path+sim_name+'_hf.png')


######################
##    FIRST PLOT    ##
######################

fig=plt.figure()

x_param = param_h
param_s = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
y_param = param_s

# print "u_mean"
# print u_mean

x_shift = x_param[1]-x_param[0]
y_shift = y_param[1]-y_param[0]

x_param = np.append(x_param, x_param[-1]+x_shift)
y_param = np.append(y_param, y_param[-1]+y_shift)
print x_param.shape
print y_param.shape

asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))
print asp_multi

ax1=fig.add_subplot(2,2,1, aspect=asp_multi)

pCol = plt.pcolor(x_param, y_param, np.transpose(calcite_total/np.max(calcite_total)),cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('PARAM L', fontsize=8)
plt.ylabel('PARAM R', fontsize=8)
plt.title('PRECIPITATED CALCITE', fontsize=10)
plt.colorbar(pCol, orientation='vertical')



ax1=fig.add_subplot(2,2,2, aspect=asp_multi)

pCol = plt.pcolor(x_param, y_param, np.transpose(glass_total/np.max(glass_total)),cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('PARAM L', fontsize=8)
plt.ylabel('PARAM R', fontsize=8)
plt.title('REMAINING BASALT', fontsize=10)
plt.colorbar(pCol, orientation='vertical')


ax1=fig.add_subplot(2,2,3, aspect=asp_multi)

pCol = plt.pcolor(x_param, y_param, np.transpose(pyrite_total/np.max(pyrite_total)),cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('PARAM L', fontsize=8)
plt.ylabel('PARAM R', fontsize=8)
plt.title('PYRITE', fontsize=10)
plt.colorbar(pCol, orientation='vertical')


ax1=fig.add_subplot(2,2,4, aspect=asp_multi)

pCol = plt.pcolor(x_param, y_param, np.transpose(saponite_total/np.max(saponite_total)),cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('PARAM L', fontsize=8)
plt.ylabel('PARAM R', fontsize=8)
plt.title('SAPONITE', fontsize=10)
plt.colorbar(pCol, orientation='vertical')

fig.set_tight_layout(True)
plt.savefig(path+'pCol.png')







#######################
##    SECOND PLOT    ##
#######################

fig=plt.figure()

# x_param = param_l
# y_param = param_r
x_param = param_h
# for i in range(len(param_s)):
#     param_s[i] = int(param_s[i])
y_param = param_s

x_param = x_param[:-1]
y_param = y_param[:-1]

x_shift = x_param[1]-x_param[0]
y_shift = y_param[1]-y_param[0]



x_param = np.append(x_param, x_param[-1]+x_shift)
y_param = np.append(y_param, y_param[-1]+y_shift)

asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))


ax1=fig.add_subplot(2,2,1, aspect=asp_multi)
x_shift = 0.0
y_shift = 0.0
print x_param
print y_param

pCol = plt.contourf(x_param, y_param, np.transpose(calcite_total[:-1,:-1]/np.max(calcite_total[:-1,:-1])),20, cmap=cm.rainbow_r)
plt.xticks(x_param[::1])
plt.yticks(y_param[::1])
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('PARAM L', fontsize=8)
plt.ylabel('PARAM R', fontsize=8)
plt.title('CALCITE', fontsize=10)
plt.colorbar(pCol, orientation='vertical', ticks=np.linspace(0.0,1.0,21))


ax1=fig.add_subplot(2,2,2, aspect=asp_multi)
x_shift = 0.0
y_shift = 0.0
print x_param
print y_param

pCol = plt.contourf(x_param, y_param, np.transpose(glass_total[:-1,:-1]/np.max(glass_total[:-1,:-1])),20, cmap=cm.rainbow_r)
plt.xticks(x_param[::1])
plt.yticks(y_param[::1])
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('PARAM L', fontsize=8)
plt.ylabel('PARAM R', fontsize=8)
plt.title('BASALT', fontsize=10)
plt.colorbar(pCol, orientation='vertical', ticks=np.linspace(0.0,1.0,21))


ax1=fig.add_subplot(2,2,3, aspect=asp_multi)
x_shift = 0.0
y_shift = 0.0
print x_param
print y_param

pCol = plt.contourf(x_param, y_param, np.transpose(pyrite_total[:-1,:-1]/np.max(pyrite_total[:-1,:-1])),20, cmap=cm.rainbow_r)
plt.xticks(x_param[::1])
plt.yticks(y_param[::1])
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('PARAM L', fontsize=8)
plt.ylabel('PARAM R', fontsize=8)
plt.title('PYRITE', fontsize=10)
plt.colorbar(pCol, orientation='vertical', ticks=np.linspace(0.0,1.0,21))


ax1=fig.add_subplot(2,2,4, aspect=asp_multi)
x_shift = 0.0
y_shift = 0.0
print x_param
print y_param

pCol = plt.contourf(x_param, y_param, np.transpose(saponite_total[:-1,:-1]/np.max(saponite_total[:-1,:-1])),20, cmap=cm.rainbow_r)
plt.xticks(x_param[::1])
plt.yticks(y_param[::1])
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('PARAM L', fontsize=8)
plt.ylabel('PARAM R', fontsize=8)
plt.title('SAPONITE', fontsize=10)
plt.colorbar(pCol, orientation='vertical', ticks=np.linspace(0.0,1.0,21))



fig.set_tight_layout(True)
plt.savefig(path+'pContour.eps')

